"""Fixed-validation learning curve using whole training location groups. Never opens test images."""
import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, default=Path('dataset_dev'))
    parser.add_argument('--pipeline', type=Path, default=Path('models/final/pipeline.joblib'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/learning_curve_final'))
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 2:
        raise ValueError('Use at least two repeats.')
    if args.output_dir.exists():
        raise FileExistsError('Output exists; choose a new --output-dir to preserve previous results.')
    with (args.dataset_dir / 'images.csv').open(encoding='utf-8-sig', newline='') as f:
        records = list(csv.DictReader(f))
    subsets = {s: [r for r in records if r['split'] == s] for s in ('train', 'validation')}
    group_sets = {s: {r['group_id'] for r in records if r['split'] == s}
                  for s in ('train', 'validation', 'test')}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not group_sets[a] & group_sets[b], 'Location-group leakage detected.'
    ys = {s: np.array([int(r['label']) for r in subsets[s]]) for s in subsets}
    assert all(set(y) == {0, 1} for y in ys.values()), 'Both classes required.'
    saved = joblib.load(args.pipeline)
    feature = clone(saved.named_steps['features'])
    paths = {}
    root = args.dataset_dir.resolve()
    for split in subsets:
        paths[split] = [(root / r['relative_path']).resolve() for r in subsets[split]]
        assert all(x.is_relative_to(root) for x in paths[split]), 'Unsafe image path.'
    feature.fit(paths['train'], ys['train'])
    x = {s: feature.transform(paths[s]) for s in subsets}
    # This project's feature extractor is stateless; scaler and estimator are refitted below.
    head = Pipeline([(name, clone(step)) for name, step in saved.steps[1:]])
    groups = defaultdict(list)
    for i, r in enumerate(subsets['train']):
        groups[r['group_id']].append(i)
    by_class = {c: sorted(g for g, idx in groups.items() if set(ys['train'][idx]) == {c}) for c in (0, 1)}
    assert sum(map(len, by_class.values())) == len(groups), 'Mixed-label group.'
    sizes = sorted(set((max(1, int(np.ceil(len(by_class[0])*f))),
                        max(1, int(np.ceil(len(by_class[1])*f))))
                       for f in (.2, .4, .6, .8, 1.0)))
    results = []
    for repeat in range(args.repeats):
        rng = np.random.default_rng(42 + repeat)
        ordered = {c: rng.permutation(by_class[c]).tolist() for c in (0, 1)}
        for n0, n1 in sizes:
            chosen = ordered[0][:n0] + ordered[1][:n1]
            idx = np.array(sorted(i for g in chosen for i in groups[g]))
            model = clone(head)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                model.fit(x['train'][idx], ys['train'][idx])
            row = dict(repeat=repeat, groups=len(chosen), outside_groups=n0,
                       inside_groups=n1, train_images=len(idx),
                       convergence_warning=any(issubclass(w.category, ConvergenceWarning) for w in caught))
            for split, matrix, target in [('train', x['train'][idx], ys['train'][idx]),
                                           ('validation', x['validation'], ys['validation'])]:
                prediction = model.predict(matrix)
                row[split+'_accuracy'] = accuracy_score(target, prediction)
                row[split+'_macro_f1'] = f1_score(target, prediction, labels=[0,1], average='macro', zero_division=0)
            results.append(row)
        print(f'Completed repeat {repeat+1}/{args.repeats}', flush=True)
    args.output_dir.mkdir(parents=True)
    def write_csv(name, rows):
        with (args.output_dir/name).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_csv('runs.csv', results)
    summary = []
    for n0,n1 in sizes:
        selected = [r for r in results if r['outside_groups']==n0 and r['inside_groups']==n1]
        row = dict(groups=n0+n1, outside_groups=n0, inside_groups=n1)
        for key in ('train_images','train_accuracy','validation_accuracy','train_macro_f1','validation_macro_f1'):
            values = [r[key] for r in selected]
            row[key+'_mean'] = float(np.mean(values)); row[key+'_std'] = float(np.std(values, ddof=1))
        summary.append(row)
    write_csv('summary.csv', summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(12,4.8))
    for ax,metric,title in zip(axes,('accuracy','macro_f1'),('Accuracy','Macro F1')):
        xx = np.array([r['groups'] for r in summary])
        for split,color in [('train','tab:blue'),('validation','tab:orange')]:
            mean = np.array([r[split+'_'+metric+'_mean'] for r in summary])
            sd = np.array([r[split+'_'+metric+'_std'] for r in summary])
            ax.plot(xx,mean,'o-',color=color,label=split.capitalize())
            ax.fill_between(xx,np.maximum(0,mean-sd),np.minimum(1,mean+sd),color=color,alpha=.15)
        ax.set(xlabel='Whole training location groups',ylabel=title,ylim=(0,1.03),xticks=xx)
        ax.grid(alpha=.25); ax.legend()
    fig.suptitle('Fixed-validation learning curve: saved model settings\nBands: +/- 1 SD over training-group subsets (not confidence intervals)')
    fig.tight_layout()
    fig.savefig(args.output_dir/'learning_curve.png',dpi=180)
    fig.savefig(args.output_dir/'learning_curve.pdf')
    plt.close(fig)
    info = dict(pipeline=str(args.pipeline), repeats=args.repeats,
                validation_images=len(ys['validation']), test_images_opened=0,
                protocol='Nested class-stratified whole-group train subsets; fixed original validation; scaler refit per subset',
                estimator_params=saved.named_steps['classifier'].get_params(),
                convergence_warning_runs=sum(r['convergence_warning'] for r in results))
    (args.output_dir/'protocol.json').write_text(json.dumps(info,indent=2,default=str),encoding='utf-8')
    print('Saved plot, raw runs, summary and protocol to',args.output_dir)
    print('Final point:',summary[-1])

if __name__ == '__main__':
    main()
