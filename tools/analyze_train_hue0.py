#!/usr/bin/env python3
"""Analyze a saved binary image pipeline and compare two predeclared retraining arms.

Run from the project root; place this file in tools/. Never opens test images.
Only load trusted joblib files. Candidate selection uses training-group CV only.
"""
import argparse
import hashlib
import json
import platform
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

HERE = Path(__file__).resolve().parent
for root in (Path.cwd(), HERE.parent):
    if (root / 'src').is_dir():
        sys.path.insert(0, str(root))
        break


def write_json(path, data):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=convert), encoding='utf-8')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y, pred):
    return dict(accuracy=float(accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, labels=[0, 1], average='macro', zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist(),
                per_class=classification_report(y, pred, labels=[0, 1],
                    target_names=['outside', 'inside'], output_dict=True, zero_division=0))


def linear_score(z, classifier):
    return np.asarray(z) @ classifier.coef_[0] + classifier.intercept_[0]


def predict_scores(score):
    return (np.asarray(score) > 0).astype(int)


def numerical_fit(raw, y, template, keep):
    scaler = clone(template.named_steps['scale'])
    clf = clone(template.named_steps['classifier'])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        z = scaler.fit_transform(raw[:, keep], y)
        clf.fit(z, y)
    if not np.array_equal(clf.classes_, [0, 1]):
        raise ValueError('Expected classes [0,1].')
    if clf.coef_.shape != (1, int(keep.sum())):
        raise ValueError('Expected a binary linear classifier.')
    warning_rows = [dict(category=w.category.__name__, message=str(w.message)) for w in caught]
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return scaler, clf, converged, warning_rows


def feature_table(ext, dimension):
    from src.features import feature_block_sizes
    fc, pc = ext.feature_config, ext.preprocessing_config
    sizes = feature_block_sizes(fc, pc)
    if sum(sizes.values()) != dimension:
        raise ValueError('Feature dimensions do not match project configuration.')
    size = pc.image_size
    if not isinstance(size, (int, np.integer)):
        raise ValueError('This script expects a square integer image_size.')
    h = w = int(size)
    rows, offset = [], 0
    for block, count in sizes.items():
        for local in range(count):
            r = dict(feature_index=offset+local, block=block, local_index=local,
                     description='', x0=None, y0=None, x1=None, y1=None)
            if block == 'color':
                tile, rem = divmod(local, 3 * fc.color_bins)
                channel, bin_id = divmod(rem, fc.color_bins)
                ty, tx = divmod(tile, fc.color_grid)
                xe = np.cumsum([0] + [len(x) for x in np.array_split(np.arange(w), fc.color_grid)])
                ye = np.cumsum([0] + [len(x) for x in np.array_split(np.arange(h), fc.color_grid)])
                r.update(x0=int(xe[tx]), y0=int(ye[ty]), x1=int(xe[tx+1]), y1=int(ye[ty+1]),
                         description=f'HSV {"HSV"[channel]} bin {bin_id}; range {bin_id/fc.color_bins:.6f}..{(bin_id+1)/fc.color_bins:.6f}; tile ({ty},{tx})')
            elif block == 'hog':
                c, b, o = fc.hog_pixels_per_cell, fc.hog_cells_per_block, fc.hog_orientations
                shape = (h//c-b+1, w//c-b+1, b, b, o)
                by, bx, cy, cx, k = map(int, np.unravel_index(local, shape))
                x, y0 = (bx+cx)*c, (by+cy)*c
                r.update(x0=x, y0=y0, x1=x+c, y1=y0+c,
                         description=f'block ({by},{bx}); cell ({cy},{cx}); gradient {180*k/o:.3f}..{180*(k+1)/o:.3f} deg')
            elif block == 'lbp':
                r['description'] = f'Global uniform LBP histogram bin {local}; not spatially localized'
            rows.append(r)
        offset += count
    return pd.DataFrame(rows)


def load_records(dataset):
    frame = pd.read_csv(dataset/'images.csv', dtype=str, keep_default_na=False)
    required = {'image_id', 'relative_path', 'label', 'group_id', 'split'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing columns: {required-set(frame.columns)}')
    if frame.empty or (frame[list(required)] == '').any().any():
        raise ValueError('Empty dataset or blank required fields.')
    if not set(frame.split).issubset({'train', 'validation', 'test'}):
        raise ValueError('Unexpected split name.')
    labels = frame.label.str.strip().str.lower().map({'0':0, '1':1, 'outside':0, 'inside':1})
    if labels.isna().any():
        raise ValueError('Unexpected label.')
    frame['y'] = labels.astype(int)
    for col in ('image_id', 'relative_path'):
        if frame[col].duplicated().any():
            raise ValueError(f'Duplicate {col}.')
    if (frame.groupby('group_id').split.nunique() > 1).any():
        raise ValueError('Location group crosses splits.')
    if (frame.groupby('group_id').y.nunique() > 1).any():
        raise ValueError('Location group crosses classes.')
    for split in ('train', 'validation'):
        if set(frame.loc[frame.split == split, 'y']) != {0, 1}:
            raise ValueError(f'{split} must contain both classes.')
    locations = dataset/'locations.csv'
    if not locations.exists():
        raise FileNotFoundError('locations.csv required for boundary checks.')
    loc = pd.read_csv(locations, dtype=str, keep_default_na=False)
    if not {'group_id','class_name','boundary_verified'}.issubset(loc.columns):
        raise ValueError('locations.csv needs group_id,class_name,boundary_verified.')
    if loc.group_id.duplicated().any():
        raise ValueError('Duplicate location record.')
    lookup = loc.set_index('group_id')
    for group, part in frame.groupby('group_id'):
        if group not in lookup.index:
            raise ValueError(f'Missing location record: {group}')
        row = lookup.loc[group]
        expected = 'inside' if part.y.iloc[0] else 'outside'
        if row.class_name.strip().lower() != expected:
            raise ValueError(f'Location label mismatch: {group}')
        if row.boundary_verified.strip().lower() not in {'yes','true','1'}:
            raise ValueError(f'Boundary not verified: {group}; correct the record after checking, not blindly.')
    return frame


def paths_and_hashes(frame, dataset):
    paths, hashes = [], []
    for row in frame.itertuples():
        p = (dataset/row.relative_path).resolve()
        if not p.is_relative_to(dataset) or not p.is_file():
            raise ValueError(f'Invalid image path: {p}')
        paths.append(p)
        hashes.append(sha256(p))
    tmp = frame[['split','group_id']].copy()
    tmp['sha256'] = hashes
    for _, part in tmp.groupby('sha256'):
        if part.group_id.nunique() > 1 or part.split.nunique() > 1:
            raise ValueError('Exact duplicate image bytes cross groups/splits in development data.')
    return paths, hashes


def export_predictions(name, frame, pred, score, output):
    table = frame[['image_id','relative_path','group_id','split','y']].copy()
    table['prediction'] = np.asarray(pred, dtype=int)
    table['score'] = score
    table['correct'] = table.y == table.prediction
    table.to_csv(output/f'{name}_predictions.csv', index=False)
    table.groupby(['split','group_id','y']).agg(
        total=('correct','size'), correct=('correct','sum'), accuracy=('correct','mean')
    ).reset_index().to_csv(output/f'{name}_groups.csv', index=False)
    result = {}
    for split, part in table.groupby('split'):
        result[split] = metrics(part.y, part.prediction)
    write_json(output/f'{name}_metrics.json', result)
    return result


def export_coefficients(name, scaler, clf, keep, mapping, output, eps):
    table = mapping.copy()
    table['included'] = keep
    weights = np.zeros(len(keep))
    weights[keep] = clf.coef_[0]
    table['weight'] = weights
    table['active'] = keep & (np.abs(weights) > eps)
    table.to_csv(output/f'{name}_coefficients.csv', index=False)
    table[table.active].to_csv(output/f'{name}_active_features.csv', index=False)


def saved_analysis(model, raw, paths, dev, mapping, output, eps):
    scaler, clf = model.named_steps['scale'], model.named_steps['classifier']
    z = scaler.transform(raw)
    contributions = z * clf.coef_[0]
    score = np.asarray(model.decision_function(paths)).ravel()
    if not np.allclose(linear_score(z, clf), score, atol=1e-6):
        raise ValueError('Contribution sum disagrees with pipeline score.')
    pred = np.asarray(model.predict(paths)).astype(int)
    results = {'saved_original': export_predictions('saved_original', dev, pred, score, output)}
    export_coefficients('saved_original', scaler, clf, np.ones(raw.shape[1],bool), mapping, output, eps)
    active = np.flatnonzero(np.abs(clf.coef_[0]) > eps)
    pieces = []
    for j in active:
        t = dev[['image_id','group_id','split','y']].copy()
        t['feature_index'] = j
        t['raw_value'] = raw[:,j]
        t['standardized_value'] = z[:,j]
        t['weight'] = clf.coef_[0,j]
        t['contribution'] = contributions[:,j]
        pieces.append(t)
    if pieces:
        pd.concat(pieces, ignore_index=True).merge(mapping,on='feature_index').to_csv(
            output/'saved_active_contributions.csv',index=False)
    no0 = score-contributions[:,0]
    pred0 = predict_scores(no0)
    results['saved_hue0_zeroed_diagnostic'] = export_predictions(
        'saved_hue0_zeroed_diagnostic',dev,pred0,no0,output)
    t = dev[['image_id','group_id','split','y']].copy()
    t['score'] = score
    t['hue0_contribution'] = contributions[:,0]
    t['score_without_hue0'] = no0
    t['pred'] = pred
    t['pred_without_hue0'] = pred0
    t['fixed'] = (pred != dev.y.to_numpy()) & (pred0 == dev.y.to_numpy())
    t['broken'] = (pred == dev.y.to_numpy()) & (pred0 != dev.y.to_numpy())
    t.to_csv(output/'hue0_diagnostic.csv',index=False)
    t.groupby(['split','group_id']).agg(n=('y','size'),fixed=('fixed','sum'),broken=('broken','sum')).reset_index().to_csv(output/'hue0_fixed_broken.csv',index=False)
    return results


def run_cv(raw, y, groups, template, arms, folds, repeats, seed, output, eps):
    group_labels = pd.DataFrame({'group':groups,'y':y}).drop_duplicates()
    if group_labels.groupby('y').size().min() < folds:
        raise ValueError('Too few location groups per class for requested folds.')
    rows, stabilities, warning_rows = [], [], []
    for repeat in range(repeats):
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed+repeat)
        splits = list(cv.split(raw,y,groups))
        oof = {name: np.full(len(y),-1,dtype=int) for name in arms}
        for fold,(tr,va) in enumerate(splits):
            if set(y[tr]) != {0,1} or set(y[va]) != {0,1}:
                raise ValueError('A fold lacks a class. Use fewer folds or reconsider the predeclared seed.')
            assert not set(groups[tr]) & set(groups[va])
            for name,keep in arms.items():
                scale,clf,ok,caught = numerical_fit(raw[tr],y[tr],template,keep)
                pred = clf.predict(scale.transform(raw[va][:,keep]))
                oof[name][va] = pred
                m = metrics(y[va],pred)
                rows.append(dict(repeat=repeat,fold=fold,arm=name,n=len(va),
                    accuracy=m['accuracy'],macro_f1=m['macro_f1'],
                    balanced_accuracy=m['balanced_accuracy'],converged=ok))
                warning_rows.extend(dict(repeat=repeat,fold=fold,arm=name,**w) for w in caught)
                coef = np.zeros(raw.shape[1]); coef[keep] = clf.coef_[0]
                for j in np.flatnonzero(np.abs(coef)>eps):
                    stabilities.append(dict(repeat=repeat,fold=fold,arm=name,feature_index=int(j),weight=float(coef[j])))
        for name in arms:
            if (oof[name] < 0).any():
                raise AssertionError('Incomplete out-of-fold predictions.')
            m = metrics(y,oof[name])
            rows.append(dict(repeat=repeat,fold='pooled_oof',arm=name,n=len(y),
                accuracy=m['accuracy'],macro_f1=m['macro_f1'],balanced_accuracy=m['balanced_accuracy'],
                converged=all(r['converged'] for r in rows if r['repeat']==repeat and r['arm']==name)))
            pd.DataFrame(dict(row=np.arange(len(y)),group_id=groups,y=y,prediction=oof[name])).to_csv(
                output/f'cv_{name}_repeat_{repeat}_oof.csv',index=False)
    scores = pd.DataFrame(rows)
    scores.to_csv(output/'cv_scores.csv',index=False)
    write_json(output/'cv_warnings.json', warning_rows)
    summary = scores[scores.fold=='pooled_oof'].groupby('arm').agg(
        mean_macro_f1=('macro_f1','mean'),sd_macro_f1=('macro_f1','std'),
        mean_accuracy=('accuracy','mean'),mean_balanced_accuracy=('balanced_accuracy','mean'),
        all_converged=('converged','all')).reset_index()
    summary.to_csv(output/'cv_summary.csv',index=False)
    paired = scores[scores.fold=='pooled_oof'].pivot(index='repeat',columns='arm',values='macro_f1')
    paired['drop_minus_full'] = paired['drop_hue0']-paired['full']
    paired.to_csv(output/'cv_paired_macro_f1.csv')
    active = pd.DataFrame(stabilities,columns=['repeat','fold','arm','feature_index','weight'])
    active.to_csv(output/'cv_active_coefficients.csv',index=False)
    stability = active.groupby(['arm','feature_index']).agg(selected_folds=('weight','size'),
        mean_weight_when_selected=('weight','mean'),positive_folds=('weight',lambda v:int((v>0).sum())),
        negative_folds=('weight',lambda v:int((v<0).sum()))).reset_index()
    stability['total_folds'] = folds*repeats
    stability['selection_frequency'] = stability.selected_folds/(folds*repeats)
    stability.to_csv(output/'cv_feature_stability.csv',index=False)
    # Predeclared rule: no recommendation on nonconvergence; exact ties retain full.
    if not summary.all_converged.all():
        selected = None
    else:
        selected = sorted(summary.to_dict('records'),key=lambda r:(
            -r['mean_macro_f1'],-r['mean_balanced_accuracy'],-r['mean_accuracy'],r['arm']!='full'))[0]['arm']
    return summary, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline',type=Path,default=Path('models/final/pipeline.joblib'))
    parser.add_argument('--dataset-dir',type=Path,default=Path('dataset_dev'))
    parser.add_argument('--output-dir',type=Path,required=True,help='Must not exist; never overwrite final models.')
    parser.add_argument('--mode',choices=['analyze','compare'],default='compare')
    parser.add_argument('--folds',type=int,default=3)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--epsilon',type=float,default=1e-8)
    args = parser.parse_args()
    if args.folds < 2 or args.repeats < 1 or args.epsilon <= 0:
        parser.error('Need folds>=2, repeats>=1, epsilon>0.')
    dataset, source = args.dataset_dir.resolve(), args.pipeline.resolve()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError('Output directory already exists; choose a new run name.')
    if out == source.parent or out in source.parents:
        raise ValueError('Output must not contain the source model.')
    records = load_records(dataset)
    dev = records[records.split.isin(['train','validation'])].copy().reset_index(drop=True)
    paths, hashes = paths_and_hashes(dev,dataset)
    model = joblib.load(source)
    if list(model.named_steps) != ['features','scale','classifier']:
        raise ValueError('Expected original features -> scale -> classifier pipeline, without selectors.')
    clf, ext = model.named_steps['classifier'], model.named_steps['features']
    if not np.array_equal(clf.classes_,[0,1]) or clf.coef_.shape[0] != 1:
        raise ValueError('Expected binary linear classifier, classes [0,1].')
    # Project extractor is stateless; raw feature caching does not fit a scaler or selector.
    if not hasattr(ext,'feature_config') or not hasattr(ext,'preprocessing_config'):
        raise ValueError('Expected project ImageFeatureExtractor.')
    if not ext.feature_config.use_color:
        raise ValueError('Hue #0 requires enabled color features.')
    if clf.get_params().get('random_state',None) is None and args.mode=='compare':
        raise ValueError('Classifier random_state=None: set a documented seed in a separate baseline first.')
    print('Extracting train/validation pixels only; test images remain unopened.',flush=True)
    raw = np.asarray(ext.transform(paths))
    if raw.ndim!=2 or not np.isfinite(raw).all() or raw.shape[1]!=clf.coef_.shape[1]:
        raise ValueError('Invalid feature matrix.')
    mapping = feature_table(ext,raw.shape[1])
    if not (mapping.iloc[0]['block']=='color' and mapping.iloc[0]['local_index']==0):
        raise ValueError('Feature 0 is not Hue bin 0.')
    out.mkdir(parents=True,exist_ok=False)
    records.groupby(['split','y']).agg(images=('image_id','size'),groups=('group_id','nunique')).reset_index().to_csv(out/'split_counts.csv',index=False)
    dev[['image_id','relative_path','group_id','split','y']].assign(sha256=hashes).to_csv(out/'development_manifest.csv',index=False)
    mapping.to_csv(out/'feature_mapping.csv',index=False)
    metadata = dict(arguments=vars(args),source_pipeline_sha256=sha256(source),
        images_csv_sha256=sha256(dataset/'images.csv'),locations_csv_sha256=sha256(dataset/'locations.csv'),
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,
        classifier_params=repr(clf.get_params()),feature_config=repr(ext.feature_config),
        preprocessing_config=repr(ext.preprocessing_config),test_images_opened=0,
        cv_selection='mean pooled OOF macro-F1, balanced accuracy, accuracy; exact ties full',
        caveat='Feature-0 hypothesis was informed by validation; this is exploratory, not independent confirmation.')
    write_json(out/'run_metadata.json',metadata)
    results = saved_analysis(model,raw,paths,dev,mapping,out,args.epsilon)
    selected, cv_summary = None, None
    if args.mode=='compare':
        train = dev.split.eq('train').to_numpy()
        y = dev.y.to_numpy()
        full = np.ones(raw.shape[1],dtype=bool)
        drop = full.copy(); drop[0]=False
        arms = dict(full=full,drop_hue0=drop)
        cv_summary,selected = run_cv(raw[train],y[train],dev.group_id.to_numpy()[train],
            model,arms,args.folds,args.repeats,args.seed,out,args.epsilon)
        for name,keep in arms.items():
            scaler,candidate,converged,caught = numerical_fit(raw[train],y[train],model,keep)
            steps = [('features',clone(ext))]
            if name=='drop_hue0':
                steps.append(('drop_hue0',FunctionTransformer(np.delete,kw_args={'obj':[0],'axis':1},validate=False)))
            steps += [('scale',scaler),('classifier',candidate)]
            pipeline = Pipeline(steps)
            z = scaler.transform(raw[:,keep])
            score = linear_score(z,candidate)
            pred = candidate.predict(z)
            folder = out/'candidates'/name
            folder.mkdir(parents=True)
            destination = folder/'pipeline.joblib'
            joblib.dump(pipeline,destination)
            loaded = joblib.load(destination)
            # Test path and byte inference after serialization using validation only.
            check = [paths[i] for i in np.flatnonzero(~train)[:2]]
            expected = loaded.decision_function(check)
            np.testing.assert_allclose(expected,loaded.decision_function([p.read_bytes() for p in check]),atol=1e-6)
            np.testing.assert_allclose(expected,score[np.flatnonzero(~train)[:2]],atol=1e-6)
            np.testing.assert_array_equal(loaded.predict(check),pred[np.flatnonzero(~train)[:2]])
            write_json(folder/'experiment_metadata.json',dict(
                arm=name,converged=converged,warnings=caught,pipeline_sha256=sha256(destination),
                source_pipeline_sha256=sha256(source),trained_on_splits=['train'],
                class_labels={'0':'outside','1':'inside'},test_images_opened=0,
                note='Experimental candidate, not promoted to final; metadata is not a guaranteed WebUI schema.'))
            results[f'retrained_{name}'] = export_predictions(f'retrained_{name}',dev,pred,score,out)
            export_coefficients(f'retrained_{name}',scaler,candidate,keep,mapping,out,args.epsilon)
            if not converged:
                selected=None
    write_json(out/'all_metrics.json',results)
    write_json(out/'cv_recommendation.json',dict(recommended_arm=selected,
        automatically_promoted=False,test_images_opened=0,
        warning='Recommendation uses training CV only. Hypothesis was informed by validation. Do not tune until validation reaches a target.'))
    lines=['# Hue #0 analysis and retraining comparison','',
        'No test image bytes were read. Original pipeline was not modified.','',
        '| Model | Split | Accuracy | Macro-F1 |','|---|---|---:|---:|']
    for name,split_results in results.items():
        for split,m in split_results.items():
            lines.append(f"| {name} | {split} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} |")
    lines += ['',f'Training-group CV recommendation: {selected or "none (analysis-only or convergence review required)"}.',
        '', 'Zeroed contribution means standardized feature 0 is set to zero (training mean), not raw hue set to zero.',
        'Retraining removes original feature index 0 before fitting the scaler and classifier.',
        'CV refits scaling and classifier inside each fold; both arms use identical group folds and classifier settings.',
        'Repeated folds overlap; their standard deviations are descriptive, not confidence intervals.',
        'The feature-0 hypothesis was informed by validation. These results are exploratory.',
        'No candidate is automatically promoted. Candidate experiment_metadata.json is not a guaranteed WebUI metadata format.',
        'Exact duplicate-byte checks cover train/validation only, not edited copies or test images.',
        'Boundary verification flags are checked, but this script cannot establish geographic truth.']
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))
    print('\nResults:',out)


if __name__=='__main__':
    main()
