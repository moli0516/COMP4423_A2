"""Map active coefficients onto a train/validation image; no training or test decoding."""
import argparse
import csv
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import feature_block_sizes
from src.preprocessing import load_and_preprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pipeline', default='models/final/pipeline.joblib')
    p.add_argument('--dataset-dir', type=Path, default=Path('dataset_dev'))
    p.add_argument('--image-id', help='image_id in train or validation; default: first validation image')
    p.add_argument('--output-dir', type=Path, default=Path('results/active_features'))
    a = p.parse_args()
    if a.output_dir.exists():
        raise FileExistsError('Choose a new --output-dir to avoid overwriting results.')
    with (a.dataset_dir/'images.csv').open(encoding='utf-8-sig', newline='') as f:
        records = list(csv.DictReader(f))
    if a.image_id:
        row = next(r for r in records if r['image_id']==a.image_id)
    else:
        row = next(r for r in records if r['split']=='validation')
    if row['split'] not in ('train','validation'):
        raise ValueError('Only train/validation images allowed; test remains untouched.')
    root = a.dataset_dir.resolve()
    path = (root/row['relative_path']).resolve()
    assert path.is_relative_to(root)
    model = joblib.load(a.pipeline)
    ext = model.named_steps['features']
    fc, pc = ext.feature_config, ext.preprocessing_config
    clf = model.named_steps['classifier']
    assert list(clf.classes_)==[0,1], 'Expected binary classes [0,1].'
    weights = np.asarray(clf.coef_).ravel()
    raw = ext.transform([path])
    scaled = model.named_steps['scale'].transform(raw)[0]
    contributions = weights*scaled
    sizes = feature_block_sizes(fc,pc)
    assert sum(sizes.values())==weights.size
    image = load_and_preprocess(path,pc)
    h,w = image.shape[:2]
    cell,b,o = fc.hog_pixels_per_cell,fc.hog_cells_per_block,fc.hog_orientations
    hog_shape = (h//cell-b+1,w//cell-b+1,b,b,o)
    active=[]
    offset=0
    for block,size in sizes.items():
        for local in np.flatnonzero(np.abs(weights[offset:offset+size])>1e-8):
            j=int(offset+local)
            item=dict(feature_index=j,block=block,local_index=int(local),weight=float(weights[j]),
                      standardized_value=float(scaled[j]),contribution=float(contributions[j]),
                      description='',x0='',y0='',x1='',y1='')
            if block=='hog':
                by,bx,cy,cx,k=map(int,np.unravel_index(local,hog_shape))
                x0,y0=(bx+cx)*cell,(by+cy)*cell
                item.update(x0=x0,y0=y0,x1=x0+cell,y1=y0+cell,
                            description=f'block ({by},{bx}); cell ({cy},{cx}); gradient bin {k}: {180*k/o:.1f}-{180*(k+1)/o:.1f} deg')
            elif block=='color':
                tile,rem=divmod(int(local),3*fc.color_bins)
                ch,k=divmod(rem,fc.color_bins)
                ty,tx=divmod(tile,fc.color_grid)
                yedges=np.cumsum([0]+[len(v) for v in np.array_split(np.arange(h),fc.color_grid)])
                xedges=np.cumsum([0]+[len(v) for v in np.array_split(np.arange(w),fc.color_grid)])
                item.update(x0=int(xedges[tx]),y0=int(yedges[ty]),x1=int(xedges[tx+1]),y1=int(yedges[ty+1]),
                            description=f'HSV {"HSV"[ch]} bin {k}: [{k/fc.color_bins:.4f}, {(k+1)/fc.color_bins:.4f}]; tile ({ty},{tx})')
            else:
                item['description']=f'Global uniform LBP histogram bin {local}; not spatially localized'
            active.append(item)
        offset+=size
    a.output_dir.mkdir(parents=True)
    with (a.output_dir/'active_features.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(active[0])); writer.writeheader(); writer.writerows(active)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    hog_items=[v for v in active if v['block']=='hog']
    fig,axes=plt.subplots(1,3,figsize=(17,6))
    axes[0].imshow(image); axes[0].set_title('Actual preprocessed input'); axes[0].axis('off')
    axes[1].imshow(image)
    for v in hog_items:
        colour='crimson' if v['weight']>0 else 'royalblue'
        axes[1].add_patch(Rectangle((v['x0']-.5,v['y0']-.5),cell,cell,fill=False,edgecolor=colour,linewidth=1.6))
    for i,(xy,items) in enumerate(__import__('itertools').groupby(sorted(hog_items,key=lambda v:(v['y0'],v['x0'])),key=lambda v:(v['y0'],v['x0']))):
        group=list(items)
        axes[1].text(xy[1]+1,xy[0]+cell/2,','.join(str(v['feature_index']) for v in group),fontsize=6,
                     color='black',bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
    axes[1].set_title('Active HOG cell supports (not saliency)\nRed/blue: positive/negative coefficients'); axes[1].axis('off')
    ranked=sorted(active,key=lambda v:abs(v['contribution']))
    axes[2].barh(range(len(ranked)),[v['contribution'] for v in ranked],
                 color=['crimson' if v['contribution']>=0 else 'royalblue' for v in ranked])
    axes[2].set_yticks(range(len(ranked)),[f"{v['block']} #{v['feature_index']}" for v in ranked],fontsize=8)
    axes[2].axvline(0,color='gray',lw=.8)
    axes[2].set_title('This image: coefficient x standardized feature\nPositive supports inside; negative supports outside')
    score=float(clf.intercept_[0]+contributions.sum())
    actual=float(model.decision_function([path])[0])
    assert np.isclose(score,actual)
    fig.suptitle(f"{row['image_id']} | true={row['label']} | predicted={int(model.predict([path])[0])} | score={score:.3f} | intercept={clf.intercept_[0]:.3f}",fontsize=11)
    fig.tight_layout()
    fig.savefig(a.output_dir/'feature_locations.png',dpi=180)
    fig.savefig(a.output_dir/'feature_locations.pdf')
    plt.close(fig)
    for v in active:
        print(v['feature_index'],v['block'],v['description'],f"weight={v['weight']:.4f}, contribution={v['contribution']:.4f}")
    print('Saved to',a.output_dir)

if __name__=='__main__':
    main()
