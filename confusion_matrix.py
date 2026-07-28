"""
Confusion matrix tool for MCTformer CAM outputs.
讀 cam-npy-layer*/ 內的 npy CAM，對應 GT 算 confusion matrix，
顯示「某類 GT 被誤判成什麼類」的分布。

Example:
    python confusion_matrix.py \
        --cam-dir saved_model/Postdam_pr_ni_lr1/cam-npy-layer12 \
        --gt-dir  ../data/Postdam/voc12/VOCdevkit/VOC2012/SegmentationClass \
        --list    ../data/Postdam/train_id.txt \
        --dataset postdam
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x


# gt_classes: full set in raw GT (index = raw GT value)
# pred_classes: what the model actually predicts (5 in no-bg setups)
# gt_to_pred: map raw GT value -> pred col index, or None for "extra GT class" (e.g. clutter)
DATASETS = {
    # Vaihingen/Postdam: raw GT 0=Clutter(red, ignored at train), 1..5=Imperv..Car
    'vaihingen': {
        'gt_classes':   ['Clutter', 'Imperv', 'Build', 'Low_v', 'Tree', 'Car'],
        'pred_classes': ['Imperv', 'Build', 'Low_v', 'Tree', 'Car'],
        'gt_row_of': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},  # raw gt -> row idx
        'pred_col_of_gt': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4},   # raw gt -> matching pred col (for diag)
    },
    'postdam': {
        'gt_classes':   ['Clutter', 'Imperv', 'Build', 'Low_v', 'Tree', 'Car'],
        'pred_classes': ['Imperv', 'Build', 'Low_v', 'Tree', 'Car'],
        'gt_row_of': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        'pred_col_of_gt': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
    },
    'deepglobe': {
        'gt_classes':   ['Urban', 'Agri', 'Range', 'Forest', 'Water', 'Barren'],
        'pred_classes': ['Urban', 'Agri', 'Range', 'Forest', 'Water', 'Barren'],
        'gt_row_of': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 0: 255},  # 0 = ignore
        'pred_col_of_gt': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5},
    },
    'landslide': {
        'gt_classes':   ['bg', 'landslide'],
        'pred_classes': ['bg', 'landslide'],
        'gt_row_of': {0: 0, 1: 1},
        'pred_col_of_gt': {0: 0, 1: 1},
    },
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cam-dir', required=True)
    p.add_argument('--gt-dir', required=True)
    p.add_argument('--list', required=True)
    p.add_argument('--dataset', required=True, choices=list(DATASETS.keys()))
    p.add_argument('--save', default=None, help='optional: dump confusion matrix to .csv')
    args = p.parse_args()

    ds = DATASETS[args.dataset]
    gt_classes = ds['gt_classes']        # rows
    pred_classes = ds['pred_classes']    # cols
    gt_row_of = ds['gt_row_of']          # raw gt value -> row idx (or 255 = ignore)
    pred_col_of_gt = ds['pred_col_of_gt']
    n_rows = len(gt_classes)
    n_cols = len(pred_classes)

    # Build lookup table: raw gt value (0..255) -> row index (255 = ignore)
    gt_lut = np.full(256, 255, dtype=np.int64)
    for raw, row in gt_row_of.items():
        gt_lut[raw] = row

    name_list = pd.read_csv(args.list, names=['filename'])['filename'].values

    conf = np.zeros((n_rows, n_cols), dtype=np.int64)
    skipped = 0
    processed = 0

    pbar = tqdm(name_list, desc='eval', file=sys.stdout, mininterval=1.0)
    for name in pbar:
        cam_file = os.path.join(args.cam_dir, name + '.npy')
        gt_file = os.path.join(args.gt_dir, name + '.png')
        if not (os.path.exists(cam_file) and os.path.exists(gt_file)):
            skipped += 1
            continue

        cam_dict = np.load(cam_file, allow_pickle=True).item()
        if len(cam_dict) == 0:
            skipped += 1
            continue
        h, w = list(cam_dict.values())[0].shape
        tensor = np.zeros((n_cols, h, w), dtype=np.float32)
        for k, v in cam_dict.items():
            tensor[k] = v
        pred = np.argmax(tensor, axis=0).astype(np.int64)  # 0..n_cols-1

        gt_raw = np.array(Image.open(gt_file)).astype(np.int64)
        gt_row = gt_lut[gt_raw]  # convert to row idx, 255 = ignore

        valid = gt_row < 255
        flat_gt = gt_row[valid]
        flat_pred = pred[valid]
        idx = flat_gt * n_cols + flat_pred
        bins = np.bincount(idx, minlength=n_rows * n_cols)
        conf += bins.reshape(n_rows, n_cols)
        processed += 1

    print(f'\nprocessed {processed} / skipped {skipped} (total {len(name_list)})', flush=True)

    # Build row -> matching pred col map (for diagonal mark / recall)
    row_to_col = {}
    for raw, col in pred_col_of_gt.items():
        row_to_col[gt_row_of[raw]] = col

    # ---- print confusion matrix (rows = GT n_rows, cols = pred n_cols) ----
    col_w = 9
    print('\nConfusion Matrix  (rows = GT, cols = predicted, values = %% of that GT row)\n')
    header = '%-10s' % 'GT \\ Pred'
    for c in pred_classes:
        header += ('%' + str(col_w) + 's') % c
    header += '%12s' % 'GT pixels'
    print(header)
    print('-' * len(header))

    for i in range(n_rows):
        total = conf[i].sum()
        row = '%-10s' % gt_classes[i]
        match_col = row_to_col.get(i, -1)
        for j in range(n_cols):
            pct = 100.0 * conf[i, j] / max(total, 1)
            mark = '*' if j == match_col else ' '
            row += ('%' + str(col_w - 1) + '.2f%s') % (pct, mark)
        row += '%12d' % total
        print(row)

    print('\n(* = matching predicted class, row 0 has no match = extra GT-only class)')

    # ---- per-GT-class breakdown ----
    print('\nFor each GT class, where its pixels were predicted:')
    for i in range(n_rows):
        total = conf[i].sum()
        if total == 0:
            print(f'  GT {gt_classes[i]:8s} : no GT pixels')
            continue
        match_col = row_to_col.get(i, -1)
        order = sorted(
            [(j, conf[i, j]) for j in range(n_cols)],
            key=lambda x: -x[1])
        parts = []
        for j, cnt in order:
            pct = 100.0 * cnt / total
            if pct < 0.5:
                continue
            tag = '(correct)' if j == match_col else ''
            parts.append(f'{pred_classes[j]} {pct:.1f}% {tag}'.strip())
        if match_col >= 0:
            recall = 100.0 * conf[i, match_col] / total
            print(f'  GT {gt_classes[i]:8s} (recall {recall:.1f}%): ' + ', '.join(parts))
        else:
            print(f'  GT {gt_classes[i]:8s} (no matching pred class):  ' + ', '.join(parts))

    # ---- precision view: each predicted class is made of what GT ----
    print('\nFor each predicted class, where its pixels came from:')
    for j in range(n_cols):
        col_total = conf[:, j].sum()
        if col_total == 0:
            print(f'  pred {pred_classes[j]:8s}: no predictions')
            continue
        # find matching GT row for this pred col
        match_row = next((r for r, c in row_to_col.items() if c == j), -1)
        order = sorted(
            [(i, conf[i, j]) for i in range(n_rows)],
            key=lambda x: -x[1])
        parts = []
        for i, cnt in order:
            pct = 100.0 * cnt / col_total
            if pct < 0.5:
                continue
            tag = '(correct)' if i == match_row else ''
            parts.append(f'{gt_classes[i]} {pct:.1f}% {tag}'.strip())
        precision = 100.0 * conf[match_row, j] / col_total if match_row >= 0 else 0.0
        print(f'  pred {pred_classes[j]:8s} (prec {precision:.1f}%): ' + ', '.join(parts))

    if args.save:
        df = pd.DataFrame(conf, index=gt_classes, columns=pred_classes)
        df.to_csv(args.save)
        print(f'\nSaved raw confusion matrix to {args.save}')


if __name__ == '__main__':
    main()
