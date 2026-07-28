"""
Slice ISPRS Potsdam IRRG TIF tiles into 224x224 patches matching the existing
RGB-version naming scheme, so the same train_id.txt / cls_labels.npy can be reused.

Source:  /workspace/data/Postdam/3_Ortho_IRRG/top_potsdam_{X}_{Y}_IRRG.tif  (6000x6000, 3-ch IRRG)
Output:  /workspace/data/Postdam_IRRG/voc12/VOCdevkit/VOC2012/JPEGImages/
            top_potsdam_{X}_{Y}_{y}_{x}.jpg                                 (224x224, 3-ch IRRG)

Symlinks all label/id files from existing Postdam directory.
"""
import os
import sys
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

SRC_IRRG_DIR = '/workspace/data/Postdam/3_Ortho_IRRG'
DST_ROOT     = '/workspace/data/Postdam_IRRG'
SRC_RGB_ROOT = '/workspace/data/Postdam'

PATCH = 224
STRIDE = 224          # no overlap, matches existing 27x27 grid
# 6000 → starts at 0, 224, 448, ..., 5824 → 27 unique positions
# (last patch goes from 5824 to 6048 — slight clipping handled below)

VOC_DST = os.path.join(DST_ROOT, 'voc12', 'VOCdevkit', 'VOC2012')
JPEG_DST = os.path.join(VOC_DST, 'JPEGImages')

def make_dirs():
    os.makedirs(JPEG_DST, exist_ok=True)
    os.makedirs(VOC_DST, exist_ok=True)


def slice_tif(tif_path):
    """Cut a 6000x6000 TIF into 224x224 patches matching RGB version offsets."""
    arr = np.array(Image.open(tif_path))
    H, W = arr.shape[:2]
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f'unexpected shape {arr.shape} for {tif_path}')

    # Extract area id from filename: top_potsdam_X_Y_IRRG.tif
    base = os.path.basename(tif_path).replace('_IRRG.tif', '')
    # base = "top_potsdam_X_Y"

    n_saved = 0
    for y in range(0, H, STRIDE):
        for x in range(0, W, STRIDE):
            # Clip to image bounds (last row/col may extend past 6000)
            y2 = min(y + PATCH, H)
            x2 = min(x + PATCH, W)
            patch = arr[y:y2, x:x2]
            # Pad to 224x224 if at edge
            if patch.shape[0] < PATCH or patch.shape[1] < PATCH:
                pad = np.zeros((PATCH, PATCH, 3), dtype=arr.dtype)
                pad[:patch.shape[0], :patch.shape[1]] = patch
                patch = pad

            out_name = f'{base}_{y}_{x}.jpg'
            out_path = os.path.join(JPEG_DST, out_name)
            Image.fromarray(patch).save(out_path, quality=95)
            n_saved += 1
    return n_saved


def main():
    print(f'== Setting up {DST_ROOT} ==')
    make_dirs()
    print(f'Output JPEG dir: {JPEG_DST}')
    print('Note: labels / id lists / cls_labels.npy stay in original Postdam dir;')
    print('      run scripts must point --gt_dir/--img-list/--label-file-path to /workspace/data/Postdam.')

    tifs = sorted(glob.glob(os.path.join(SRC_IRRG_DIR, 'top_potsdam_*_IRRG.tif')))
    print(f'\n-- slicing {len(tifs)} IRRG TIFs → {JPEG_DST} --')

    total = 0
    for tif in tqdm(tifs, desc='tiles'):
        total += slice_tif(tif)
    print(f'\nDone. Saved {total} patches.')

    # quick verification
    n_existing = len(os.listdir(JPEG_DST))
    print(f'JPEGImages now has {n_existing} files')

    # cross-check against RGB version
    rgb_dir = os.path.join(SRC_RGB_ROOT, 'voc12/VOCdevkit/VOC2012/JPEGImages')
    if os.path.isdir(rgb_dir):
        n_rgb = len(os.listdir(rgb_dir))
        print(f'(RGB version has {n_rgb} files for comparison)')


if __name__ == '__main__':
    main()
