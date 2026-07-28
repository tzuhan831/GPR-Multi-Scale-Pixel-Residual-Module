import os
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_labels
    HAS_CRF = True
except ImportError:
    HAS_CRF = False

categories_voc = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
              'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

categories_vaihingen = ['background', 'Impervious', 'Building', 'Low_veg', 'Tree', 'Car']  # 6 classes with background

categories_postdam = ['background', 'Impervious', 'Building', 'Low_veg', 'Tree', 'Car']  # 6 classes with background

# ISPRS standard colormap (class index → RGB), used for Vaihingen/Postdam pseudo-mask visualization
ISPRS_COLORMAP = np.array([
    [255, 255, 255],  # 0: Impervious → white
    [0,   0,   255],  # 1: Building   → blue
    [0,   255, 255],  # 2: Low_veg    → cyan
    [0,   128,   0],  # 3: Tree       → dark green
    [255, 255,   0],  # 4: Car        → yellow
], dtype=np.uint8)

def apply_isprs_colormap(predict):
    """Convert class index array (H,W) to RGB image using ISPRS colormap."""
    h, w = predict.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(len(ISPRS_COLORMAP)):
        rgb[predict == i] = ISPRS_COLORMAP[i]
    return rgb

# DeepGlobe Land Cover：6 類（含 background 共 7 個 entry）
# mask 編碼：0=Unknown（已從 cls_labels 排除，視同 ignore）, 1-6=Urban/Agri/Range/Forest/Water/Barren, 255=ignore
categories_deepglobe = ['background', 'Urban', 'Agriculture', 'Rangeland', 'Forest', 'Water', 'Barren']

# DeepGlobe official palette（RGB），跟 ISPRS_COLORMAP 用法一致
DEEPGLOBE_COLORMAP = np.array([
    [0,   255, 255],  # 0: Urban       → cyan
    [255, 255,   0],  # 1: Agriculture → yellow
    [255, 0,   255],  # 2: Rangeland   → magenta
    [0,   255,   0],  # 3: Forest      → green
    [0,     0, 255],  # 4: Water       → blue
    [255, 255, 255],  # 5: Barren      → white
], dtype=np.uint8)

def apply_deepglobe_colormap(predict):
    """Convert class index array (H,W) to RGB image using DeepGlobe colormap."""
    h, w = predict.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(len(DEEPGLOBE_COLORMAP)):
        rgb[predict == i] = DEEPGLOBE_COLORMAP[i]
    return rgb

categories = categories_voc  # default


def crf_inference_label(img, labels, n_labels, t=10, gt_prob=0.7):
    """Apply CRF post-processing to refine segmentation labels."""
    if not HAS_CRF:
        raise ImportError("pydensecrf not installed. Run: pip install pydensecrf")

    h, w = labels.shape
    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)
    d.setUnaryEnergy(unary)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=np.ascontiguousarray(img), compat=10)

    q = d.inference(t)
    return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0).astype(np.uint8)

def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21, input_type='png', threshold=1.0, printlog=False, no_background=False, cat_names=None, out_crf=False, img_dir=None, out_dir=None, ignore_clutter=False, conf_thresh=0.0):
    TP = []
    P = []
    T = []
    for i in range(num_cls):
        TP.append(multiprocessing.Value('i', 0, lock=True))
        P.append(multiprocessing.Value('i', 0, lock=True))
        T.append(multiprocessing.Value('i', 0, lock=True))

    def compare(start,step,TP,P,T,input_type,threshold,no_background,out_crf,img_dir,out_dir,ignore_clutter,conf_thresh):
        for idx in range(start,len(name_list),step):
            name = name_list[idx]
            if input_type == 'png':
                predict_file = os.path.join(predict_folder,'%s.png'%name)
                predict = np.array(Image.open(predict_file)) #cv2.imread(predict_file)
                if num_cls == 81:
                    predict = predict - 91
            elif input_type == 'npy':
                predict_file = os.path.join(predict_folder,'%s.npy'%name)
                if not os.path.exists(predict_file):
                    # Skip images without CAM (e.g., all-zero labels)
                    continue
                predict_dict = np.load(predict_file, allow_pickle=True).item()
                h, w = list(predict_dict.values())[0].shape

                if no_background:
                    # For datasets without background class (e.g., original Vaihingen)
                    tensor = np.zeros((num_cls,h,w),np.float32)
                    for key in predict_dict.keys():
                        tensor[key] = predict_dict[key]
                    predict = np.argmax(tensor, axis=0).astype(np.uint8)
                    # 2026-06-10 加 conf-thresh：pixel max CAM 值 < thresh 標 255（IoU 計算時當 FN）
                    max_conf = tensor.max(axis=0) if conf_thresh > 0 else None

                    # Apply CRF if requested (no_background mode)
                    if out_crf and img_dir is not None:
                        img_path = os.path.join(img_dir, '%s.jpg' % name)
                        if not os.path.exists(img_path):
                            img_path = os.path.join(img_dir, '%s.png' % name)
                        orig_image = np.array(Image.open(img_path).convert("RGB"))
                        predict = crf_inference_label(orig_image, predict, n_labels=num_cls)

                    # Apply confidence threshold AFTER CRF: low-conf → ignore sentinel
                    if conf_thresh > 0:
                        predict[max_conf < conf_thresh] = 255

                    # Save output if out_dir specified (no_background mode)
                    if out_dir is not None:
                        if num_cls == 5 and predict.max() < len(ISPRS_COLORMAP):
                            save_arr = apply_isprs_colormap(predict)
                        elif num_cls == 6 and predict.max() < len(DEEPGLOBE_COLORMAP):
                            save_arr = apply_deepglobe_colormap(predict)
                        else:
                            save_arr = predict
                        predict_img = Image.fromarray(save_arr)
                        predict_img.save(os.path.join(out_dir, '%s.png' % name))
                else:
                    # WeakTr style: with softmax and label_key mapping
                    cam = np.array([predict_dict[key] for key in predict_dict.keys()])
                    label_key = np.array([key+1 for key in predict_dict.keys()]).astype(np.uint8)
                    cam = np.pad(cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=threshold)
                    label_key = np.pad(label_key, (1, 0), mode='constant', constant_values=0)

                    # Apply softmax before argmax (WeakTr style)
                    cam = F.softmax(torch.tensor(cam).float(), dim=0).numpy()
                    predict = np.argmax(cam, axis=0).astype(np.uint8)

                    # Apply CRF if requested
                    if out_crf and img_dir is not None:
                        img_path = os.path.join(img_dir, '%s.jpg' % name)
                        if not os.path.exists(img_path):
                            img_path = os.path.join(img_dir, '%s.png' % name)
                        orig_image = np.array(Image.open(img_path).convert("RGB"))
                        predict = crf_inference_label(orig_image, predict, n_labels=cam.shape[0])

                    predict = label_key[predict]  # Map back to labels

                    # Save output if out_dir specified
                    if out_dir is not None:
                        if num_cls == 5 and predict.max() < len(ISPRS_COLORMAP):
                            save_arr = apply_isprs_colormap(predict)
                        elif num_cls == 6 and predict.max() < len(DEEPGLOBE_COLORMAP):
                            save_arr = apply_deepglobe_colormap(predict)
                        else:
                            save_arr = predict
                        predict_img = Image.fromarray(save_arr)
                        predict_img.save(os.path.join(out_dir, '%s.png' % name))

            gt_file = os.path.join(gt_folder,'%s.png'%name)
            gt = np.array(Image.open(gt_file))
            # Remap Vaihingen/Potsdam GT labels: [1,2,3,4,5] -> [0,1,2,3,4]
            if no_background and num_cls == 5:
                # 舊版 remap：Clutter (raw GT=0) 沒被處理，跟新 Imperv (0) 重疊 → Clutter pixel 被當 Imperv GT 算 IoU
                # remap = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
                if ignore_clutter:
                    # 新版：把 Clutter 標 255 (ignore)，從 IoU 排除
                    remap = {0: 255, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
                else:
                    remap = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
                for old_val, new_val in remap.items():
                    gt[gt == old_val] = new_val
            # DeepGlobe GT labels: [1,2,3,4,5,6] -> [0,1,2,3,4,5]，0 (Unknown) 視同 ignore (255)
            elif no_background and num_cls == 6:
                remap = {0: 255, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
                for old_val, new_val in remap.items():
                    gt[gt == old_val] = new_val
            cal = gt<255
            mask = (predict==gt) * cal
      
            for i in range(num_cls):
                P[i].acquire()
                P[i].value += np.sum((predict==i)*cal)
                P[i].release()
                T[i].acquire()
                T[i].value += np.sum((gt==i)*cal)
                T[i].release()
                TP[i].acquire()
                TP[i].value += np.sum((gt==i)*mask)
                TP[i].release()
    p_list = []
    for i in range(8):
        p = multiprocessing.Process(target=compare, args=(i,8,TP,P,T,input_type,threshold,no_background,out_crf,img_dir,out_dir,ignore_clutter,conf_thresh))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = [] 
    for i in range(num_cls):
        IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value+1e-10))
        T_TP.append(T[i].value/(TP[i].value+1e-10))
        P_TP.append(P[i].value/(TP[i].value+1e-10))
        FP_ALL.append((P[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
        FN_ALL.append((T[i].value-TP[i].value)/(T[i].value + P[i].value - TP[i].value + 1e-10))
    # Use provided category names or default
    if cat_names is None:
        cat_names = categories

    loglist = {}
    for i in range(num_cls):
        if i < len(cat_names):
            loglist[cat_names[i]] = IoU[i] * 100
        else:
            loglist[f'class_{i}'] = IoU[i] * 100

    miou = np.mean(np.array(IoU))
    loglist['mIoU'] = miou * 100
    fp = np.mean(np.array(FP_ALL))
    loglist['FP'] = fp * 100
    fn = np.mean(np.array(FN_ALL))
    loglist['FN'] = fn * 100
    if printlog:
        for i in range(num_cls):
            name = cat_names[i] if i < len(cat_names) else f'class_{i}'
            if i%2 != 1:
                print('%11s:%7.3f%%'%(name,IoU[i]*100),end='\t')
            else:
                print('%11s:%7.3f%%'%(name,IoU[i]*100))
        print('\n======================================================')
        print('%11s:%7.3f%%'%('mIoU',miou*100))
        print('\n')
        print(f'FP = {fp*100}, FN = {fn*100}')
    return loglist

def writelog_pretty(predict_dir, metric, comment, best_thr=None, cat_names=None):
    """Save human-readable evaluation results to predict_dir/evalresult.txt"""
    import time
    filepath = os.path.join(predict_dir, 'evalresult.txt')
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'a') as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        f.write('\t%s\n' % comment)
        if best_thr is not None:
            f.write('Best threshold: %.3f\tmIoU: %.3f%%\n' % (best_thr, metric['mIoU']))
        f.write('\n--- Per-class IoU ---\n')
        keys = [k for k in metric.keys() if k not in ('mIoU', 'FP', 'FN')]
        for i, name in enumerate(keys):
            end = '\n' if i % 2 == 1 else '\t'
            f.write('%11s: %7.3f%%' % (name, metric[name]) + end)
        if len(keys) % 2 == 1:
            f.write('\n')
        f.write('\n======================================================\n')
        f.write('%11s: %7.3f%%\n' % ('mIoU', metric['mIoU']))
        f.write('\nFP = %s, FN = %s\n' % (metric['FP'], metric['FN']))
        f.write('=====================================\n\n')


def writedict(file, dictionary):
    s = ''
    for key in dictionary.keys():
        sub = '%s:%s  '%(key, dictionary[key])
        s += sub
    s += '\n'
    file.write(s)

def writelog(filepath, metric, comment):
    filepath = filepath
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    logfile = open(filepath,'a')
    import time
    logfile.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logfile.write('\t%s\n'%comment)
    writedict(logfile, metric)
    logfile.write('=====================================\n')
    logfile.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--list", default='./VOC2012/ImageSets/Segmentation/train.txt', type=str)
    parser.add_argument("--predict_dir", default='./out_rw', type=str)
    parser.add_argument("--gt_dir", default='./VOC2012/SegmentationClass', type=str)
    parser.add_argument('--logfile', default='./evallog.txt',type=str)
    parser.add_argument('--comment', required=True, type=str)
    parser.add_argument('--type', default='png', choices=['npy', 'png'], type=str)
    parser.add_argument('--t', default=None, type=float)
    parser.add_argument('--curve', default=False, type=bool)
    parser.add_argument('--num_classes', default=21, type=int)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=60, type=int)
    parser.add_argument('--no-background', action='store_true', help='For datasets without background class (e.g., Vaihingen)')
    parser.add_argument('--dataset', default='voc', choices=['voc', 'vaihingen', 'postdam', 'deepglobe'], type=str, help='Dataset type for category names')
    parser.add_argument('--out-crf', action='store_true', help='Apply CRF post-processing')
    parser.add_argument('--img_dir', default=None, type=str, help='Image directory for CRF (required if --out-crf)')
    parser.add_argument('--out-dir', default=None, type=str, help='Output directory for CRF-refined masks')
    parser.add_argument('--ignore-clutter', action='store_true',
                        help='ISPRS only: treat Clutter (raw GT=0) as ignore (255). Default off preserves legacy behavior where Clutter is counted as Imperv GT.')
    parser.add_argument('--conf-thresh', default=0.0, type=float,
                        help='If > 0, pixel with max class CAM < conf_thresh marked as 255 (treated as FN for whatever GT class is there). Only used in --no-background mode.')
    args = parser.parse_args()

    # Set categories based on dataset
    if args.dataset == 'vaihingen':
        cat_names = categories_vaihingen
    elif args.dataset == 'postdam':
        cat_names = categories_postdam
    elif args.dataset == 'deepglobe':
        cat_names = categories_deepglobe
    else:
        cat_names = categories_voc

    if args.type == 'npy':
        assert args.t is not None or args.curve
    df = pd.read_csv(args.list, names=['filename'])
    name_list = df['filename'].values
    no_bg = getattr(args, 'no_background', False)
    if no_bg and cat_names[0] == 'background':
        cat_names = cat_names[1:]
    out_crf = getattr(args, 'out_crf', False)
    img_dir = getattr(args, 'img_dir', None)
    out_dir = getattr(args, 'out_dir', None)

    # Create output directory if specified
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    if not args.curve:
        loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, args.num_classes, args.type, args.t, printlog=True, no_background=no_bg, cat_names=cat_names, out_crf=out_crf, img_dir=img_dir, out_dir=out_dir, ignore_clutter=args.ignore_clutter, conf_thresh=args.conf_thresh)
        writelog(args.logfile, loglist, args.comment)
        writelog_pretty(args.predict_dir, loglist, args.comment)
    else:
        l = []
        max_mIoU = 0.0
        best_thr = 0.0
        mIoU_history = []  # 2026-05-19：用於早停判定
        for i in range(args.start, args.end):
            t = i/100.0
            loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, args.num_classes, args.type, t, no_background=no_bg, cat_names=cat_names, out_crf=out_crf, img_dir=img_dir, out_dir=out_dir, ignore_clutter=args.ignore_clutter, conf_thresh=args.conf_thresh)
            l.append(loglist['mIoU'])
            if no_bg:
                print('%d/%d threshold: %.3f\tmIoU: %.3f%%'%(i, args.end, t, loglist['mIoU']))
            else:
                print('%d/%d background score: %.3f\tmIoU: %.3f%%'%(i, args.end, t, loglist['mIoU']))
            # 舊版：第一次「沒嚴格進步」就 break，遇到 plateau（如 t=0 與 t=0.01 數值相同）會卡死
            # if loglist['mIoU'] > max_mIoU:
            #     max_mIoU = loglist['mIoU']
            #     best_thr = t
            # else:
            #     break

            # 新版：跑完整個 threshold range，全局取最高值
            # if loglist['mIoU'] > max_mIoU:
            #     max_mIoU = loglist['mIoU']
            #     best_thr = t

            # 2026-05-19：DeepGlobe 64200 tile 單次 eval ~1.5h，跑 60 個 threshold = 90h
            # no_background mode 下 threshold 根本沒被用到（line 111-116 直接 argmax），
            # 必然每個 threshold 都一樣。加早停：連續 3 個 threshold mIoU 完全相同 → break
            if loglist['mIoU'] > max_mIoU:
                max_mIoU = loglist['mIoU']
                best_thr = t
            mIoU_history.append(loglist['mIoU'])
            if len(mIoU_history) >= 3 and mIoU_history[-1] == mIoU_history[-2] == mIoU_history[-3]:
                print('Early stop: 連續 3 個 threshold mIoU=%.6f%% 完全相同' % loglist['mIoU'])
                break
        print('Best threshold: %.3f\tmIoU: %.3f%%' % (best_thr, max_mIoU))
        # Re-run with best threshold to get per-class IoU
        print('\n--- Per-class IoU at best threshold ---')
        best_loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, args.num_classes, args.type, best_thr, printlog=True, no_background=no_bg, cat_names=cat_names, out_crf=out_crf, img_dir=img_dir, out_dir=out_dir, ignore_clutter=args.ignore_clutter, conf_thresh=args.conf_thresh)
        writelog(args.logfile, best_loglist, args.comment)
        writelog_pretty(args.predict_dir, best_loglist, args.comment, best_thr=best_thr)

