#!/usr/bin/env python3
"""
用 fvcore 量 PixelResidualStem（PR 模組）帶來的計算量。

做兩件事：
  1) 建「有 PR」的完整模型，用 FlopCountAnalysis 量整模型，
     再用 by_module() 抓出 `pixel_residual` 那一項 → PR 模組在模型內的實際 FLOPs
  2) 建「無 PR」的 baseline 模型對照 → 兩者差值 = PR 淨增量（交叉驗證）

注意：fvcore 的 "flops" 實際是 MACs（乘加各算一次），
      要換成傳統 FLOPs 定義 ×2。下面同時印 MACs 與參數量。

用法：
  python count_flops_pr.py                 # 預設 DeepGlobe pr_ni (instance, 448, 3ch, 6cls)
  python count_flops_pr.py --input-size 448 --num-classes 6 --in-chans 3 --pr-norm instance
"""
import argparse
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count
from timm.models import create_model
import models  # noqa: F401  # 註冊 deit_small_MCTformerPlus


def build(use_pr, args):
    return create_model(
        'deit_small_MCTformerPlus',
        pretrained=False,
        num_classes=args.num_classes,
        drop_rate=0.0, drop_path_rate=0.0, drop_block_rate=None,
        input_size=args.input_size,
        use_pixel_residual=use_pr,
        use_multi_stage_pr=False,
        use_stacked_pr=False,
        stacked_pr_layers=2,
        pr_kernels=args.pr_kernels,
        pr_norm=args.pr_norm,
        use_branch_norm=args.use_branch_norm,
        in_chans=args.in_chans,
    ).eval()


def analyze(model, x):
    fa = FlopCountAnalysis(model, x)
    fa.unsupported_ops_warnings(False)
    fa.uncalled_modules_warnings(False)
    return fa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-size', type=int, default=448)
    ap.add_argument('--num-classes', type=int, default=6)
    ap.add_argument('--in-chans', type=int, default=3)
    ap.add_argument('--pr-norm', type=str, default='instance')
    ap.add_argument('--pr-kernels', nargs='+', type=int, default=[5, 7])
    ap.add_argument('--use-branch-norm', action='store_true')
    args = ap.parse_args()

    x = torch.randn(1, args.in_chans, args.input_size, args.input_size)
    print(f"input = 1x{args.in_chans}x{args.input_size}x{args.input_size}, "
          f"num_classes={args.num_classes}, pr_norm={args.pr_norm}\n")

    G, M = 1e9, 1e6
    m_pr = build(True, args)
    pc = parameter_count(m_pr)
    tot_p = pc['']
    pr_p = pc.get('pixel_residual', 0)

    # ---- PR 模組本身的計算量：只 trace pixel_residual（避開 ViT pos-embed interpolate 的 trace 坑）----
    pr_flops = analyze(m_pr.pixel_residual, x).total()

    print("================ 計算量（fvcore, 單位=MACs）================")
    print(f"PR 模組            : {pr_flops/G:9.4f} GMACs | {pr_p/M:8.4f} M params")
    print(f"（傳統 FLOPs ≈ MACs×2；PR ≈ {pr_flops*2/G:.4f} GFLOPs）\n")

    # ---- 完整模型參數量（不需 trace）+ 佔比 ----
    base_p = parameter_count(build(False, args))['']
    print(f"完整模型 +PR params: {tot_p/M:8.4f} M   |  無PR: {base_p/M:8.4f} M")
    print(f"PR 參數佔比        : {pr_p/tot_p*100:6.3f}%   (淨增 {(tot_p-base_p)/M:.4f} M)")

    # ---- 嘗試量完整模型 MACs（ViT pos-embed 在 trace 下可能報錯，失敗就跳過）----
    try:
        total = analyze(m_pr, x).total()
        print(f"\n完整模型 +PR       : {total/G:9.4f} GMACs")
        print(f"PR MACs 佔比       : {pr_flops/total*100:6.3f}%")
    except Exception as e:
        print(f"\n[完整模型 MACs 略過] ViT pos-embed interpolate 無法 jit-trace："
              f" {type(e).__name__}")
        print(" → PR 絕對 MACs 已量出（上方）；如需佔比可另用 hook-based 工具(ptflops/thop)。")


if __name__ == '__main__':
    main()
