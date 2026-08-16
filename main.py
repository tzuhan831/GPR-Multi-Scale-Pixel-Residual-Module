import argparse
import datetime
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler

from datasets import build_dataset
from engine import train_one_epoch, evaluate, generate_attention_maps_ms
import models
import utils
import os
import numpy as np
import random

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=45, type=int)

    # Model parameters
    parser.add_argument('--model', default='deit_small_MCTformerPlus', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=448, type=int, help='448')
    parser.add_argument('--image-ch', dest='image_ch', default=3, type=int,
                        help='input channels (3=RGB, 4=RGB+extra channel)')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')


    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data-path', default='VOCdevkit/VOC2012', type=str, help='dataset path')
    parser.add_argument('--img-list', default='voc12', type=str, help='image list path')
    parser.add_argument('--data-set', default='VOC12', type=str, help='dataset')


    parser.add_argument('--output_dir', default='saved_model',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)


    # generating attention maps
    parser.add_argument('--gen_attention_maps', default=False, action='store_true')
    parser.add_argument('--cam-on-val', default=False, action='store_true',
                        help='gen_attention_maps 時用 val_id.txt 而非 train_aug_id.txt（搭配 --gen_attention_maps 使用）')
    parser.add_argument('--patch-size', type=int, default=16)
    parser.add_argument('--attention-dir', type=str, default='cam-png')
    parser.add_argument('--layer-index', type=int, default=12, help='extract attention maps from the last layers')

    parser.add_argument('--patch-attn-refine', type=bool, default=True)
    parser.add_argument('--visualize-cls-attn', type=bool, default=True)

    parser.add_argument('--cam-npy-dir', type=str, default='cam-npy')
    parser.add_argument("--scales", nargs='+', type=float, default=[1.0,0.75,1.25])
    parser.add_argument('--label-file-path', type=str, default=None)
    parser.add_argument('--attention-type', type=str, default='fused')


    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument("--loss-weight", default=1.0, type=float)
    parser.add_argument("--num-cct", default=12, type=int)


    parser.add_argument('--use-pixel-residual', action='store_true', default=False,
                        help='在 PatchEmbed 前插入 PixelResidualStem：'
                             '多尺度修正量以加法殘差疊加回原圖（x + g * correction），'
                             '原圖光譜資訊完整保留。')
    parser.add_argument('--use-multi-stage-pr', action='store_true', default=False,
                        help='使用 MultiStagePixelResidualStem：多層串聯，線性 proj。'
                             '搭配 --pr-kernels 指定各層 k_large。')
    parser.add_argument('--use-stacked-pr', action='store_true', default=False,
                        help='使用 StackedPixelResidualStem：每層都是 (1, 3, 5) 的多層串聯，'
                             '層數由 --stacked-pr-layers 控制（預設 2）。')
    parser.add_argument('--stacked-pr-layers', type=int, default=2,
                        help='StackedPixelResidualStem 的層數（預設 2，常用 2 或 3）。')
    parser.add_argument('--pr-kernels', nargs='+', type=int, default=[5, 7],
                        help='MultiStagePixelResidualStem 各層的 k_large（預設 5 9）。'
                             '層數由值的數量決定。例如 5 9 → Layer0(1,3,5) + Layer1(1,5,9)。')
    parser.add_argument('--pr-norm', type=str, default='instance',
                        choices=['instance', 'batch', 'group', 'none'],
                        help='PRS correction 的正規化方式（預設 instance）。'
                             'instance=InstanceNorm2d, batch=BatchNorm2d, '
                             'group=GroupNorm(1,C)（≈LayerNorm）, none=Identity。')
    parser.add_argument('--use-branch-norm', action='store_true', default=False,
                        help='V2: 在 PixelResidualStem 每條 branch 後加 InstanceNorm2d(affine=True)，'
                             '解決 4-channel 輸入時 slope DC bias 跨 branch 累積不一致的問題。')
    parser.add_argument('--pr-13', dest='pr_13', action='store_true', default=False,
                        help='PixelResidualStem 只用 (1×1, 3×3) 兩條 branch，完全拿掉 5×5 那條'
                             '（最大感受野縮到 3×3）。與 --pr-strip 互斥（--pr-13 優先）。'
                             'train 與 gen_attention_maps 必須一致，否則 load_state_dict 出錯。')
    parser.add_argument('--pr-strip', dest='pr_strip', action='store_true', default=False,
                        help='PixelResidualStem 的 branch5 換成「depthwise 可分離序列」'
                             '1×1 lift → dw 1×5 → dw 5×1（SegNeXt MSCA 風格，論文表12「strip5」）。'
                             '重建完整 5×5 方形感受野（含對角），算力比 full5 省。'
                             '優先序 k13 > strip。'
                             'train 與 gen_attention_maps 必須一致，否則 load_state_dict 出錯。')
    parser.add_argument('--pr-lr-mult', type=float, default=1.0,
                        help='PR 模組（pixel_residual.*）的 lr 倍率（預設 1.0=與 backbone 同 lr）。'
                             '>1 時 PR 走獨立 param group，lr×mult 且 weight_decay=0。'
                             'Postdam 經驗：mult=1 比 mult=10 高 ~1.0 mIoU。')

    return parser


def main(args):

    print(args)

    device = torch.device(args.device)

    seed = args.seed
    # cudnn.benchmark = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    # dataset_train_, args.nb_classes = build_dataset(is_train=False, gen_attn=True, args=args)
    # 加 --cam-on-val 時讓 dataset 走「train=False, gen_attn=False」分支自動撿 val_id.txt
    dataset_train_, args.nb_classes = build_dataset(
        is_train=False, gen_attn=not args.cam_on_val, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_train_ = torch.utils.data.DataLoader(
        dataset_train_,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    print(f"Creating model: {args.model}")


    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        input_size=args.input_size,
        use_pixel_residual=args.use_pixel_residual,
        use_multi_stage_pr=args.use_multi_stage_pr,
        use_stacked_pr=args.use_stacked_pr,
        stacked_pr_layers=args.stacked_pr_layers,
        pr_kernels=args.pr_kernels,
        pr_norm=args.pr_norm,
        use_branch_norm=args.use_branch_norm,
        pr_k13=args.pr_13,
        pr_strip=args.pr_strip,
        in_chans=args.image_ch,
    )


    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        try:
            checkpoint_model = checkpoint['model']
        except:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # 4-channel (RGB+slope) patch_embed.proj weight 擴充（對齊 FRD 作法）
        if args.image_ch != 3 and 'patch_embed.proj.weight' in checkpoint_model:
            proj_weight = checkpoint_model['patch_embed.proj.weight']
            if proj_weight.shape[1] == 3:
                new_weight = torch.zeros(proj_weight.shape[0], args.image_ch,
                                         proj_weight.shape[2], proj_weight.shape[3])
                new_weight[:, :3, :, :] = proj_weight
                torch.nn.init.xavier_uniform_(new_weight[:, 3:, :, :])
                checkpoint_model['patch_embed.proj.weight'] = new_weight
                print(f"Expanded patch_embed.proj.weight: 3 -> {args.image_ch} channels")

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        # 舊版：從 patch_embed.num_patches 取，但 MCTformerPlus 的 patch_embed 是 parent ViT 用
        # default img_size=224 建的（永遠 196），與真正 pos_embed_pat 大小不符
        # num_patches = model.patch_embed.num_patches
        # 新版：直接從 model.pos_embed_pat 取真正目標 token 數
        num_patches = model.pos_embed_pat.shape[1]
        if args.finetune.startswith('https'):
            num_extra_tokens = 1
        else:
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches

        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches ** 0.5)

        if args.finetune.startswith('https') and 'MCTformer' in args.model:
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens].repeat(1,args.nb_classes,1)
        else:
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]

        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]

        # 舊版（B）：無條件 2× nearest-repeat（landslide 224 會壞）
        # pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        # pos_tokens = pos_tokens[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(
        #     pos_tokens.size(0), pos_tokens.size(1), pos_tokens.size(2)*2, pos_tokens.size(3) *2)
        # pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)

        # 新版（A+B）：用真正的 num_patches 算 new_size，分情況處理
        if new_size == orig_size:
            # 224 配 deit_224：pretrained 直接用
            pass
        elif new_size == orig_size * 2:
            # 448 配 deit_224（Postdam/Vaihingen 標準路徑）：2× nearest-repeat
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = pos_tokens[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(
                pos_tokens.size(0), pos_tokens.size(1), pos_tokens.size(2)*2, pos_tokens.size(3) *2)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        else:
            # 任意尺寸：bicubic
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)

        checkpoint_model['pos_embed_cls'] = extra_tokens
        checkpoint_model['pos_embed_pat'] = pos_tokens

        if args.finetune.startswith('https') and 'MCTformer' in args.model:
            cls_token_checkpoint = checkpoint_model['cls_token']
            new_cls_token = cls_token_checkpoint.repeat(1,args.nb_classes,1)
            checkpoint_model['cls_token'] = new_cls_token

        model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr

    # 舊版（已併入下面 if 分支）：PR params 用 lr×10 + weight_decay=0
    # 上一版的 unconditional create_optimizer：
    # optimizer = create_optimizer(args, model)

    # 新版：依 --pr-lr-mult 決定走哪條路徑
    pr_active = (args.use_pixel_residual or args.use_multi_stage_pr
                 or args.use_stacked_pr)
    if pr_active and args.pr_lr_mult != 1.0:
        # PR 模組獨立 param group：lr × mult，weight_decay=0
        pr_params = list(model.pixel_residual.parameters())
        pr_ids = {id(p) for p in pr_params}
        base_params = [p for p in model.parameters() if id(p) not in pr_ids]
        optimizer = torch.optim.AdamW([
            {'params': base_params, 'weight_decay': args.weight_decay},
            {'params': pr_params,   'lr': args.lr * args.pr_lr_mult, 'weight_decay': 0.0},
        ], lr=args.lr, eps=args.opt_eps)
        print(f"[Optimizer] PR params using lr×{args.pr_lr_mult} (={args.lr*args.pr_lr_mult:.2e}), WD=0")
    else:
        # 預設路徑：所有 param 同 lr，timm 自動處理 WD（norm/bias 不上 WD）
        optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    output_dir = Path(args.output_dir)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"mAP of the network on the {len(dataset_val)} test images: {test_stats['mAP']*100:.1f}%")
        return

    if args.gen_attention_maps:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        generate_attention_maps_ms(data_loader_train_, model, device, args)
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad,
            args=args
        )

        lr_scheduler.step(epoch)

        if pr_active:
            if args.use_multi_stage_pr or args.use_stacked_pr:
                for si, gate in enumerate(model.pixel_residual.gates):
                    gv = torch.sigmoid(gate).flatten().tolist()
                    if len(gv) == 1:
                        print(f"[Epoch {epoch}] PR stage{si} gate: {gv[0]:.4f}")
                    else:
                        print(f"[Epoch {epoch}] PR stage{si} gates: R={gv[0]:.4f} G={gv[1]:.4f} B={gv[2]:.4f}")
            else:
                pr_gates = torch.sigmoid(model.pixel_residual.gate).flatten().tolist()
                if len(pr_gates) == 1:
                    print(f"[Epoch {epoch}] PR gate: {pr_gates[0]:.4f}")
                else:
                    print(f"[Epoch {epoch}] PR gates: R={pr_gates[0]:.4f} G={pr_gates[1]:.4f} B={pr_gates[2]:.4f}")

        test_stats = evaluate(data_loader_val, model, device)

        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                ckpt = {'model': model.state_dict(), 'epoch': epoch}
                utils.save_on_master(ckpt, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if pr_active:
            if args.use_multi_stage_pr or args.use_stacked_pr:
                for si, gate in enumerate(model.pixel_residual.gates):
                    gv = torch.sigmoid(gate).flatten().tolist()
                    for ci, v in enumerate(gv):
                        log_stats[f'pr_stage{si}_gate_ch{ci}'] = v
            else:
                pr_gates = torch.sigmoid(model.pixel_residual.gate).flatten().tolist()
                if len(pr_gates) == 1:
                    log_stats['pr_gate'] = pr_gates[0]
                else:
                    for i, v in enumerate(pr_gates):
                        log_stats[f'pr_gate_ch{i}'] = v

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    torch.save({'model': model.state_dict()}, output_dir / 'checkpoint.pth')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)