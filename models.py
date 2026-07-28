import torch
import torch.nn as nn
from functools import partial
from vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, to_2tuple
import torch.nn.functional as F

import math

__all__ = ['deit_small_MCTformerPlus']


def _build_norm(norm_type, num_channels):
    """根據 norm_type 字串建立對應的正規化層。"""
    if norm_type == 'instance':
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm_type == 'batch':
        return nn.BatchNorm2d(num_channels, affine=True)
    elif norm_type == 'group':
        return nn.GroupNorm(1, num_channels, affine=True)
    elif norm_type == 'none':
        return nn.Identity()
    else:
        raise ValueError(f'Unknown norm_type: {norm_type}')


class PixelResidualStem(nn.Module):
    """Pixel-space multi-scale stem with additive residual connection.
    Per-channel gate：每個 RGB channel 有獨立的修正強度控制。

    use_branch_norm (V2): 在每條 branch 後加 InstanceNorm（per-branch scale alignment）。
        解決 4-channel 輸入時 slope 的 DC bias 跨 branch 累積不一致的問題。
        affine=True 讓模型可以學會反推 IN，最差退回 V1（無此 IN）行為。
    """

    def __init__(self, in_chans=3, mid_chans=32, norm_type='instance', use_branch_norm=False,
                 k13=False, strip=False):
        super().__init__()
        self.branch1 = nn.Conv2d(in_chans, mid_chans, kernel_size=1)
        self.branch3 = nn.Conv2d(in_chans, mid_chans, kernel_size=3, padding=1)
        # k13：只留 1×1 + 3×3 兩條，完全拿掉 5×5 那條（分支從 3 → 2，concat 為 mid×2）
        #   motivation：把最大感受野縮到 3×3，測「更小 kernel」對密集小物件是否更好
        # strip（論文表12「strip5」）：把 full 5×5 換成「depthwise 可分離序列」
        #   1×1 lift(3→mid) → dw 1×5 → dw 5×1（SegNeXt MSCA 風格）。
        #   重建完整 5×5 方形感受野（含對角，rank-1 可分離），
        #   但 depthwise 使算力最省（此格 MAC/px ≈ 416，比 full5 2400 少）。
        #   輸出仍是 mid_chans → n_branch=3、proj 寬度與 full5 完全相同（最乾淨的方形 ablation）。
        self.k13 = k13
        self.strip = strip and not k13                   # 優先序：k13 > strip > full5
        if self.k13:
            n_branch = 2                    # 只有 branch1 + branch3
        elif self.strip:
            self.branch5_lift = nn.Conv2d(in_chans, mid_chans, kernel_size=1)                                       # 3→mid 通道 lift
            self.branch5_h = nn.Conv2d(mid_chans, mid_chans, kernel_size=(1, 5), padding=(0, 2), groups=mid_chans)  # dw 1×5
            self.branch5_v = nn.Conv2d(mid_chans, mid_chans, kernel_size=(5, 1), padding=(2, 0), groups=mid_chans)  # dw 5×1
            n_branch = 3
        else:
            self.branch5 = nn.Conv2d(in_chans, mid_chans, kernel_size=5, padding=2)  # 原版 full 5×5
            n_branch = 3

        # V2 新增：per-branch InstanceNorm（寫死 InstanceNorm2d，不走 norm_type）
        self.use_branch_norm = use_branch_norm
        if use_branch_norm:
            self.bn1 = nn.InstanceNorm2d(mid_chans, affine=True)
            self.bn3 = nn.InstanceNorm2d(mid_chans, affine=True)
            if self.k13:
                pass                        # 無 branch5，不建 bn5
            elif self.strip:
                self.bn5 = nn.InstanceNorm2d(mid_chans, affine=True)   # 施加在序列 strip 的輸出上
            else:
                self.bn5 = nn.InstanceNorm2d(mid_chans, affine=True)

        self.proj = nn.Conv2d(mid_chans * n_branch, in_chans, kernel_size=1)
        # self.norm = nn.InstanceNorm2d(in_chans, affine=True)  # 原版固定 InstanceNorm
        self.norm = _build_norm(norm_type, in_chans)
        # self.gate = nn.Parameter(torch.full((in_chans, 1, 1), -3.0))  # per-channel gate
        self.gate = nn.Parameter(torch.tensor(-3.0))  # scalar gate（好的/ 版本）

    def forward(self, x):
        if self.use_branch_norm:
            feats = [self.bn1(self.branch1(x)), self.bn3(self.branch3(x))]
            if self.k13:
                pass                        # 只有 1×1 + 3×3
            elif self.strip:
                b5 = self.branch5_v(self.branch5_h(self.branch5_lift(x)))   # 序列串接
                feats += [self.bn5(b5)]
            else:
                feats += [self.bn5(self.branch5(x))]
        else:
            feats = [self.branch1(x), self.branch3(x)]
            if self.k13:
                pass                        # 只有 1×1 + 3×3
            elif self.strip:
                feats += [self.branch5_v(self.branch5_h(self.branch5_lift(x)))]  # 序列串接
            else:
                feats += [self.branch5(x)]
        correction = self.proj(torch.cat(feats, dim=1))
        correction = self.norm(correction)
        return x + torch.sigmoid(self.gate) * correction





class MultiStagePixelResidualStem(nn.Module):
    """多層串聯式 PixelResidualStem。

    每層三條 branch：k=1（固定）、k_medium（繼承上層 k_large）、k_large。
    單步線性投影：96 → in_chans（GELU 兩步版測過 -7 mIoU 已棄用）。
    Gate 初始值平緩遞減（-3, -3.5, -4, ...），各 stage 都能收到有效梯度。

    Args:
        kernel_larges: list[int]，每層的 k_large，層數由長度決定。
            預設 [5, 7] → 2 層：Layer0(1,3,5), Layer1(1,5,7)
        in_chans: 輸入 channel 數（預設 3）
        mid_chans: 每條 branch 的輸出 channel 數（預設 32）
    """

    def __init__(self, kernel_larges=(5, 7), in_chans=3, mid_chans=32, norm_type='instance'):
        super().__init__()
        self.num_stages = len(kernel_larges)
        self.stages = nn.ModuleList()
        self.gates = nn.ParameterList()

        k_medium = 3  # 第一層固定
        for i, k_large in enumerate(kernel_larges):
            stage = nn.ModuleDict({
                'branch1': nn.Conv2d(in_chans, mid_chans, kernel_size=1),
                'branch_m': nn.Conv2d(in_chans, mid_chans, kernel_size=k_medium,
                                      padding=k_medium // 2),
                'branch_l': nn.Conv2d(in_chans, mid_chans, kernel_size=k_large,
                                      padding=k_large // 2),
                'proj': nn.Conv2d(mid_chans * 3, in_chans, kernel_size=1),
                # 'norm': nn.InstanceNorm2d(in_chans, affine=True),  # 原版固定 InstanceNorm
                'norm': _build_norm(norm_type, in_chans),
            })
            self.stages.append(stage)

            # 舊版：gate_init = -3.0 - 2.0 * i  # -3, -5, -7（deeper layer 幾乎死掉）
            gate_init = -3.0 - 0.5 * i  # -3, -3.5, -4（讓 deeper layer 也能學）
            self.gates.append(nn.Parameter(torch.full((in_chans, 1, 1), gate_init)))

            k_medium = k_large  # 下一層的 k_medium 繼承本層 k_large

    def forward(self, x):
        for stage, gate in zip(self.stages, self.gates):
            feat = torch.cat([stage['branch1'](x),
                              stage['branch_m'](x),
                              stage['branch_l'](x)], dim=1)
            correction = stage['proj'](feat)
            correction = stage['norm'](correction)
            x = x + torch.sigmoid(gate) * correction
        return x


class StackedPixelResidualStem(nn.Module):
    """每層都是 (1, 3, 5) 三條 branch 的多層串聯式 PR。

    與 MultiStagePixelResidualStem 不同之處：
      - MSPR 各層 kernel 不同，遞增（如 (1,3,5)→(1,5,7)）
      - StackedPR 各層 kernel **完全相同**：每層都是 (1, 3, 5)
        → 同尺度修正反覆套用，每層只負責「微調」，不擴大感受野

    Args:
        num_stages: 層數（2 或 3 或更多）
        in_chans: 輸入 channel 數
        mid_chans: 每條 branch 輸出 channel 數
        norm_type: 'instance' / 'batch' / 'group' / 'none'
    """

    def __init__(self, num_stages=2, in_chans=3, mid_chans=32, norm_type='instance'):
        super().__init__()
        self.num_stages = num_stages
        self.stages = nn.ModuleList()
        self.gates = nn.ParameterList()

        for i in range(num_stages):
            stage = nn.ModuleDict({
                'branch1': nn.Conv2d(in_chans, mid_chans, kernel_size=1),
                'branch3': nn.Conv2d(in_chans, mid_chans, kernel_size=3, padding=1),
                'branch5': nn.Conv2d(in_chans, mid_chans, kernel_size=5, padding=2),
                'proj': nn.Conv2d(mid_chans * 3, in_chans, kernel_size=1),
                'norm': _build_norm(norm_type, in_chans),
            })
            self.stages.append(stage)
            # 跟 MSPR 一樣的平緩遞減 -3, -3.5, -4, ...
            gate_init = -3.0 - 0.5 * i
            self.gates.append(nn.Parameter(torch.full((in_chans, 1, 1), gate_init)))

    def forward(self, x):
        for stage, gate in zip(self.stages, self.gates):
            feat = torch.cat([stage['branch1'](x),
                              stage['branch3'](x),
                              stage['branch5'](x)], dim=1)
            correction = stage['proj'](feat)
            correction = stage['norm'](correction)
            x = x + torch.sigmoid(gate) * correction
        return x


class MCTformerPlus(VisionTransformer):
    def __init__(self, decay_parameter=0.996, input_size=244,
                 use_pixel_residual=False,
                 use_multi_stage_pr=False, pr_kernels=(5, 9),
                 pr_norm='instance', use_branch_norm=False,
                 use_stacked_pr=False, stacked_pr_layers=2,
                 pr_k13=False, pr_strip=False,
                 *args, **kwargs):
        in_chans = kwargs.get('in_chans', 3)
        super().__init__(*args, **kwargs)

        self.use_pixel_residual = (use_pixel_residual or use_multi_stage_pr
                                   or use_stacked_pr)
        self.use_multi_stage_pr = use_multi_stage_pr
        self.use_stacked_pr = use_stacked_pr
        if use_stacked_pr:
            self.pixel_residual = StackedPixelResidualStem(
                num_stages=stacked_pr_layers, in_chans=in_chans, norm_type=pr_norm)
        elif use_multi_stage_pr:
            self.pixel_residual = MultiStagePixelResidualStem(
                kernel_larges=pr_kernels, in_chans=in_chans, norm_type=pr_norm)
        elif use_pixel_residual:
            self.pixel_residual = PixelResidualStem(
                in_chans=in_chans, norm_type=pr_norm, use_branch_norm=use_branch_norm,
                k13=pr_k13, strip=pr_strip)

        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.head.apply(self._init_weights)

        img_size = to_2tuple(input_size)
        patch_size = to_2tuple(self.patch_embed.patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed_cls = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed_pat = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed_cls, std=.02)
        trunc_normal_(self.pos_embed_pat, std=.02)
        print(self.training)
        self.decay_parameter=decay_parameter


    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - self.num_classes
        N = self.num_patches
        if npatch == N and w == h:
            return self.pos_embed_pat
        patch_pos_embed = self.pos_embed_pat
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]

        patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
            )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward_features(self, x, n=12):
        B, nc, w, h = x.shape

        if self.use_pixel_residual:
            x = self.pixel_residual(x)

        x = self.patch_embed(x)
        h_featmap = h // self.patch_embed.patch_size[0]
        w_featmap = w // self.patch_embed.patch_size[0]

        if not self.training:
            pos_embed_pat = self.interpolate_pos_encoding(x, w, h)
            x = x + pos_embed_pat
        else:
            x = x + self.pos_embed_pat

        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_tokens = cls_tokens + self.pos_embed_cls

        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        attn_weights    = []
        class_embeddings = []

        for i, blk in enumerate(self.blocks):
            x, weights_i = blk(x)
            attn_weights.append(weights_i)
            class_embeddings.append(x[:, 0:self.num_classes])

        return x[:, 0:self.num_classes], x[:, self.num_classes:], attn_weights, class_embeddings

    def forward(self, x, return_att=False, forward_feat=False, use_fine=False,
                n_layers=12, attention_type='fused'):
        w, h = x.shape[2:]
        x_cls, x_patch, attn_weights, all_x_cls = self.forward_features(x)

        n, p, c = x_patch.shape
        if w != h:
            w0 = w // self.patch_embed.patch_size[0]
            h0 = h // self.patch_embed.patch_size[0]
            x_patch = torch.reshape(x_patch, [n, w0, h0, c])
        else:
            x_patch = torch.reshape(x_patch, [n, int(p ** 0.5), int(p ** 0.5), c])
        x_patch = x_patch.permute([0, 3, 1, 2])
        feat = x_patch.contiguous()             # (B, embed_dim, H, W) for PPC
        x_patch = self.head(feat)              # main_cam: (B, num_classes, h_feat, w_feat)

        x_patch_flattened = x_patch.view(x_patch.shape[0], x_patch.shape[1], -1).permute(0, 2, 1)

        sorted_patch_token, indices = torch.sort(x_patch_flattened, -2, descending=True)
        weights = torch.logspace(start=0, end=x_patch_flattened.size(-2) - 1,
                                  steps=x_patch_flattened.size(-2), base=self.decay_parameter).cuda()
        x_patch_logits = torch.sum(sorted_patch_token * weights.unsqueeze(0).unsqueeze(-1), dim=-2) / weights.sum()

        x_cls_logits = x_cls.mean(-1)

        output = [x_cls_logits, torch.stack(all_x_cls), x_patch_logits]

        if return_att:
            feature_map = x_patch.detach().clone()  # B * C * 14 * 14
            feature_map = F.relu(feature_map)
            n, c, h, w = feature_map.shape

            attn_weights = torch.stack([aw.mean(dim=1) for aw in attn_weights])  # 12 * B * N * N
            mtatt = attn_weights[-n_layers:].mean(0)[:, 0:self.num_classes, self.num_classes:].reshape([n, c, h, w])
            patch_attn = attn_weights[:, :, self.num_classes:, self.num_classes:]
            if attention_type == 'fused':
                cams = mtatt * feature_map  # B * C * 14 * 14
                cams = torch.sqrt(cams)
            elif attention_type == 'patchcam':
                cams = feature_map
            elif attention_type == 'mct':
                cams = mtatt
            else:
                raise f'Error! {attention_type} is not defined!'

            x_logits = (x_cls_logits + x_patch_logits) / 2
            return x_logits, cams, patch_attn
        elif forward_feat:
            coarse_cam = x_patch.clone()
            if use_fine:
                feature_map = x_patch.detach().clone()
                feature_map = F.relu(feature_map)
                n, c, h, w = feature_map.shape

                attn_weights = torch.stack([aw.mean(dim=1) for aw in attn_weights])  # 12 * B * N * N
                mtatt = attn_weights[-n_layers:].mean(0)[:, 0:self.num_classes, self.num_classes:].reshape([n, c, h, w])
                cams = mtatt * feature_map
                cams = torch.sqrt(cams)

                patch_attn = attn_weights[:, :, self.num_classes:, self.num_classes:]
                patch_attn_sum = torch.sum(patch_attn, dim=0)  # B * HW * HW
                fine_cam = torch.matmul(
                    patch_attn_sum.unsqueeze(1),
                    cams.view(cams.shape[0], cams.shape[1], -1, 1)
                ).reshape(cams.shape)
                return x_cls_logits, torch.stack(all_x_cls), x_patch_logits, feat, fine_cam
            else:
                return x_cls_logits, torch.stack(all_x_cls), x_patch_logits, feat, coarse_cam
        else:
            return output

@register_model
def deit_small_MCTformerPlus(pretrained=False, **kwargs):
    model = MCTformerPlus(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['cls_token', 'pos_embed']}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model