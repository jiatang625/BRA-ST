import timm
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models import register_model
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import _cfg
from torch.nn import Flatten
from ops.bra_legacy import BiLevelRoutingAttention
from _common import Attention, AttentionLePE, DWConv
def get_pe_layer(emb_dim, pe_dim=None, name='none'):
    if name == 'none':
        return nn.Identity()
    else:
        raise ValueError(f'PE name {name} is not surpported!')
class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=-1,
                 num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 mlp_ratio=4, mlp_dwconv=False,
                 side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=False):
        super().__init__()
        qk_dim = qk_dim or dim
        # modules
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # important to avoid attention collapsing
        if topk > 0:
            self.attn = BiLevelRoutingAttention(dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                                                qk_scale=qk_scale, kv_per_win=kv_per_win,
                                                kv_downsample_ratio=kv_downsample_ratio,
                                                kv_downsample_kernel=kv_downsample_kernel,
                                                kv_downsample_mode=kv_downsample_mode,
                                                topk=topk, param_attention=param_attention, param_routing=param_routing,
                                                diff_routing=diff_routing, soft_routing=soft_routing,
                                                side_dwconv=side_dwconv,
                                                auto_pad=auto_pad)
        elif topk == -1:
            self.attn = Attention(dim=dim)
        elif topk == -2:
            self.attn = AttentionLePE(dim=dim, side_dwconv=side_dwconv)
        elif topk == 0:
            self.attn = nn.Sequential(Rearrange('n h w c -> n c h w'),  # compatiability
                                      nn.Conv2d(dim, dim, 1),  # pseudo qkv linear
                                      nn.Conv2d(dim, dim, 5, padding=2, groups=dim),  # pseudo attention
                                      nn.Conv2d(dim, dim, 1),  # pseudo out linear
                                      Rearrange('n c h w -> n h w c')
                                      )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio * dim)),
                                 DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio * dim), dim)
                                 )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # tricks: layer scale & pre_norm/post_norm
        if layer_scale_init_value > 0:
            self.use_layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.use_layer_scale = False
        self.pre_norm = pre_norm

    def forward(self, x):
        """
        x: NCHW tensor
        """
        # conv pos embedding
        x = x + self.pos_embed(x)
        # permute to NHWC tensor for attention & mlp
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        # attention & mlp
        if self.pre_norm:
            if self.use_layer_scale:
                x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))  # (N, H, W, C)
            else:
                x = x + self.drop_path(self.attn(self.norm1(x)))  # (N, H, W, C)
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # (N, H, W, C)
        else:  # https://kexue.fm/archives/9009
            if self.use_layer_scale:
                x = self.norm1(x + self.drop_path(self.gamma1 * self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.gamma2 * self.mlp(x)))  # (N, H, W, C)
            else:
                x = self.norm1(x + self.drop_path(self.attn(x)))  # (N, H, W, C)
                x = self.norm2(x + self.drop_path(self.mlp(x)))  # (N, H, W, C)

        # permute back
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x
class BiFormer(nn.Module):
    def __init__(self, depth=[3, 4, 8, 3], in_chans=3, num_classes=1000, embed_dim=[64, 128, 320, 512],
                 head_dim=64, qk_scale=None, representation_size=None,
                 drop_path_rate=0., drop_rate=0.,
                 use_checkpoint_stages=[],
                 ########
                 n_win=7,
                 kv_downsample_mode='ada_avgpool',
                 kv_per_wins=[2, 2, -1, -1],
                 topks=[8, 8, -1, -1],
                 side_dwconv=5,
                 layer_scale_init_value=-1,
                 qk_dims=[None, None, None, None],
                 param_routing=False, diff_routing=False, soft_routing=False,
                 pre_norm=True,
                 pe=None,
                 pe_stages=[0],
                 before_attn_dwconv=3,
                 auto_pad=False,
                 # -----------------------
                 kv_downsample_kernels=[4, 2, 1, 1],
                 kv_downsample_ratios=[4, 2, 1, 1],  # -> kv_per_win = [2, 2, 2, 1]
                 mlp_ratios=[4, 4, 4, 4],
                 param_attention='qkvo',
                 mlp_dwconv=False):

        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        ############ downsample layers (patch embeddings) ######################
        self.downsample_layers = nn.ModuleList()
        # NOTE: uniformer uses two 3*3 conv, while in many other transformers this is one 7*7 conv
        stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0] // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim[0] // 2, embed_dim[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0]),
        )
        if (pe is not None) and 0 in pe_stages:
            stem.append(get_pe_layer(emb_dim=embed_dim[0], name=pe))
        if use_checkpoint_stages:
            stem = checkpoint_wrapper(stem)
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv2d(embed_dim[i], embed_dim[i + 1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(embed_dim[i + 1])
            )
            if (pe is not None) and i + 1 in pe_stages:
                downsample_layer.append(get_pe_layer(emb_dim=embed_dim[i + 1], name=pe))
            if use_checkpoint_stages:
                downsample_layer = checkpoint_wrapper(downsample_layer)
            self.downsample_layers.append(downsample_layer)
        ##########################################################################
        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        nheads = [dim // head_dim for dim in qk_dims]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=embed_dim[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        topk=topks[i],
                        num_heads=nheads[i],
                        n_win=n_win,
                        qk_dim=qk_dims[i],
                        qk_scale=qk_scale,
                        kv_per_win=kv_per_wins[i],
                        kv_downsample_ratio=kv_downsample_ratios[i],
                        kv_downsample_kernel=kv_downsample_kernels[i],
                        kv_downsample_mode=kv_downsample_mode,
                        param_attention=param_attention,
                        param_routing=param_routing,
                        diff_routing=diff_routing,
                        soft_routing=soft_routing,
                        mlp_ratio=mlp_ratios[i],
                        mlp_dwconv=mlp_dwconv,
                        side_dwconv=side_dwconv,
                        before_attn_dwconv=before_attn_dwconv,
                        pre_norm=pre_norm,
                        auto_pad=auto_pad) for j in range(depth[i])],
            )
            if i in use_checkpoint_stages:
                stage = checkpoint_wrapper(stage)
            self.stages.append(stage)
            cur += depth[i]
        ##########################################################################
        self.norm = nn.BatchNorm2d(embed_dim[-1])
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
    def get_classifier(self):
        return self.head
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)  # res = (56, 28, 14, 7), wins = (64, 16, 4, 1)
            x = self.stages[i](x)
        x = self.norm(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.flatten(2).mean(-1)
        x = self.head(x)
        return x

#################### model variants #######################
model_urls = {
    "biformer_tiny_in1k": "https://api.onedrive.com/v1.0/shares/s!AkBbczdRlZvChHEOoGkgwgQzEDlM/root/content",
    "biformer_small_in1k": "https://api.onedrive.com/v1.0/shares/s!AkBbczdRlZvChHDyM-x9KWRBZ832/root/content",
    "biformer_base_in1k": "https://api.onedrive.com/v1.0/shares/s!AkBbczdRlZvChHI_XPhoadjaNxtO/root/content",
}

@register_model
def biformer_tiny(pretrained=False, pretrained_cfg=None,
                  pretrained_cfg_overlay=None, **kwargs):
    model = BiFormer(
        depth=[2, 2, 8, 2],
        embed_dim=[64, 128, 256, 512], mlp_ratios=[3, 3, 3, 3],
        # ------------------------------
        n_win=7,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[64, 128, 256, 512],
        head_dim=32,
        param_routing=False, diff_routing=False, soft_routing=False,
        pre_norm=True,
        pe=None,
        # -------------------------------
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        model_key = 'biformer_tiny_in1k'
        url = model_urls[model_key]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True,file_name=f"{model_key}.pth")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def biformer_small(pretrained=False, pretrained_cfg=None,
                   pretrained_cfg_overlay=None, **kwargs):
    model = BiFormer(
        depth=[4, 4, 18, 4],
        embed_dim=[64, 128, 256, 512], mlp_ratios=[3, 3, 3, 3],
        # ------------------------------
        n_win=7,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[64, 128, 256, 512],
        head_dim=32,
        param_routing=False, diff_routing=False, soft_routing=False,
        pre_norm=True,
        pe=None,
        # -------------------------------
        **kwargs)
    model.default_cfg = _cfg()

    if pretrained:
        model_key = 'biformer_small_in1k'
        url = model_urls[model_key]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True,file_name=f"{model_key}.pth")
        model.load_state_dict(checkpoint["model"])
    return model
@register_model
def biformer_base(pretrained=False, pretrained_cfg=None,
                  pretrained_cfg_overlay=None, **kwargs):
    model = BiFormer(
        depth=[4, 4, 18, 4],
        embed_dim=[96, 192, 384, 768], mlp_ratios=[3, 3, 3, 3],
        # use_checkpoint_stages=[0, 1, 2, 3],
        use_checkpoint_stages=[],
        # ------------------------------
        n_win=7,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[96, 192, 384, 768],
        head_dim=32,
        param_routing=False, diff_routing=False, soft_routing=False,
        pre_norm=True,
        pe=None,
        # -------------------------------
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        model_key = 'biformer_base_in1k'
        url = model_urls[model_key]
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True,file_name=f"{model_key}.pth")
        model.load_state_dict(checkpoint["model"])
    return model

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)
class ConvGRU(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 padding: int = 1):
        super().__init__()
        self.channels = channels
        self.ih = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding),
            nn.Sigmoid()
        )
        self.hh = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size, padding=padding),
            nn.Tanh()
        )
    def forward_single_frame(self, x, h):
        r, z = self.ih(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
        c = self.hh(torch.cat([x, r * h], dim=1))
        h = (1 - z) * h + z * c
        return h, h
    def forward_time_series(self, x, h):
        o = []
        for xt in x.unbind(dim=1):
            ot, h = self.forward_single_frame(xt, h)
            o.append(ot)
        o = torch.stack(o, dim=1)
        return o, h
    def forward(self, x, h):
        if h is None:
            h = torch.zeros((x.size(0), x.size(-3), x.size(-2), x.size(-1)),
                            device=x.device, dtype=x.dtype)
        if x.ndim == 5:
            return self.forward_time_series(x, h)
        else:
            return self.forward_single_frame(x, h)
class GRU(nn.Module):
    def __init__(self, channels):
        super(GRU, self).__init__()
        self.channels = channels
        self.gru = ConvGRU(self.channels)
    def forward(self, x):
        x1, x2 = x.split(self.channels, dim=1)
        x2, _ = self.gru(x2, None)
        return torch.cat([x1, x2], dim=1)

backbone_channels = {"resnet18": 64, "resnet34": 64, "resnet50": 256}
class TDM(nn.Module):
    def __init__(self, backbone="resnet18"):
        super(TDM, self).__init__()
        resnet1 = timm.create_model(backbone, pretrained=False)
        resnet2 = timm.create_model(backbone, pretrained=False)
        self.doubleConv_x = DoubleConv(in_channels=3, out_channels=64, mid_channels=32)
        self.down_x = Down(64, 64)
        self.resnet_layer1 = nn.Sequential(*list(resnet1.children())[4])
        self.layer1_bak = nn.Sequential(*list(resnet2.children())[4])
        self.alpha = 0.5
        self.beta = 0.5
        self.conv1_5 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.avg_diff = nn.AvgPool2d(kernel_size=2, stride=2)
        self.maxpool_diff = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.gru1 = GRU(32)
        self.gru2 = GRU(backbone_channels[backbone] // 2)
        self.gru3 = GRU(backbone_channels[backbone] // 2)
    def forward(self, x):
        x1, x2, x3, x4, x5 = x[:, 0:3, :, :], x[:, 3:6, :, :], x[:, 6:9, :, :], x[:, 9:12, :, :], x[:, 12:15, :, :]
        x_c5 = self.conv1_5(self.avg_diff(
            torch.cat([x2 - x1, x3 - x2, x4 - x3, x5 - x4], 1).view(-1, 12, x2.size()[2], x2.size()[3])
        ))
        x_diff = self.maxpool_diff(1.0 / 1.0 * x_c5)
        temp_out_diff1 = x_diff#中间的通道
        x_diff = self.resnet_layer1(x_diff)  # 最下面的通道
        temp_out_diff1 = self.gru1(temp_out_diff1)
        x_diff = self.gru2(x_diff)  # 最下面的通道
       #It 输入通道
        x = self.doubleConv_x(x3)
        x = self.down_x(x)
        x = self.maxpool(x)
        temp_out_diff1 = F.interpolate(temp_out_diff1, x.size()[2:]) #UpSample
        #第一次连接
        x = self.alpha * x + self.beta * temp_out_diff1
        x = self.layer1_bak(x)
        x_diff = F.interpolate(x_diff, x.size()[2:])#UpSample
        #第二次连接
        x = self.alpha * x + self.beta * x_diff
        x = self.gru3(x)
        return x
class AIF2M(nn.Module):
    def __init__(self, dims):
        super(AIF2M, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(dims, dims, kernel_size=3, padding=1, groups=dims, bias=False),
            nn.BatchNorm2d(dims),
            nn.ReLU(),
            nn.Conv2d(dims, dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(dims),
            nn.Sigmoid()
        )
        self.att1 = ECAAttention()
        self.att2 = ECAAttention()

    def forward(self, frame_x, temporality_x):
        frame_x_residual = frame_x
        temporality_x_residual = temporality_x
        frame_channel_att = self.att1(frame_x)
        temporality_channel_att = self.att2(temporality_x)
        fusion_feature = frame_channel_att + temporality_channel_att
        fusion_feature = self.fusion(fusion_feature)
        frame_x = frame_x * fusion_feature
        temporality_x = temporality_x * fusion_feature
        return frame_x + frame_x_residual, temporality_x + temporality_x_residual
class ECAAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.gap(x)
        y = y.squeeze(-1).permute(0, 2, 1)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1).unsqueeze(-1)
        return x * y.expand_as(x)
class GLFFM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(GLFFM, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.la_conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels // reduction),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.ga_gap = nn.AdaptiveAvgPool2d(1)
        self.ga_fc = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1),
            nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x1, x2):
        residual1 = x1
        residual2 = x2
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x = x1 + x2
        la_weight = self.la_conv(x)
        b, c, _, _ = x.size()
        ga_weight = self.ga_gap(x)
        ga_weight = ga_weight.squeeze(-1).permute(0, 2, 1)
        ga_weight = self.ga_fc(ga_weight)
        ga_weight = ga_weight.permute(0, 2, 1).unsqueeze(-1)
        att_weight = la_weight + ga_weight.expand_as(la_weight)
        att_weight = self.sigmoid(att_weight)
        residual1_weight = residual1 * att_weight
        residual2_weight = residual2 * att_weight
        residual1_weight = residual1 + residual1_weight
        residual2_weight = residual2 + residual2_weight
        return torch.cat((residual1_weight, residual2_weight), dim=1)

class Ours(nn.Module):
    def __init__(self,backbone='resnet18',attFusion=True,num_class=1,amplitude=20000,depth=[3, 4, 8, 3], in_chans=5, num_classes=1000, embed_dim=[64, 128, 320, 512],
                 head_dim=64, qk_scale=None, representation_size=None,
                 drop_path_rate=0., drop_rate=0.,
                 use_checkpoint_stages=[],
                 ########
                 n_win=7,
                 kv_downsample_mode='ada_avgpool',
                 kv_per_wins=[2, 2, -1, -1],
                 topks=[8, 8, -1, -1],
                 side_dwconv=5,
                 layer_scale_init_value=-1,
                 qk_dims=[None, None, None, None],
                 param_routing=False, diff_routing=False, soft_routing=False,
                 pre_norm=True,
                 pe=None,
                 pe_stages=[0],
                 before_attn_dwconv=3,
                 auto_pad=False,
                 #-----------------------
                 kv_downsample_kernels=[4, 2, 1, 1],
                 kv_downsample_ratios=[4, 2, 1, 1], # -> kv_per_win = [2, 2, 2, 1]
                 mlp_ratios=[4, 4, 4, 4],
                 param_attention='qkvo',
                 mlp_dwconv=False):
        super(Ours,self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        ############ downsample layers (patch embeddings) ######################
        self.downsample_layers = nn.ModuleList()
        # NOTE: uniformer uses two 3*3 conv, while in many other transformers this is one 7*7 conv
        stem = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim[0] // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0] // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim[0] // 2, embed_dim[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(embed_dim[0]),
        )
        if (pe is not None) and 0 in pe_stages:
            stem.append(get_pe_layer(emb_dim=embed_dim[0], name=pe))
        if use_checkpoint_stages:
            stem = checkpoint_wrapper(stem)
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv2d(embed_dim[i], embed_dim[i + 1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(embed_dim[i + 1])
            )
            if (pe is not None) and i + 1 in pe_stages:
                downsample_layer.append(get_pe_layer(emb_dim=embed_dim[i + 1], name=pe))
            if use_checkpoint_stages:
                downsample_layer = checkpoint_wrapper(downsample_layer)
            self.downsample_layers.append(downsample_layer)
        ##########################################################################

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        nheads = [dim // head_dim for dim in qk_dims]
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=embed_dim[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        topk=topks[i],
                        num_heads=nheads[i],
                        n_win=n_win,
                        qk_dim=qk_dims[i],
                        qk_scale=qk_scale,
                        kv_per_win=kv_per_wins[i],
                        kv_downsample_ratio=kv_downsample_ratios[i],
                        kv_downsample_kernel=kv_downsample_kernels[i],
                        kv_downsample_mode=kv_downsample_mode,
                        param_attention=param_attention,
                        param_routing=param_routing,
                        diff_routing=diff_routing,
                        soft_routing=soft_routing,
                        mlp_ratio=mlp_ratios[i],
                        mlp_dwconv=mlp_dwconv,
                        side_dwconv=side_dwconv,
                        before_attn_dwconv=before_attn_dwconv,
                        pre_norm=pre_norm,
                        auto_pad=auto_pad) for j in range(depth[i])],
            )
            if i in use_checkpoint_stages:
                stage = checkpoint_wrapper(stage)
            self.stages.append(stage)
            cur += depth[i]

        ##########################################################################
        self.norm = nn.BatchNorm2d(embed_dim[-1])
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

        self.amplitude = amplitude
        # self.predeal = CDC(in_channels=15,out_channels=15)
        self.reduction = 16
        backbone1 = timm.create_model(backbone)
        backbone2 = timm.create_model(backbone)
        backbone3 = timm.create_model(backbone)

        self.temporality_level_layer1 = TDM()
        self.temporality_level_layer2 = nn.Sequential(*list(backbone1.children())[5])
        self.temporality_level_layer3 = nn.Sequential(*list(backbone2.children())[6])
        self.temporality_level_layer4 = nn.Sequential(*list(backbone3.children())[7])
        # self.temporality_level_layer4 = block(group=groups)
        self.feature_interaction1 = AIF2M(64)
        self.feature_interaction2 = AIF2M(128)
        self.feature_interaction3 = AIF2M(256)
        self.feature_interaction4 = AIF2M(512)

        if attFusion:
            self.final_fusion = GLFFM(256, self.reduction)
        else:
            self.final_fusion = None

        self.classifier = nn.Linear(1024, num_class)

        self.apply(self.init_weight)
        self.flatten = Flatten()
        self.fn1 = nn.Linear(in_features=512, out_features=128)
        self.fn2 = nn.Linear(in_features=128, out_features=32)
        self.fn3 = nn.Linear(in_features=32, out_features=8)
        self.fn4 = nn.Linear(in_features=8, out_features=2)
        self.drop = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)

    def get_phase(self, data):
        data = torch.fft.fft2(data, dim=(-2, -1))
        new_data = self.amplitude * torch.exp(data.angle() * 1j)
        data_ift = torch.fft.ifft2(new_data, dim=(-2, -1))
        return data_ift.real / 255.

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    def forward(self,x,y):
        # x=self.predeal(x)
        frame_x1 =self.downsample_layers[0](x)#(3,64,56,56)
        frame_x1 = self.stages[0](frame_x1)#(3,64,56,56)
        temproality_x1 = self.temporality_level_layer1(y)#(3,64,56,56)
        frame_x1,temproality_x1 = self.feature_interaction1(frame_x1,temproality_x1)#(3,64,56,56)

        frame_x2 = self.downsample_layers[1](frame_x1)#(3,128,28,28)
        frame_x2 = self.stages[1](frame_x2)#(3,128,28,28)
        temproality_x2 = self.temporality_level_layer2(temproality_x1)#(3,128,28,28)
        frame_x2, temproality_x2 = self.feature_interaction2(frame_x2, temproality_x2) #(3,128,28,28)

        frame_x3 =self.downsample_layers[2](frame_x2) #(3,256,14,14)
        frame_x3 = self.stages[2](frame_x3)#(3,256,14,14)
        temproality_x3 = self.temporality_level_layer3(temproality_x2)  #(3,256,14,14)
        frame_x3, temproality_x3 = self.feature_interaction3(frame_x3, temproality_x3)
        #
        # frame_x4 = self.downsample_layers[3](frame_x3)#(3,512,7,7)
        # frame_x4 = self.stages[3](frame_x4)
        # temproality_x4 = self.temporality_level_layer4(temproality_x3)#(3,512,7,7)
        # frame_x4, temproality_x4 = self.feature_interaction4(frame_x4, temproality_x4)

        if self.final_fusion is None:
            temporality_x = F.adaptive_avg_pool2d(temproality_x3, 1)
            temporality_x = temporality_x.view(temporality_x.size(0), -1)
            frame_x = F.adaptive_avg_pool2d(frame_x3, 1)
            frame_x = frame_x.view(frame_x.size(0), -1)
            y = torch.cat((temporality_x, frame_x), dim=1)
        else:
            y = self.final_fusion(temproality_x3, frame_x3)
            y = F.adaptive_avg_pool2d(y, 1)
            y = y.view(y.size(0), -1)

        out = self.fn1(y)
        out = self.drop(out)
        out = self.fn2(out)
        out = self.drop(out)
        out = self.fn3(out)
        out = self.drop(out)
        out = self.fn4(out)
        out = self.softmax(out)
        return out

    @staticmethod
    def init_weight(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, (nn.Linear, nn.Conv1d)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

if __name__ == '__main__':
    #ours =Ours(1)
    model = Ours(backbone="resnet18", attFusion=True,depth=[4, 4, 18, 4],
                 embed_dim=[64, 128, 256, 512], mlp_ratios=[3, 3, 3, 3],
                   # ------------------------------
                 n_win=7,
                 kv_downsample_mode='identity',
                 kv_per_wins=[-1, -1, -1, -1],
                 topks=[1, 4, 16, -2],
                 side_dwconv=5,
                 before_attn_dwconv=3,
                 layer_scale_init_value=-1,
                 qk_dims=[64, 128, 256, 512],
                 head_dim=32,
                 param_routing=False, diff_routing=False, soft_routing=False,
                 pre_norm=True,
                 pe=None)
    x = torch.randn(3, 5, 224, 224)
    y = torch.randn(3, 15, 224, 224)
    #out = model(x)
    print(model(x,y).size)