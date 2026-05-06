"""
UnDIP CNN Architecture: UnmixArch

Encoder-decoder with skip connections for hyperspectral abundance estimation.
Faithfully reproduced from the official UnDIP repository:
    https://github.com/BehnoodRasti/UnDIP

Paper:
    B. Rasti, B. Koirala, P. Scheunders, P. Ghamisi.
    "UnDIP: Hyperspectral Unmixing Using Deep Image Prior."
    IEEE Transactions on Geoscience and Remote Sensing, 2022.

Architecture key points:
    - Input  : random noise tensor of shape (1, p, H, W)
    - Output : abundance maps of shape (1, p, H, W), passed through Softmax
               to enforce the Abundance Sum-to-one Constraint (ASC).
    - Depth  : configurable via num_channels_down/up/skip lists.
    - Default used in paper: 1-scale, channels_down=256, channels_up=256,
      channels_skip=4, filter 3x3, LeakyReLU, bilinear upsampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility layers (from DIP / UnDIP common.py)
# ---------------------------------------------------------------------------

class Concat(nn.Module):
    """Concatenate outputs of multiple sub-modules along given dimension."""
    def __init__(self, dim, *modules):
        super().__init__()
        self.dim = dim
        for idx, module in enumerate(modules):
            self.add_module(str(idx), module)

    def forward(self, x):
        # Align spatial dimensions before concatenating (handles odd-size mismatch)
        outputs = [m(x) for m in self._modules.values()]
        if len(outputs) > 1:
            # Crop or pad all outputs to match the first output's spatial size
            target_h = outputs[0].shape[2]
            target_w = outputs[0].shape[3]
            aligned = [outputs[0]]
            for o in outputs[1:]:
                if o.shape[2] != target_h or o.shape[3] != target_w:
                    o = o[:, :, :target_h, :target_w]
                aligned.append(o)
            outputs = aligned
        return torch.cat(outputs, dim=self.dim)


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='reflection',
         downsample_mode='stride'):
    """Conv layer with optional reflection padding."""
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':
        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        else:
            raise ValueError(f'Unsupported downsample_mode: {downsample_mode}')
        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride,
                          padding=to_pad, bias=bias)
    layers = [m for m in [padder, convolver, downsampler] if m is not None]
    return nn.Sequential(*layers)


def bn(num_features):
    return nn.BatchNorm2d(num_features)


def act(act_fun='LeakyReLU'):
    if act_fun == 'LeakyReLU':
        return nn.LeakyReLU(0.2, inplace=True)
    elif act_fun == 'Swish':
        return nn.SiLU(inplace=True)
    elif act_fun == 'ELU':
        return nn.ELU(inplace=True)
    elif act_fun == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_fun == 'none':
        return nn.Identity()
    else:
        raise ValueError(f'Unknown activation: {act_fun}')


# ---------------------------------------------------------------------------
# UnmixArch — the actual model used in UnDIP
# ---------------------------------------------------------------------------

def build_unmix_arch(
        num_input_channels: int,
        num_output_channels: int,
        num_channels_down=(256,),
        num_channels_up=(256,),
        num_channels_skip=(4,),
        filter_size_down: int = 3,
        filter_size_up: int = 3,
        filter_skip_size: int = 1,
        need_sigmoid: bool = True,
        need_bias: bool = True,
        pad: str = 'reflection',
        upsample_mode: str = 'bilinear',
        downsample_mode: str = 'stride',
        act_fun: str = 'LeakyReLU',
        need1x1_up: bool = True,
) -> nn.Sequential:
    """
    Build the UnmixArch encoder-decoder with skip connections.

    Architecture (for the 1-scale default from the UnDIP paper):
    ┌────────────────────────────────────────────────────────┐
    │  Input: z  (1, p, H, W)  — random noise               │
    │                                                        │
    │  ┌─── Skip branch ───┐                                │
    │  │ Conv1x1→4ch→BN→LR │                                │
    │  └───────────────────┘                                │
    │            │                                           │
    │  ┌─── Encoder ──────┐                                 │
    │  │ Conv3x3(s=2)→256 │                                 │
    │  │ BN → LeakyReLU   │                                 │
    │  │ Conv3x3→256       │                                 │
    │  │ BN → LeakyReLU   │                                 │
    │  │ Upsample ×2      │                                 │
    │  └──────────────────┘                                 │
    │            │                                           │
    │  Concat(skip_ch + up_ch, axis=1)                      │
    │  BN                                                    │
    │  Conv3x3→256, BN, LeakyReLU                           │
    │  Conv1x1→256, BN, LeakyReLU   ← (need1x1_up)         │
    │  Conv1x1→p                                            │
    │  Softmax (channel-wise, enforces ASC)                 │
    │                                                        │
    │  Output: A_hat  (1, p, H, W)                          │
    └────────────────────────────────────────────────────────┘

    Args:
        num_input_channels  : p (number of endmembers) — depth of noise input.
        num_output_channels : p — depth of abundance map output.
        num_channels_down   : list of ints, one per encoder scale.
        num_channels_up     : list of ints, one per decoder scale.
        num_channels_skip   : list of ints (skip connection widths).
        filter_size_down    : encoder conv kernel size.
        filter_size_up      : decoder conv kernel size.
        filter_skip_size    : skip branch conv kernel size.
        need_sigmoid        : if True, apply Softmax (ASC constraint).
        need_bias           : use bias in convolutions.
        pad                 : 'reflection' or 'zero'.
        upsample_mode       : 'bilinear' or 'nearest'.
        downsample_mode     : 'stride' (default) or 'avg'/'max'.
        act_fun             : activation function name.
        need1x1_up          : add 1×1 conv+BN+act after each decoder stage.

    Returns:
        nn.Sequential model.
    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip), \
        "num_channels_down, num_channels_up, num_channels_skip must have same length"

    n_scales = len(num_channels_down)

    upsample_modes   = [upsample_mode]   * n_scales
    downsample_modes = [downsample_mode] * n_scales
    filter_sizes_dn  = [filter_size_down] * n_scales
    filter_sizes_up  = [filter_size_up]   * n_scales

    last_scale = n_scales - 1

    model     = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels

    for i in range(n_scales):
        deeper    = nn.Sequential()
        skip_path = nn.Sequential()

        # Connect skip + deeper via Concat or just deeper
        if num_channels_skip[i] != 0:
            model_tmp.add_module(f'concat_{i}', Concat(1, skip_path, deeper))
        else:
            model_tmp.add_module(f'deeper_{i}', deeper)

        # BN after concat
        bn_ch = num_channels_skip[i] + (
            num_channels_up[i + 1] if i < last_scale else num_channels_down[i]
        )
        model_tmp.add_module(f'bn_post_concat_{i}', bn(bn_ch))

        # Skip branch: 1-scale conv from input
        if num_channels_skip[i] != 0:
            skip_path.add_module('skip_conv',
                conv(input_depth, num_channels_skip[i],
                     filter_skip_size, bias=need_bias, pad=pad))
            skip_path.add_module('skip_bn', bn(num_channels_skip[i]))
            skip_path.add_module('skip_act', act(act_fun))

        # Encoder: stride-2 downsampling conv
        deeper.add_module('enc_conv1',
            conv(input_depth, num_channels_down[i],
                 filter_sizes_dn[i], stride=2, bias=need_bias,
                 pad=pad, downsample_mode=downsample_modes[i]))
        deeper.add_module('enc_bn1', bn(num_channels_down[i]))
        deeper.add_module('enc_act1', act(act_fun))

        deeper.add_module('enc_conv2',
            conv(num_channels_down[i], num_channels_down[i],
                 filter_sizes_dn[i], bias=need_bias, pad=pad))
        deeper.add_module('enc_bn2', bn(num_channels_down[i]))
        deeper.add_module('enc_act2', act(act_fun))

        deeper_main = nn.Sequential()
        if i == last_scale:
            k = num_channels_down[i]
        else:
            deeper.add_module(f'deeper_main_{i}', deeper_main)
            k = num_channels_up[i + 1]

        # Upsampling
        deeper.add_module('upsample',
            nn.Upsample(scale_factor=2, mode=upsample_modes[i],
                        align_corners=(True if upsample_modes[i] == 'bilinear' else None)))

        # Decoder conv
        model_tmp.add_module(f'dec_conv_{i}',
            conv(num_channels_skip[i] + k, num_channels_up[i],
                 filter_sizes_up[i], bias=need_bias, pad=pad))
        model_tmp.add_module(f'dec_bn_{i}', bn(num_channels_up[i]))
        model_tmp.add_module(f'dec_act_{i}', act(act_fun))

        if need1x1_up:
            model_tmp.add_module(f'dec_1x1_{i}',
                conv(num_channels_up[i], num_channels_up[i], 1,
                     bias=need_bias, pad=pad))
            model_tmp.add_module(f'dec_1x1_bn_{i}', bn(num_channels_up[i]))
            model_tmp.add_module(f'dec_1x1_act_{i}', act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp   = deeper_main

    # Final 1×1 conv: num_channels_up[0] → p
    model.add_module('final_conv',
        conv(num_channels_up[0], num_output_channels, 1,
             bias=need_bias, pad=pad))

    # Softmax enforces ASC (Abundance Sum-to-one Constraint)
    if need_sigmoid:
        model.add_module('softmax', nn.Softmax(dim=1))

    return model


# ---------------------------------------------------------------------------
# Wrapper module (for clean interface)
# ---------------------------------------------------------------------------

class UnDIPNet(nn.Module):
    """
    Wrapper around build_unmix_arch for the default UnDIP configuration.

    Default parameters match the Samson/Jasper runs in the paper:
        - 1 encoder-decoder scale
        - 256 channels down/up, 4 skip channels
        - 3×3 filters, reflection padding, bilinear upsampling
        - LeakyReLU activation, Softmax output
    """
    def __init__(self, p: int,
                 num_channels_down=(256,),
                 num_channels_up=(256,),
                 num_channels_skip=(4,),
                 filter_size_down: int = 3,
                 filter_size_up: int = 3,
                 filter_skip_size: int = 1,
                 pad: str = 'reflection',
                 upsample_mode: str = 'bilinear',
                 act_fun: str = 'LeakyReLU'):
        super().__init__()
        self.net = build_unmix_arch(
            num_input_channels=p,
            num_output_channels=p,
            num_channels_down=num_channels_down,
            num_channels_up=num_channels_up,
            num_channels_skip=num_channels_skip,
            filter_size_down=filter_size_down,
            filter_size_up=filter_size_up,
            filter_skip_size=filter_skip_size,
            need_sigmoid=True,
            need_bias=True,
            pad=pad,
            upsample_mode=upsample_mode,
            downsample_mode='stride',
            act_fun=act_fun,
            need1x1_up=True,
        )

    def forward(self, z):
        """
        Args:
            z: (1, p, H, W) random noise tensor.
        Returns:
            A_hat: (1, p, H, W) abundance maps, each pixel sums to 1 (Softmax).
        """
        return self.net(z)
