
import copy
from typing import Callable, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch_utils import persistence
from torch.nn.functional import silu, softplus
from einops.layers.torch import Rearrange

from training.TemporalUnet import TemporalUnet
from training.TransformerForDiffusion import TransformerForDiffusion
import torch_utils.pytorch_utils as ptu



def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')



@persistence.persistent_class
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


@persistence.persistent_class
class Conv2d(torch.nn.Module):
    def __init__(self,
                 in_channels, out_channels, kernel, bias=True, up=False, down=False,
                 resample_filter=[1, 1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
                 ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels * kernel * kernel, fan_out=out_channels * kernel * kernel)
        self.weight = torch.nn.Parameter(
            weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(
            weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]),
                                                     groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad + f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]),
                                                         groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels,
                                               stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


@persistence.persistent_class
class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype),
                                           bias=self.bias.to(x.dtype), eps=self.eps)
        return x



class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(
            dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2,
                                          input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk



@persistence.persistent_class
class UNetBlock(torch.nn.Module):
    def __init__(self,
                 in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
                 num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
                 resample_filter=[1, 1], resample_proj=False, adaptive_scale=True,
                 init=dict(), init_zero=dict(init_weight=0), init_attn=None,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down,
                            resample_filter=resample_filter, **init)
        self.affine = Linear(in_features=emb_channels, out_features=out_channels * (2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down,
                               resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels * 3, kernel=1,
                              **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3,
                                                      -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x


@persistence.persistent_class
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


@persistence.persistent_class
class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x



@persistence.persistent_class
class SongUNet(torch.nn.Module):
    def __init__(self,
                 img_resolution,  # Image resolution at input/output.
                 in_channels,  # Number of color channels at input.
                 out_channels,  # Number of color channels at output.
                 label_dim=0,  # Number of class labels, 0 = unconditional.
                 augment_dim=0,  # Augmentation label dimensionality, 0 = no augmentation.

                 model_channels=128,  # Base multiplier for the number of channels.
                 channel_mult=[1, 2, 2, 2],  # Per-resolution multipliers for the number of channels.
                 channel_mult_emb=4,  # Multiplier for the dimensionality of the embedding vector.
                 num_blocks=4,  # Number of residual blocks per resolution.
                 attn_resolutions=[16],  # List of resolutions with self-attention.
                 dropout=0.10,  # Dropout probability of intermediate activations.
                 label_dropout=0,  # Dropout probability of class labels for classifier-free guidance.

                 embedding_type='positional',  # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
                 channel_mult_noise=1,  # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
                 encoder_type='standard',  # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
                 decoder_type='standard',  # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
                 resample_filter=[1, 1],  # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
                 ):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']

        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=noise_channels,
                                             endpoint=True) if embedding_type == 'positional' else FourierEmbedding(
            num_channels=noise_channels)
        self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False,
                                  **init) if augment_dim else None
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(in_channels=cout, out_channels=cout, down=True,
                                                          **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(in_channels=caux, out_channels=caux, kernel=0, down=True,
                                                               resample_filter=resample_filter)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(in_channels=caux, out_channels=cout, kernel=3,
                                                                   down=True, resample_filter=resample_filter,
                                                                   fused_resample=True, **init)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn,
                                                                **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(in_channels=cout, out_channels=cout, attention=True,
                                                         **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn,
                                                                **block_kwargs)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res}x{res}_aux_up'] = Conv2d(in_channels=out_channels, out_channels=out_channels,
                                                             kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3,
                                                           **init_zero)

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return aux


@persistence.persistent_class
class TimeEmbedLinear(torch.nn.Module):
    def __init__(self, in_dim, out_dim, time_embed_dim):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim + time_embed_dim, out_dim)

    def forward(self, x, emb):
        return self.linear(torch.cat([x, emb], dim=1))


@persistence.persistent_class
class LinearBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, time_embed_dim, num_heads, max_length_of_trajectory=10000, eps=1e-5, dropout=0,
                 resample_filter=[1, 1], resample_proj=False, adapative_scale=True, init=dict(),
                 init_zero=dict(init_weight=0),
                 init_attn=None
                 ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.time_emb_dim = time_embed_dim
        self.dropout = dropout
        self.adaptive_scale = adapative_scale

        self.linear = TimeEmbedLinear(in_dim, out_dim, time_embed_dim)
        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=max_length_of_trajectory, eps=eps, num_groups=8)
            self.qkv = Linear(out_dim, out_dim * 3, **(init_attn if init_attn is not None else init))
            self.proj = Linear(out_dim, out_dim, **init_zero)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        local_emb = emb.unsqueeze(1)
        local_emb = local_emb.repeat(1, x.shape[1], 1)  # B, T, emb_dim
        B, T, C = x.shape
        x = x.reshape(B * T, C)
        local_emb = local_emb.reshape(B * T, -1)
        x = self.linear(x, local_emb)
        x = torch.nn.functional.relu(x)
        # x = x.reshape(B, T, -1) # B, T, C
        # x = x.reshape(B * T, -1)
        x = x.reshape(B, T, -1)
        x = self.norm2(x)
        if self.num_heads:
            qkv = self.qkv(x).reshape(B * self.num_heads, T // self.num_heads, 3, -1)
            qkv = torch.einsum('BTtC->tBCT', qkv)  # 3 x B x C x T
            q, k, v = qkv
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->nqc', w, v)
            x = self.proj(a).add_(x)
        return x


@persistence.persistent_class
class MLP_RL(torch.nn.Module):
    def __init__(self,
                 data_dim,  # the dim of the input/output vector
                 max_length_of_trajectory,  # the length of the input/output vector
                 hidden_dims=(256, 256),  #. the dims of the latent layers
                 dropout=0.10,  # Dropout probability of intermediate activations.
                 label_dim=0,  # Number of class labels, 0 = unconditional. For RL now must be 0.
                 augment_dim=0,  # Augmentation label dimensionality, 0 = no augmentation.

                 num_heads=1,  # the number of head in attention
                 embedding_type='positional',  # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
                 channel_mult_noise=1,  # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
                 channel_mult_emb=4,  # Multiplier for the dimensionality of the embedding vector.
                 resample_filter=[1, 1],
                 # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++. not clear whether it should be use in rl
                 **kwargs,
                 ):
        if label_dim:
            raise NotImplementedError
        super().__init__()
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        noise_channels = hidden_dims[0] * channel_mult_noise
        emb_channels = hidden_dims[0] * channel_mult_emb
        self.data_dim = data_dim
        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=noise_channels,
                                             endpoint=True) if embedding_type == 'positional' else FourierEmbedding(
            num_channels=noise_channels)
        self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False,
                                  **init) if augment_dim else None
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        self.max_length_of_trajectory = max_length_of_trajectory
        self.layers = torch.nn.ModuleList(
            [
                LinearBlock(data_dim, hidden_dims[0], emb_channels, num_heads,
                            max_length_of_trajectory=max_length_of_trajectory, dropout=dropout, eps=1e-6, init=init,
                            init_zero=init_zero, init_attn=init_attn)
            ]
        )
        for i in range(len(hidden_dims) - 1):
            self.layers.append(LinearBlock(hidden_dims[i], hidden_dims[i + 1], emb_channels, num_heads,
                                           max_length_of_trajectory=max_length_of_trajectory, dropout=dropout, eps=1e-6,
                                           init=init, init_zero=init_zero, init_attn=init_attn))

        self.out = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            Linear(in_features=hidden_dims[-1], out_features=data_dim, **init),
        )

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        aux = x
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Rearrange):
                aux = layer(aux)
            else:
                aux = layer(aux, emb)
        out = self.out(aux)
        return out


@persistence.persistent_class
class UNet_RL(TemporalUnet):
    def __init__(self,
                 data_dim,  # the dim of the input/output vector
                 max_length_of_trajectory=100,  # the length of the input/output vector
                 dim=128,
                 dim_mults=(1, 4, 8),
                 label_dim=0,
                 kernel_size=5,
                 **kwargs,
                 ):
        self.max_length_of_trajectory = max_length_of_trajectory
        self.data_dim = data_dim
        super().__init__(transition_dim=data_dim,
                         horizon=max_length_of_trajectory,
                         dim=dim, dim_mults=dim_mults, cond_dim=label_dim,
                         kernel_size=kernel_size
                         )

    def forward(self, x, noise_labels, returns=None, use_dropout=True, force_dropout=False, class_labels=None,
                augment_labels=None):
        x = super().forward(x, None, noise_labels, returns, use_dropout, force_dropout)
        return x

