import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from modeling_finetune_av_cy import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table, CSBlock, Mlp, PatchEmbed2D
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from einops import rearrange, repeat


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=1,
                 use_learnable_pos_emb=False, max_length=12,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=max_length,
            tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches

        # TODO: Add the cls token
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'part_tokens'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask, return_intermediate_features=None, padded_video=None):
        _, _, T, _, _ = x.shape
        x = self.patch_embed(x)
        B, N, C = x.shape
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        x_vis = x[~mask].reshape(B, -1, C)  # ~mask means visible
        tmp_x = x.reshape(padded_video.shape[0], padded_video.shape[1], -1)
        padded_video = padded_video[:, :, None].expand(-1, -1, tmp_x.shape[2])
        padded_video = padded_video.reshape(B, N, C)
        vis_padded_video = padded_video[~mask].reshape(B, -1, C)
        invis_padded_video = padded_video[mask].reshape(B, -1, C)
        intermediate_features = []
        for blk in self.blocks:
            x_vis = blk(x_vis, vis_padded_video[:, :, 0])
            intermediate_features.append(self.norm(x_vis))

        x_vis = self.norm(x_vis)

        if return_intermediate_features is None:
            return x_vis, None, vis_padded_video, invis_padded_video
        else:
            intermediate_features = [intermediate_features[i] for i in return_intermediate_features]
            return x_vis, intermediate_features, vis_padded_video, invis_padded_video

    def forward(self, x, mask, return_intermediate_features=None, padded_video=0):
        x, intermediate_features, vis_padded_video, invis_padded_video = self.forward_features(x, mask,
                                                                                               return_intermediate_features,
                                                                                               padded_video)
        x = self.head(x)
        return x, intermediate_features, vis_padded_video, invis_padded_video


class PretrainVisionTransformerEncoderMesh(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, vertex_size, patch_size=762, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=1,
                 use_learnable_pos_emb=False, max_length=12,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = nn.Linear(patch_size, embed_dim)
        num_patches = max_length

        # TODO: Add the cls token
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'part_tokens'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask, return_intermediate_features=None, padded_vertex=None):
        _, T, _ = x.shape
        x = self.patch_embed(x)
        B, N, C = x.shape
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        x_vis = x[~mask].reshape(B, -1, C)  # ~mask means visible
        tmp_x = x.reshape(padded_vertex.shape[0], padded_vertex.shape[1], -1)
        padded_vertex = padded_vertex[:, :, None].expand(-1, -1, tmp_x.shape[2])
        padded_vertex = padded_vertex.reshape(B, N, C)
        vis_padded_vertex = padded_vertex[~mask].reshape(B, -1, C)
        invis_padded_vertex = padded_vertex[mask].reshape(B, -1, C)
        intermediate_features = []
        for blk in self.blocks:
            x_vis = blk(x_vis, vis_padded_vertex[:, :, 0])
            intermediate_features.append(self.norm(x_vis))

        x_vis = self.norm(x_vis)

        if return_intermediate_features is None:
            return x_vis, None, vis_padded_vertex, invis_padded_vertex
        else:
            intermediate_features = [intermediate_features[i] for i in return_intermediate_features]
            return x_vis, intermediate_features, vis_padded_vertex, invis_padded_vertex

    def forward(self, x, mask, return_intermediate_features=None, padded_vertex=0):
        x, intermediate_features, vis_padded_vertex, invis_padded_vertex = self.forward_features(x, mask,
                                                                                                 return_intermediate_features,
                                                                                                 padded_vertex)
        x = self.head(x)
        return x, intermediate_features, vis_padded_vertex, invis_padded_vertex


class ProbPretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=1,
                 use_learnable_pos_emb=False, max_length=12,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=max_length,
            tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches

        # TODO: Add the cls token
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.depth = depth
        self.n_uncertainty_layers = 1

        self.uncertainty_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[-1], norm_layer=norm_layer,
            init_values=init_values)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'part_tokens'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask, return_intermediate_features=None, padded_video=None):
        _, _, T, _, _ = x.shape
        x = self.patch_embed(x)
        B, N, C = x.shape
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        x_vis = x[~mask].reshape(B, -1, C)  # ~mask means visible
        tmp_x = x.reshape(padded_video.shape[0], padded_video.shape[1], -1)
        padded_video = padded_video[:, :, None].expand(-1, -1, tmp_x.shape[2])
        padded_video = padded_video.reshape(B, N, C)
        vis_padded_video = padded_video[~mask].reshape(B, -1, C)
        invis_padded_video = padded_video[mask].reshape(B, -1, C)
        intermediate_features = []

        for idx in range(0, self.depth - self.n_uncertainty_layers):
            x_vis = self.blocks[idx](x_vis, vis_padded_video[:, :, 0])
            intermediate_features.append(self.norm(x_vis))
        mean_x = x_vis
        std_x = self.uncertainty_block(x_vis, vis_padded_video[:, :, 0])
        for idx in range(self.depth - self.n_uncertainty_layers, self.depth):
            mean_x = self.blocks[idx](mean_x, vis_padded_video[:, :, 0])
            intermediate_features.append(self.norm(mean_x))

        mean_x = self.norm(mean_x)

        if return_intermediate_features is None:
            return mean_x, std_x, None, vis_padded_video, invis_padded_video
        else:
            intermediate_features = [intermediate_features[i] for i in return_intermediate_features]
            return mean_x, std_x, intermediate_features, vis_padded_video, invis_padded_video

    def forward(self, x, mask, return_intermediate_features=None, padded_video=0):
        mean_x, std_x, intermediate_features, vis_padded_video, invis_padded_video = self.forward_features(x, mask,
                                                                                                           return_intermediate_features,
                                                                                                           padded_video)
        x = self.head(x)
        return mean_x, std_x, intermediate_features, vis_padded_video, invis_padded_video


class ConformerEncoder(torch.nn.Module):
    """Transformer encoder module.

    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate in attention
    :param float positional_dropout_rate: dropout rate after adding positional encoding
    :param str or torch.nn.Module input_layer: input layer type
    :param class pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    :param int positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
    :param str encoder_attn_layer_type: encoder attention layer type
    :param bool macaron_style: whether to use macaron style for positionwise layer
    :param bool use_cnn_module: whether to use convolution module
    :param bool zero_triu: whether to zero the upper triangular part of attention matrix
    :param int cnn_module_kernel: kernerl size of convolution module
    :param int padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
            self,
            attention_dim=768,
            attention_heads=12,
            linear_units=3072,
            num_blocks=12,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            normalize_before=True,
            concat_after=False,
            macaron_style=True,
            use_cnn_module=True,
            zero_triu=False,
            cnn_module_kernel=31,
            relu_type="swish",
            a_upsample_ratio=1,
    ):
        """Construct an Encoder object."""
        super(ConformerEncoder, self).__init__()
        self._register_load_state_dict_pre_hook(_pre_hook)

        pos_enc_class = RelPositionalEncoding

        self.frontend = Conv1dResNet(relu_type=relu_type, a_upsample_ratio=a_upsample_ratio)

        self.embed = torch.nn.Sequential(torch.nn.Linear(512, attention_dim),
                                         pos_enc_class(attention_dim, positional_dropout_rate))

        self.normalize_before = normalize_before
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (attention_dim, linear_units, dropout_rate)

        encoder_attn_layer = RelPositionMultiHeadedAttention
        encoder_attn_layer_args = (
            attention_heads,
            attention_dim,
            attention_dropout_rate,
            zero_triu,
        )

        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel)

        self.encoders = nn.ModuleList([
            ConformerEncoderLayer(
                attention_dim,
                encoder_attn_layer(*encoder_attn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                macaron_style,
            )
            for i in range(num_blocks)])

        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

    def forward(self, xs, masks, return_intermediate_features):
        """Encode input sequence.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        xs = self.frontend(xs)

        # xs : torch.Size([1, 5, 512])
        xs = self.embed(xs)
        # xs : torch.Size([1, 5, 768]), torch.Size([1, 9, 768])
        intermediate_features = []
        for encoder in self.encoders:
            xs, masks = encoder(xs, masks)
            intermediate_features.append(xs[0])
        xs = xs[0]
        if self.normalize_before:
            xs = self.after_norm(xs)

        if return_intermediate_features is None:
            return xs
        else:
            intermediate_features = [intermediate_features[i] for i in return_intermediate_features]
            return xs, intermediate_features

    def forward_one_step(self, xs, masks, cache=None):
        """Encode input frame.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :param List[torch.Tensor] cache: cache tensors
        :return: position embedded tensor, mask and new cache
        :rtype Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        if isinstance(self.frontend, (Conv1dResNet, Conv3dResNet)):
            xs = self.frontend(xs)

        xs = self.embed(xs)

        if cache is None:
            cache = [None for _ in range(len(self.encoders))]
        new_cache = []
        for c, e in zip(cache, self.encoders):
            xs, masks = e(xs, masks, cache=c)
            new_cache.append(xs)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks, new_cache


class PretrainVisionTransformerEncoder2D(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, use_checkpoint=False,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed2D(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, stride=patch_size)
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint
        # TODO: Add the cls token
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask, return_intermediate_features=None, padded_audio=None):
        x = self.patch_embed(x)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        B, N, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C)  # ~mask means visible
        tmp_x = x.reshape(padded_audio.shape[0], padded_audio.shape[1], -1)
        padded_audio = padded_audio[:, :, None].expand(-1, -1, tmp_x.shape[2])
        padded_audio = padded_audio.reshape(B, N, C)
        vis_padded_audio = padded_audio[~mask].reshape(B, -1, C)
        invis_padded_audio = padded_audio[mask].reshape(B, -1, C)
        intermediate_features = []

        for blk in self.blocks:
            x_vis = blk(x_vis, vis_padded_audio[:, :, 0])
            intermediate_features.append(self.norm(x_vis))

        x_vis = self.norm(x_vis)

        if return_intermediate_features is None:
            return x_vis, None, vis_padded_audio, invis_padded_audio
        else:
            intermediate_features = [intermediate_features[i] for i in return_intermediate_features]
            return x_vis, intermediate_features, vis_padded_audio, invis_padded_audio

    def forward(self, x, mask, return_intermediate_features=None, padded_audio=None):
        x, intermediate_features, vis_padded_audio, invis_padded_audio = self.forward_features(x, mask,
                                                                                               return_intermediate_features,
                                                                                               padded_audio)
        x = self.head(x)
        return x, intermediate_features, vis_padded_audio, invis_padded_audio


class ProbPretrainVisionTransformerEncoder2D(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, use_checkpoint=False,
                 use_learnable_pos_emb=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed2D(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, stride=patch_size)
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint
        # TODO: Add the cls token
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.depth = depth
        self.n_uncertainty_layers = 1

        self.uncertainty_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[-1], norm_layer=norm_layer,
            init_values=init_values)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask, return_intermediate_features=None, padded_audio=None):
        x = self.patch_embed(x)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        B, N, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C)  # ~mask means visible
        tmp_x = x.reshape(padded_audio.shape[0], padded_audio.shape[1], -1)
        padded_audio = padded_audio[:, :, None].expand(-1, -1, tmp_x.shape[2])
        padded_audio = padded_audio.reshape(B, N, C)
        vis_padded_audio = padded_audio[~mask].reshape(B, -1, C)
        invis_padded_audio = padded_audio[mask].reshape(B, -1, C)
        intermediate_features = []

        for idx in range(0, self.depth - self.n_uncertainty_layers):
            x_vis = self.blocks[idx](x_vis, vis_padded_audio[:, :, 0])
            intermediate_features.append(self.norm(x_vis))
        mean_x = x_vis

        std_x = self.uncertainty_block(x_vis, vis_padded_audio[:, :, 0])
        for idx in range(self.depth - self.n_uncertainty_layers, self.depth):
            mean_x = self.blocks[idx](mean_x, vis_padded_audio[:, :, 0])
            intermediate_features.append(self.norm(mean_x))

        mean_x = self.norm(mean_x)

        if return_intermediate_features is None:
            return mean_x, std_x, None, vis_padded_audio, invis_padded_audio
        else:
            intermediate_features = [intermediate_features[i] for i in return_intermediate_features]
            return mean_x, std_x, intermediate_features, vis_padded_audio, invis_padded_audio

    def forward(self, x, mask, return_intermediate_features=None, padded_audio=None):
        mean_x, std_x, intermediate_features, vis_padded_audio, invis_padded_audio = self.forward_features(x, mask,
                                                                                                           return_intermediate_features,
                                                                                                           padded_audio)
        x = self.head(x)
        return mean_x, std_x, intermediate_features, vis_padded_audio, invis_padded_audio


class PretrainVisionTransformerEncoderForFusion(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, embed_dim=768, embed_dim_audio=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 modal_param_sharing=False):
        super().__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            CSBlock(dim=embed_dim, context_dim=embed_dim_audio, num_heads=num_heads,
                    num_cross_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # audio
        self.modal_param_sharing = modal_param_sharing
        if not modal_param_sharing:
            self.blocks_audio = nn.ModuleList([
                CSBlock(dim=embed_dim_audio, context_dim=embed_dim, num_heads=num_heads,
                        num_cross_heads=num_heads,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[i], norm_layer=norm_layer,
                        init_values=init_values)
                for i in range(depth)])
        else:
            self.blocks_audio = self.blocks

        # do not share norm layer
        self.norm_audio = norm_layer(embed_dim_audio)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, x_audio, padded_video, padded_audio):
        for blk, blk_audio in zip(self.blocks, self.blocks_audio):
            x, x_audio = blk(x, context=x_audio, x_mask=padded_video, context_mask=padded_audio), blk_audio(x_audio,
                                                                                                            context=x,
                                                                                                            x_mask=padded_audio,
                                                                                                            context_mask=padded_video)
        # norm
        x = self.norm(x)

        x_audio = self.norm_audio(x_audio)

        return x, x_audio


# decoder with hierarchical skip connections
class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196, tubelet_size=2,
                 ):
        super().__init__()
        self.num_classes = num_classes
        # assert num_classes == 3 * tubelet_size * patch_size ** 2
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                self.blocks.append(
                    Block(
                        dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                        init_values=init_values)
                )
            else:
                self.blocks.append(
                    CSBlock(
                        dim=embed_dim, context_dim=embed_dim, num_heads=num_heads, num_cross_heads=num_heads,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                        init_values=init_values)
                )

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num, x_skip_connects=None, full_mask=None, vis_mask=None):
        for i, blk in enumerate(self.blocks):
            if i == 0:
                x = blk(x, mask=full_mask)
            else:  # hierarchical skip connections
                x = blk(x, context=x_skip_connects[-i], x_mask=full_mask, context_mask=vis_mask)
        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x))

        return x


class PretrainAudioVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 encoder_in_chans=3,
                 encoder_num_classes=0,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_num_classes=768,  # decoder_num_classes=1536
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 tubelet_size=1,
                 num_classes=0,  # avoid the error from create_fn in timm
                 in_chans=0,  # avoid the error from create_fn in timm
                 # audio
                 img_size_audio=(256, 128),  # (T, F)
                 patch_size_audio=16,
                 encoder_in_chans_audio=1,
                 encoder_embed_dim_audio=768,
                 encoder_depth_audio=12,
                 encoder_num_heads_audio=12,
                 decoder_num_classes_audio=16,
                 decoder_embed_dim_audio=512,
                 decoder_depth_audio=8,
                 decoder_num_heads_audio=8,
                 # fusion
                 encoder_fusion_depth=2,
                 encoder_fusion_num_heads=12,
                 # contrastive learning
                 inter_contrastive_temperature=0.07,
                 # intermediate layers
                 return_intermediate_features=(3, 6, 9),
                 max_length=12,
                 ):
        super().__init__()

        self.encoder_depth = encoder_depth
        self.return_intermediate_features = return_intermediate_features
        if self.return_intermediate_features is not None:
            assert len(
                self.return_intermediate_features) == decoder_depth - 1, f"Error: wrong layers (len({return_intermediate_features}) != {decoder_depth - 1}) for intermediate_features!"
            assert len(
                self.return_intermediate_features) == decoder_depth_audio - 1, f"Error: wrong layers (len({return_intermediate_features}) != {decoder_depth_audio - 1}) for intermediate_features!"
            for idx in self.return_intermediate_features:
                assert idx < encoder_depth and idx < encoder_depth_audio, f"Error: too large layer index ({idx})!"

        # video encoder
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_learnable_pos_emb=use_learnable_pos_emb,
            max_length=max_length
        )

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
        )

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)

        # audio encoder
        self.encoder_audio = PretrainVisionTransformerEncoder2D(
            img_size=img_size_audio,
            patch_size=patch_size_audio,
            in_chans=encoder_in_chans_audio,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim_audio,
            depth=encoder_depth_audio,
            num_heads=encoder_num_heads_audio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb
        )

        self.decoder_audio = PretrainVisionTransformerDecoder(
            patch_size=patch_size_audio,
            num_patches=self.encoder_audio.patch_embed.num_patches,
            num_classes=decoder_num_classes_audio,
            embed_dim=decoder_embed_dim_audio,
            depth=decoder_depth_audio,
            num_heads=decoder_num_heads_audio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=1 / 48,  # no meaning, just to scape 'assert'
        )

        self.encoder_to_decoder_audio = nn.Linear(encoder_embed_dim_audio, decoder_embed_dim_audio, bias=False)

        self.mask_token_audio = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim_audio))

        self.pos_embed_audio = get_sinusoid_encoding_table(self.encoder_audio.patch_embed.num_patches,
                                                           decoder_embed_dim_audio)

        trunc_normal_(self.mask_token_audio, std=.02)

        # cross-modal fusion encoder
        self.encoder_fusion = PretrainVisionTransformerEncoderForFusion(
            embed_dim=encoder_embed_dim,  # for video
            embed_dim_audio=encoder_embed_dim_audio,  # for audio
            depth=encoder_fusion_depth,
            num_heads=encoder_fusion_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
        )

        self.inter_contrastive_temperature = inter_contrastive_temperature

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask, x_audio, mask_audio, padded_video, padded_audio):
        # encoder: video
        x_vis, x_vis_inter_features, vis_padded_video, invis_padded_video = self.encoder(x, mask,
                                                                                         self.return_intermediate_features,
                                                                                         padded_video)  # [B, N_vis, C_e]

        # encoder: audio
        x_vis_audio, x_vis_audio_inter_features, vis_padded_audio, invis_padded_audio = self.encoder_audio(x_audio,
                                                                                                           mask_audio,
                                                                                                           self.return_intermediate_features,
                                                                                                           padded_audio)
        # hcmcl
        logits_per_video, logits_per_audio = [], []
        for x_vis_inter, x_vis_audio_inter in zip(x_vis_inter_features, x_vis_audio_inter_features):
            # pooling
            x_vis_inter[(vis_padded_video > 0)] = 0
            vis_video_num = (vis_padded_video == 0)[:, :, 0].sum(dim=1)[:, None]
            video_features_inter = x_vis_inter.sum(dim=1) / vis_video_num  # (B, C)
            x_vis_audio_inter[(vis_padded_audio > 0)] = 0
            vis_audio_num = (vis_padded_audio == 0)[:, :, 0].sum(dim=1)[:, None]
            audio_features_inter = x_vis_audio_inter.sum(dim=1) / vis_audio_num  # (B, C)

            # normalized features
            video_features_inter = video_features_inter / video_features_inter.norm(dim=1, keepdim=True)
            audio_features_inter = audio_features_inter / audio_features_inter.norm(dim=1, keepdim=True)
            # cosine similarity as logits
            logits_per_video_inter = video_features_inter @ audio_features_inter.t() / self.inter_contrastive_temperature
            logits_per_audio_inter = logits_per_video_inter.t()
            # if logits_per_video_inter.isnan().sum() >0 or logits_per_audio_inter.isnan().sum() >0 :
            #     import pdb; pdb.set_trace()
            logits_per_video.append(logits_per_video_inter)
            logits_per_audio.append(logits_per_audio_inter)

        # encoder: fusion
        x_vis, x_vis_audio = self.encoder_fusion(x_vis, x_vis_audio, vis_padded_video[:, :, 0],
                                                 vis_padded_audio[:, :, 0])
        # decoder: video
        x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accordingly.
        _, pos_N = mask.shape
        expand_pos_embed = self.pos_embed[:, :pos_N, :].expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)  # [B, N, C_d]
        padded_video_full = torch.cat([vis_padded_video[:, :, 0], invis_padded_video[:, :, 0]], dim=1)
        x_vis_inter_features = [self.encoder_to_decoder(feature) + pos_emd_vis for feature in x_vis_inter_features]
        x = self.decoder(x_full, pos_emd_mask.shape[1], x_vis_inter_features, padded_video_full,
                         vis_padded_video[:, :, 0])  # [B, N_mask, 2 * 3 * 16 * 16]
        # decoder: audio
        x_vis_audio = self.encoder_to_decoder_audio(x_vis_audio)  # [B, N_vis, C_d]
        B_audio, N_audio, C_audio = x_vis_audio.shape
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accordingly.
        expand_pos_embed_audio = self.pos_embed_audio.expand(B_audio, -1, -1).type_as(x_audio).to(
            x_audio.device).clone().detach()
        pos_emd_vis_audio = expand_pos_embed_audio[~mask_audio].reshape(B_audio, -1, C_audio)
        pos_emd_mask_audio = expand_pos_embed_audio[mask_audio].reshape(B_audio, -1, C_audio)
        x_full_audio = torch.cat([x_vis_audio + pos_emd_vis_audio, self.mask_token_audio + pos_emd_mask_audio],
                                 dim=1)  # [B, N, C_d]
        padded_audio_full = torch.cat([vis_padded_audio[:, :, 0], invis_padded_audio[:, :, 0]], dim=1)
        x_vis_audio_inter_features = [self.encoder_to_decoder_audio(feature) + pos_emd_vis_audio for feature in
                                      x_vis_audio_inter_features]
        x_audio = self.decoder_audio(x_full_audio, pos_emd_mask_audio.shape[1],
                                     x_vis_audio_inter_features, padded_audio_full,
                                     vis_padded_audio[:, :, 0])  # [B, N_mask, 1 * 16 * 16]

        return x, x_audio, logits_per_video, logits_per_audio, invis_padded_video[:, :, 0], invis_padded_audio[:, :, 0]


class PretrainAudioVisionTransformerNoMAE(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 encoder_in_chans=3,
                 encoder_num_classes=0,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_num_classes=768,  # decoder_num_classes=1536
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 tubelet_size=1,
                 num_classes=0,  # avoid the error from create_fn in timm
                 in_chans=0,  # avoid the error from create_fn in timm
                 # audio
                 img_size_audio=(256, 128),  # (T, F)
                 patch_size_audio=16,
                 encoder_in_chans_audio=1,
                 encoder_embed_dim_audio=768,
                 encoder_depth_audio=12,
                 encoder_num_heads_audio=12,
                 decoder_num_classes_audio=16,
                 decoder_embed_dim_audio=512,
                 decoder_depth_audio=8,
                 decoder_num_heads_audio=8,
                 # fusion
                 encoder_fusion_depth=2,
                 encoder_fusion_num_heads=12,
                 # contrastive learning
                 inter_contrastive_temperature=0.07,
                 # intermediate layers
                 return_intermediate_features=(3, 6, 9),
                 max_length=12,
                 ):
        super().__init__()

        self.encoder_depth = encoder_depth
        self.return_intermediate_features = return_intermediate_features
        if self.return_intermediate_features is not None:
            assert len(
                self.return_intermediate_features) == decoder_depth - 1, f"Error: wrong layers (len({return_intermediate_features}) != {decoder_depth - 1}) for intermediate_features!"
            assert len(
                self.return_intermediate_features) == decoder_depth_audio - 1, f"Error: wrong layers (len({return_intermediate_features}) != {decoder_depth_audio - 1}) for intermediate_features!"
            for idx in self.return_intermediate_features:
                assert idx < encoder_depth and idx < encoder_depth_audio, f"Error: too large layer index ({idx})!"

        # video encoder
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_learnable_pos_emb=use_learnable_pos_emb,
            max_length=max_length
        )

        # audio encoder
        self.encoder_audio = PretrainVisionTransformerEncoder2D(
            img_size=img_size_audio,
            patch_size=patch_size_audio,
            in_chans=encoder_in_chans_audio,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim_audio,
            depth=encoder_depth_audio,
            num_heads=encoder_num_heads_audio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb
        )

        self.inter_contrastive_temperature = inter_contrastive_temperature

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask, x_audio, mask_audio, padded_video, padded_audio):
        # encoder: video
        x_vis, x_vis_inter_features, vis_padded_video, invis_padded_video = self.encoder(x, mask,
                                                                                         self.return_intermediate_features,
                                                                                         padded_video)  # [B, N_vis, C_e]

        # encoder: audio
        x_vis_audio, x_vis_audio_inter_features, vis_padded_audio, invis_padded_audio = self.encoder_audio(x_audio,
                                                                                                           mask_audio,
                                                                                                           self.return_intermediate_features,
                                                                                                           padded_audio)
        # hcmcl
        logits_per_video, logits_per_audio = [], []
        for x_vis_inter, x_vis_audio_inter in zip(x_vis_inter_features, x_vis_audio_inter_features):
            # pooling
            x_vis_inter[(vis_padded_video > 0)] = 0
            vis_video_num = (vis_padded_video == 0)[:, :, 0].sum(dim=1)[:, None]
            video_features_inter = x_vis_inter.sum(dim=1) / vis_video_num  # (B, C)
            x_vis_audio_inter[(vis_padded_audio > 0)] = 0
            vis_audio_num = (vis_padded_audio == 0)[:, :, 0].sum(dim=1)[:, None]
            audio_features_inter = x_vis_audio_inter.sum(dim=1) / vis_audio_num  # (B, C)

            # normalized features
            video_features_inter = video_features_inter / video_features_inter.norm(dim=1, keepdim=True)
            audio_features_inter = audio_features_inter / audio_features_inter.norm(dim=1, keepdim=True)
            # cosine similarity as logits
            logits_per_video_inter = video_features_inter @ audio_features_inter.t() / self.inter_contrastive_temperature
            logits_per_audio_inter = logits_per_video_inter.t()
            # if logits_per_video_inter.isnan().sum() >0 or logits_per_audio_inter.isnan().sum() >0 :
            #     import pdb; pdb.set_trace()
            logits_per_video.append(logits_per_video_inter)
            logits_per_audio.append(logits_per_audio_inter)

        return x, x_audio, logits_per_video, logits_per_audio, None, None


class PretrainAudioVisionTransformerMesh(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 vertex_size=224,
                 patch_size=16,
                 encoder_in_chans=3,
                 encoder_num_classes=0,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_num_classes=768,  # decoder_num_classes=1536
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 tubelet_size=1,
                 num_classes=0,  # avoid the error from create_fn in timm
                 in_chans=0,  # avoid the error from create_fn in timm
                 # audio
                 img_size_audio=(256, 128),  # (T, F)
                 patch_size_audio=16,
                 encoder_in_chans_audio=1,
                 encoder_embed_dim_audio=768,
                 encoder_depth_audio=12,
                 encoder_num_heads_audio=12,
                 decoder_num_classes_audio=16,
                 decoder_embed_dim_audio=512,
                 decoder_depth_audio=8,
                 decoder_num_heads_audio=8,
                 # fusion
                 encoder_fusion_depth=2,
                 encoder_fusion_num_heads=12,
                 # contrastive learning
                 inter_contrastive_temperature=0.07,
                 # intermediate layers
                 return_intermediate_features=(3, 6, 9),
                 max_length=12,
                 ):
        super().__init__()

        self.encoder_depth = encoder_depth
        self.return_intermediate_features = return_intermediate_features
        if self.return_intermediate_features is not None:
            assert len(
                self.return_intermediate_features) == decoder_depth - 1, f"Error: wrong layers (len({return_intermediate_features}) != {decoder_depth - 1}) for intermediate_features!"
            assert len(
                self.return_intermediate_features) == decoder_depth_audio - 1, f"Error: wrong layers (len({return_intermediate_features}) != {decoder_depth_audio - 1}) for intermediate_features!"
            for idx in self.return_intermediate_features:
                assert idx < encoder_depth and idx < encoder_depth_audio, f"Error: too large layer index ({idx})!"

        # video encoder
        self.encoder = PretrainVisionTransformerEncoderMesh(
            vertex_size=vertex_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_learnable_pos_emb=use_learnable_pos_emb,
            max_length=max_length
        )

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=max_length,
            num_classes=vertex_size,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
        )

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(max_length, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)

        # audio encoder
        self.encoder_audio = PretrainVisionTransformerEncoder2D(
            img_size=img_size_audio,
            patch_size=patch_size_audio,
            in_chans=encoder_in_chans_audio,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim_audio,
            depth=encoder_depth_audio,
            num_heads=encoder_num_heads_audio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb
        )

        self.decoder_audio = PretrainVisionTransformerDecoder(
            patch_size=patch_size_audio,
            num_patches=self.encoder_audio.patch_embed.num_patches,
            num_classes=decoder_num_classes_audio,
            embed_dim=decoder_embed_dim_audio,
            depth=decoder_depth_audio,
            num_heads=decoder_num_heads_audio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=1 / 48,  # no meaning, just to scape 'assert'
        )

        self.encoder_to_decoder_audio = nn.Linear(encoder_embed_dim_audio, decoder_embed_dim_audio, bias=False)

        self.mask_token_audio = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim_audio))

        self.pos_embed_audio = get_sinusoid_encoding_table(self.encoder_audio.patch_embed.num_patches,
                                                           decoder_embed_dim_audio)

        trunc_normal_(self.mask_token_audio, std=.02)

        # cross-modal fusion encoder
        self.encoder_fusion = PretrainVisionTransformerEncoderForFusion(
            embed_dim=encoder_embed_dim,  # for video
            embed_dim_audio=encoder_embed_dim_audio,  # for audio
            depth=encoder_fusion_depth,
            num_heads=encoder_fusion_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
        )

        self.inter_contrastive_temperature = inter_contrastive_temperature

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask, x_audio, mask_audio, padded_vertex, padded_audio):
        # encoder: video
        x_vis, x_vis_inter_features, vis_padded_vertex, invis_padded_vertex = self.encoder(x, mask,
                                                                                           self.return_intermediate_features,
                                                                                           padded_vertex)  # [B, N_vis, C_e]

        # encoder: audio
        x_vis_audio, x_vis_audio_inter_features, vis_padded_audio, invis_padded_audio = self.encoder_audio(x_audio,
                                                                                                           mask_audio,
                                                                                                           self.return_intermediate_features,
                                                                                                           padded_audio)
        # hcmcl
        logits_per_video, logits_per_audio = [], []
        for x_vis_inter, x_vis_audio_inter in zip(x_vis_inter_features, x_vis_audio_inter_features):
            # pooling
            x_vis_inter[(vis_padded_vertex > 0)] = 0
            vis_video_num = (vis_padded_vertex == 0)[:, :, 0].sum(dim=1)[:, None]
            video_features_inter = x_vis_inter.sum(dim=1) / vis_video_num  # (B, C)
            x_vis_audio_inter[(vis_padded_audio > 0)] = 0
            vis_audio_num = (vis_padded_audio == 0)[:, :, 0].sum(dim=1)[:, None]
            audio_features_inter = x_vis_audio_inter.sum(dim=1) / vis_audio_num  # (B, C)

            # normalized features
            video_features_inter = video_features_inter / video_features_inter.norm(dim=1, keepdim=True)
            audio_features_inter = audio_features_inter / audio_features_inter.norm(dim=1, keepdim=True)
            # cosine similarity as logits
            logits_per_video_inter = video_features_inter @ audio_features_inter.t() / self.inter_contrastive_temperature
            logits_per_audio_inter = logits_per_video_inter.t()
            # if logits_per_video_inter.isnan().sum() >0 or logits_per_audio_inter.isnan().sum() >0 :
            #     import pdb; pdb.set_trace()
            logits_per_video.append(logits_per_video_inter)
            logits_per_audio.append(logits_per_audio_inter)

        # encoder: fusion
        x_vis, x_vis_audio = self.encoder_fusion(x_vis, x_vis_audio, vis_padded_vertex[:, :, 0],
                                                 vis_padded_audio[:, :, 0])
        # decoder: video
        x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accordingly.
        _, pos_N = mask.shape
        expand_pos_embed = self.pos_embed[:, :pos_N, :].expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)  # [B, N, C_d]
        padded_vertex_full = torch.cat([vis_padded_vertex[:, :, 0], invis_padded_vertex[:, :, 0]], dim=1)
        x_vis_inter_features = [self.encoder_to_decoder(feature) + pos_emd_vis for feature in x_vis_inter_features]
        x = self.decoder(x_full, pos_emd_mask.shape[1], x_vis_inter_features, padded_vertex_full,
                         vis_padded_vertex[:, :, 0])  # [B, N_mask, 2 * 3 * 16 * 16]
        # decoder: audio
        x_vis_audio = self.encoder_to_decoder_audio(x_vis_audio)  # [B, N_vis, C_d]
        B_audio, N_audio, C_audio = x_vis_audio.shape
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accordingly.
        expand_pos_embed_audio = self.pos_embed_audio.expand(B_audio, -1, -1).type_as(x_audio).to(
            x_audio.device).clone().detach()
        pos_emd_vis_audio = expand_pos_embed_audio[~mask_audio].reshape(B_audio, -1, C_audio)
        pos_emd_mask_audio = expand_pos_embed_audio[mask_audio].reshape(B_audio, -1, C_audio)
        x_full_audio = torch.cat([x_vis_audio + pos_emd_vis_audio, self.mask_token_audio + pos_emd_mask_audio],
                                 dim=1)  # [B, N, C_d]
        padded_audio_full = torch.cat([vis_padded_audio[:, :, 0], invis_padded_audio[:, :, 0]], dim=1)
        x_vis_audio_inter_features = [self.encoder_to_decoder_audio(feature) + pos_emd_vis_audio for feature in
                                      x_vis_audio_inter_features]
        x_audio = self.decoder_audio(x_full_audio, pos_emd_mask_audio.shape[1],
                                     x_vis_audio_inter_features, padded_audio_full,
                                     vis_padded_audio[:, :, 0])  # [B, N_mask, 1 * 16 * 16]

        return x, x_audio, logits_per_video, logits_per_audio, invis_padded_vertex[:, :, 0], invis_padded_audio[:, :, 0]


class PretrainAudioVisionTransformerMeshNoMAE(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 vertex_size=224,
                 patch_size=16,
                 encoder_in_chans=3,
                 encoder_num_classes=0,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_num_classes=768,  # decoder_num_classes=1536
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 tubelet_size=1,
                 num_classes=0,  # avoid the error from create_fn in timm
                 in_chans=0,  # avoid the error from create_fn in timm
                 # audio
                 img_size_audio=(256, 128),  # (T, F)
                 patch_size_audio=16,
                 encoder_in_chans_audio=1,
                 encoder_embed_dim_audio=768,
                 encoder_depth_audio=12,
                 encoder_num_heads_audio=12,
                 decoder_num_classes_audio=16,
                 decoder_embed_dim_audio=512,
                 decoder_depth_audio=8,
                 decoder_num_heads_audio=8,
                 # fusion
                 encoder_fusion_depth=2,
                 encoder_fusion_num_heads=12,
                 # contrastive learning
                 inter_contrastive_temperature=0.07,
                 # intermediate layers
                 return_intermediate_features=(3, 6, 9),
                 max_length=12,
                 ):
        super().__init__()

        self.encoder_depth = encoder_depth
        self.return_intermediate_features = return_intermediate_features
        if self.return_intermediate_features is not None:
            assert len(
                self.return_intermediate_features) == decoder_depth - 1, f"Error: wrong layers (len({return_intermediate_features}) != {decoder_depth - 1}) for intermediate_features!"
            assert len(
                self.return_intermediate_features) == decoder_depth_audio - 1, f"Error: wrong layers (len({return_intermediate_features}) != {decoder_depth_audio - 1}) for intermediate_features!"
            for idx in self.return_intermediate_features:
                assert idx < encoder_depth and idx < encoder_depth_audio, f"Error: too large layer index ({idx})!"

        # video encoder
        self.encoder = PretrainVisionTransformerEncoderMesh(
            vertex_size=vertex_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_learnable_pos_emb=use_learnable_pos_emb,
            max_length=max_length
        )

        # audio encoder
        self.encoder_audio = PretrainVisionTransformerEncoder2D(
            img_size=img_size_audio,
            patch_size=patch_size_audio,
            in_chans=encoder_in_chans_audio,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim_audio,
            depth=encoder_depth_audio,
            num_heads=encoder_num_heads_audio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb
        )

        self.inter_contrastive_temperature = inter_contrastive_temperature

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask, x_audio, mask_audio, padded_vertex, padded_audio):
        # encoder: video
        x_vis, x_vis_inter_features, vis_padded_vertex, invis_padded_vertex = self.encoder(x, mask,
                                                                                           self.return_intermediate_features,
                                                                                           padded_vertex)  # [B, N_vis, C_e]

        # encoder: audio
        x_vis_audio, x_vis_audio_inter_features, vis_padded_audio, invis_padded_audio = self.encoder_audio(x_audio,
                                                                                                           mask_audio,
                                                                                                           self.return_intermediate_features,
                                                                                                           padded_audio)
        # hcmcl
        logits_per_video, logits_per_audio = [], []
        for x_vis_inter, x_vis_audio_inter in zip(x_vis_inter_features, x_vis_audio_inter_features):
            # pooling
            x_vis_inter[(vis_padded_vertex > 0)] = 0
            vis_video_num = (vis_padded_vertex == 0)[:, :, 0].sum(dim=1)[:, None]
            video_features_inter = x_vis_inter.sum(dim=1) / vis_video_num  # (B, C)
            x_vis_audio_inter[(vis_padded_audio > 0)] = 0
            vis_audio_num = (vis_padded_audio == 0)[:, :, 0].sum(dim=1)[:, None]
            audio_features_inter = x_vis_audio_inter.sum(dim=1) / vis_audio_num  # (B, C)

            # normalized features
            video_features_inter = video_features_inter / video_features_inter.norm(dim=1, keepdim=True)
            audio_features_inter = audio_features_inter / audio_features_inter.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logits_per_video_inter = video_features_inter @ audio_features_inter.t() / self.inter_contrastive_temperature
            logits_per_audio_inter = logits_per_video_inter.t()
            # if logits_per_video_inter.isnan().sum() >0 or logits_per_audio_inter.isnan().sum() >0 :
            #     import pdb; pdb.set_trace()
            logits_per_video.append(logits_per_video_inter)
            logits_per_audio.append(logits_per_audio_inter)

        x_vis[(vis_padded_vertex > 0)] = 0
        vis_video_num = (vis_padded_vertex == 0)[:, :, 0].sum(dim=1)[:, None]
        video_features = x_vis.sum(dim=1) / vis_video_num  # (B, C)
        x_vis_audio[(vis_padded_audio > 0)] = 0
        vis_audio_num = (vis_padded_audio == 0)[:, :, 0].sum(dim=1)[:, None]
        audio_features = x_vis_audio.sum(dim=1) / vis_audio_num  # (B, C)

        # normalized features
        video_features = video_features / video_features.norm(dim=1, keepdim=True)
        audio_features = audio_features / audio_features.norm(dim=1, keepdim=True)

        return video_features, audio_features, logits_per_video, logits_per_audio, None, None


class PretrainAudioVisionTransformerMeshConformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 vertex_size=224,
                 patch_size=16,
                 encoder_in_chans=3,
                 encoder_num_classes=0,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_num_classes=768,  # decoder_num_classes=1536
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 tubelet_size=1,
                 num_classes=0,  # avoid the error from create_fn in timm
                 in_chans=0,  # avoid the error from create_fn in timm
                 # audio
                 img_size_audio=(256, 128),  # (T, F)
                 patch_size_audio=16,
                 encoder_in_chans_audio=1,
                 encoder_embed_dim_audio=768,
                 encoder_depth_audio=12,
                 encoder_num_heads_audio=12,
                 decoder_num_classes_audio=16,
                 decoder_embed_dim_audio=512,
                 decoder_depth_audio=8,
                 decoder_num_heads_audio=8,
                 # fusion
                 encoder_fusion_depth=2,
                 encoder_fusion_num_heads=12,
                 # contrastive learning
                 inter_contrastive_temperature=0.07,
                 # intermediate layers
                 return_intermediate_features=(3, 6, 9),
                 max_length=12,
                 ):
        super().__init__()

        self.encoder_depth = encoder_depth
        self.return_intermediate_features = return_intermediate_features

        if self.return_intermediate_features is not None:
            for idx in self.return_intermediate_features:
                assert idx < encoder_depth and idx < encoder_depth_audio, f"Error: too large layer index ({idx})!"

        # video encoder
        self.encoder = PretrainVisionTransformerEncoderMesh(
            vertex_size=vertex_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_learnable_pos_emb=use_learnable_pos_emb,
            max_length=max_length
        )

        # audio encoder
        self.encoder_audio = ConformerEncoder(num_blocks=encoder_depth)

        self.inter_contrastive_temperature = inter_contrastive_temperature

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask, x_audio, padded_vertex):
        # encoder: video
        x_vis, x_vis_inter_features, vis_padded_vertex, invis_padded_vertex = self.encoder(x, mask,
                                                                                           self.return_intermediate_features,
                                                                                           padded_vertex)  # [B, N_vis, C_e]
        # encoder: audio
        x_vis_audio, x_vis_audio_inter_features = self.encoder_audio(x_audio, None, self.return_intermediate_features)

        # hcmcl
        logits_per_video, logits_per_audio = [], []
        for x_vis_inter, x_vis_audio_inter in zip(x_vis_inter_features, x_vis_audio_inter_features):
            # pooling
            video_features_inter = x_vis_inter.mean(dim=1)
            audio_features_inter = x_vis_audio_inter.mean(dim=1)  # (B, C)

            # normalized features
            video_features_inter = video_features_inter / video_features_inter.norm(dim=1, keepdim=True)
            audio_features_inter = audio_features_inter / audio_features_inter.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logits_per_video_inter = video_features_inter @ audio_features_inter.t() / self.inter_contrastive_temperature
            logits_per_audio_inter = logits_per_video_inter.t()
            # if logits_per_video_inter.isnan().sum() >0 or logits_per_audio_inter.isnan().sum() >0 :
            #     import pdb; pdb.set_trace()
            logits_per_video.append(logits_per_video_inter)
            logits_per_audio.append(logits_per_audio_inter)

        video_features = x_vis.mean(dim=1)
        audio_features = x_vis_audio.mean(dim=1)  # (B, C)

        # normalized features
        video_features = video_features / video_features.norm(dim=1, keepdim=True)
        audio_features = audio_features / audio_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_video = video_features @ audio_features.t() / self.inter_contrastive_temperature
        logit_audio = logit_video.t()
        # if logits_per_video_inter.isnan().sum() >0 or logits_per_audio_inter.isnan().sum() >0 :
        #     import pdb; pdb.set_trace()

        return logit_video, logit_audio, logits_per_video, logits_per_audio, None, None


class ProbPretrainAudioVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 encoder_in_chans=3,
                 encoder_num_classes=0,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_num_classes=768,  # decoder_num_classes=1536
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 tubelet_size=1,
                 num_classes=0,  # avoid the error from create_fn in timm
                 in_chans=0,  # avoid the error from create_fn in timm
                 # audio
                 img_size_audio=(256, 128),  # (T, F)
                 patch_size_audio=16,
                 encoder_in_chans_audio=1,
                 encoder_embed_dim_audio=768,
                 encoder_depth_audio=12,
                 encoder_num_heads_audio=12,
                 decoder_num_classes_audio=16,
                 decoder_embed_dim_audio=512,
                 decoder_depth_audio=8,
                 decoder_num_heads_audio=8,
                 # fusion
                 encoder_fusion_depth=2,
                 encoder_fusion_num_heads=12,
                 # contrastive learning
                 inter_contrastive_temperature=0.07,
                 # intermediate layers
                 return_intermediate_features=(2, 5, 8),
                 max_length=12,
                 ):
        super().__init__()

        self.encoder_depth = encoder_depth
        self.return_intermediate_features = return_intermediate_features
        if self.return_intermediate_features is not None:
            assert len(
                self.return_intermediate_features) == decoder_depth - 1, f"Error: wrong layers (len({return_intermediate_features}) != {decoder_depth - 1}) for intermediate_features!"
            assert len(
                self.return_intermediate_features) == decoder_depth_audio - 1, f"Error: wrong layers (len({return_intermediate_features}) != {decoder_depth_audio - 1}) for intermediate_features!"
            for idx in self.return_intermediate_features:
                assert idx < encoder_depth and idx < encoder_depth_audio, f"Error: too large layer index ({idx})!"

        # video encoder
        self.encoder = ProbPretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_learnable_pos_emb=use_learnable_pos_emb,
            max_length=max_length
        )

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
        )

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)

        # audio encoder
        self.encoder_audio = ProbPretrainVisionTransformerEncoder2D(
            img_size=img_size_audio,
            patch_size=patch_size_audio,
            in_chans=encoder_in_chans_audio,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim_audio,
            depth=encoder_depth_audio,
            num_heads=encoder_num_heads_audio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb
        )

        self.decoder_audio = PretrainVisionTransformerDecoder(
            patch_size=patch_size_audio,
            num_patches=self.encoder_audio.patch_embed.num_patches,
            num_classes=decoder_num_classes_audio,
            embed_dim=decoder_embed_dim_audio,
            depth=decoder_depth_audio,
            num_heads=decoder_num_heads_audio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=1 / 48,  # no meaning, just to scape 'assert'
        )

        self.encoder_to_decoder_audio = nn.Linear(encoder_embed_dim_audio, decoder_embed_dim_audio, bias=False)

        self.mask_token_audio = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim_audio))

        self.pos_embed_audio = get_sinusoid_encoding_table(self.encoder_audio.patch_embed.num_patches,
                                                           decoder_embed_dim_audio)

        trunc_normal_(self.mask_token_audio, std=.02)

        # cross-modal fusion encoder
        self.encoder_fusion = PretrainVisionTransformerEncoderForFusion(
            embed_dim=encoder_embed_dim,  # for video
            embed_dim_audio=encoder_embed_dim_audio,  # for audio
            depth=encoder_fusion_depth,
            num_heads=encoder_fusion_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
        )

        self.inter_contrastive_temperature = inter_contrastive_temperature
        shift = 5 * torch.ones(1)
        negative_scale = 5 * torch.ones(1)

        self.shift = nn.Parameter(shift)
        self.negative_scale = nn.Parameter(negative_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask, x_audio, mask_audio, padded_video, padded_audio):
        # encoder: video
        mean_x_vis, std_x_vis, x_vis_inter_features, vis_padded_video, invis_padded_video = self.encoder(x, mask,
                                                                                                         self.return_intermediate_features,
                                                                                                         padded_video)  # [B, N_vis, C_e]
        # encoder: audio
        mean_x_vis_audio, std_x_vis_audio, x_vis_audio_inter_features, vis_padded_audio, invis_padded_audio = self.encoder_audio(
            x_audio,
            mask_audio,
            self.return_intermediate_features,
            padded_audio)
        v_mean_x = mean_x_vis.clone()
        a_mean_x = mean_x_vis_audio.clone()
        v_std_x = std_x_vis.clone()
        a_std_x = std_x_vis_audio.clone()
        # hcmcl
        logits_per_video, logits_per_audio = [], []
        for x_vis_inter, x_vis_audio_inter in zip(x_vis_inter_features, x_vis_audio_inter_features):
            # pooling
            x_vis_inter[(vis_padded_video > 0)] = 0
            vis_video_num = (vis_padded_video == 0)[:, :, 0].sum(dim=1)[:, None]
            video_features_inter = x_vis_inter.sum(dim=1) / vis_video_num  # (B, C)
            x_vis_audio_inter[(vis_padded_audio > 0)] = 0
            vis_audio_num = (vis_padded_audio == 0)[:, :, 0].sum(dim=1)[:, None]
            audio_features_inter = x_vis_audio_inter.sum(dim=1) / vis_audio_num  # (B, C)

            # normalized features
            video_features_inter = video_features_inter / video_features_inter.norm(dim=1, keepdim=True)
            audio_features_inter = audio_features_inter / audio_features_inter.norm(dim=1, keepdim=True)
            # cosine similarity as logits
            logits_per_video_inter = video_features_inter @ audio_features_inter.t() / self.inter_contrastive_temperature
            logits_per_audio_inter = logits_per_video_inter.t()
            # if logits_per_video_inter.isnan().sum() >0 or logits_per_audio_inter.isnan().sum() >0 :
            #     import pdb; pdb.set_trace()
            logits_per_video.append(logits_per_video_inter)
            logits_per_audio.append(logits_per_audio_inter)
        # encoder: fusion
        x_vis, x_vis_audio = self.encoder_fusion(mean_x_vis, mean_x_vis_audio, vis_padded_video[:, :, 0],
                                                 vis_padded_audio[:, :, 0])
        # decoder: video
        x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accordingly.
        _, pos_N = mask.shape
        expand_pos_embed = self.pos_embed[:, :pos_N, :].expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)  # [B, N, C_d]
        padded_video_full = torch.cat([vis_padded_video[:, :, 0], invis_padded_video[:, :, 0]], dim=1)
        x_vis_inter_features = [self.encoder_to_decoder(feature) + pos_emd_vis for feature in x_vis_inter_features]
        x = self.decoder(x_full, pos_emd_mask.shape[1], x_vis_inter_features, padded_video_full,
                         vis_padded_video[:, :, 0])  # [B, N_mask, 2 * 3 * 16 * 16]
        # decoder: audio
        x_vis_audio = self.encoder_to_decoder_audio(x_vis_audio)  # [B, N_vis, C_d]
        B_audio, N_audio, C_audio = x_vis_audio.shape
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accordingly.
        expand_pos_embed_audio = self.pos_embed_audio.expand(B_audio, -1, -1).type_as(x_audio).to(
            x_audio.device).clone().detach()
        pos_emd_vis_audio = expand_pos_embed_audio[~mask_audio].reshape(B_audio, -1, C_audio)
        pos_emd_mask_audio = expand_pos_embed_audio[mask_audio].reshape(B_audio, -1, C_audio)
        x_full_audio = torch.cat([x_vis_audio + pos_emd_vis_audio, self.mask_token_audio + pos_emd_mask_audio],
                                 dim=1)  # [B, N, C_d]
        padded_audio_full = torch.cat([vis_padded_audio[:, :, 0], invis_padded_audio[:, :, 0]], dim=1)
        x_vis_audio_inter_features = [self.encoder_to_decoder_audio(feature) + pos_emd_vis_audio for feature in
                                      x_vis_audio_inter_features]
        x_audio = self.decoder_audio(x_full_audio, pos_emd_mask_audio.shape[1],
                                     x_vis_audio_inter_features, padded_audio_full,
                                     vis_padded_audio[:, :, 0])  # [B, N_mask, 1 * 16 * 16]

        v_mean_x[(vis_padded_video > 0)] = 0
        vis_video_num = (vis_padded_video == 0)[:, :, 0].sum(dim=1)[:, None]
        v_mean_x = v_mean_x.sum(dim=1) / vis_video_num  # (B, C)
        a_mean_x[(vis_padded_audio > 0)] = 0
        vis_audio_num = (vis_padded_audio == 0)[:, :, 0].sum(dim=1)[:, None]
        a_mean_x = a_mean_x.sum(dim=1) / vis_audio_num  # (B, C)

        v_std_x[(vis_padded_video > 0)] = 0
        vis_video_num = (vis_padded_video == 0)[:, :, 0].sum(dim=1)[:, None]
        v_std_x = v_std_x.sum(dim=1) / vis_video_num  # (B, C)
        a_std_x[(vis_padded_audio > 0)] = 0
        vis_audio_num = (vis_padded_audio == 0)[:, :, 0].sum(dim=1)[:, None]
        a_std_x = a_std_x.sum(dim=1) / vis_audio_num  # (B, C)

        mu_pdist = ((v_mean_x.unsqueeze(1) - a_mean_x.unsqueeze(0)) ** 2).sum(-1)
        sigma_pdist = (torch.exp(v_std_x).unsqueeze(1) + torch.exp(a_std_x).unsqueeze(0)).sum(-1)
        logits = mu_pdist + sigma_pdist

        logits = -self.negative_scale * logits + self.shift

        return x, x_audio, logits_per_video, logits_per_audio, invis_padded_video[:, :, 0], invis_padded_audio[:, :,
                                                                                            0], v_mean_x, v_std_x, a_mean_x, a_std_x, mu_pdist, sigma_pdist, logits


class MultiTaskPretrainAudioVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 encoder_in_chans=3,
                 encoder_num_classes=0,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_num_classes=768,  # decoder_num_classes=1536
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 tubelet_size=1,
                 num_classes=0,  # avoid the error from create_fn in timm
                 in_chans=0,  # avoid the error from create_fn in timm
                 # audio
                 img_size_audio=(256, 128),  # (T, F)
                 patch_size_audio=16,
                 encoder_in_chans_audio=1,
                 encoder_embed_dim_audio=768,
                 encoder_depth_audio=12,
                 encoder_num_heads_audio=12,
                 decoder_num_classes_audio=16,
                 decoder_embed_dim_audio=512,
                 decoder_depth_audio=8,
                 decoder_num_heads_audio=8,
                 # fusion
                 encoder_fusion_depth=2,
                 encoder_fusion_num_heads=12,
                 # contrastive learning
                 inter_contrastive_temperature=0.07,
                 # intermediate layers
                 return_intermediate_features=(3, 6, 9),
                 max_length=12,
                 ):
        super().__init__()

        self.encoder_depth = encoder_depth
        self.return_intermediate_features = return_intermediate_features
        if self.return_intermediate_features is not None:
            assert len(
                self.return_intermediate_features) == decoder_depth - 1, f"Error: wrong layers (len({return_intermediate_features}) != {decoder_depth - 1}) for intermediate_features!"
            assert len(
                self.return_intermediate_features) == decoder_depth_audio - 1, f"Error: wrong layers (len({return_intermediate_features}) != {decoder_depth_audio - 1}) for intermediate_features!"
            for idx in self.return_intermediate_features:
                assert idx < encoder_depth and idx < encoder_depth_audio, f"Error: too large layer index ({idx})!"

        # video encoder
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=encoder_in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_learnable_pos_emb=use_learnable_pos_emb,
            max_length=max_length
        )

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
        )

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)

        # audio encoder
        self.encoder_audio = PretrainVisionTransformerEncoder2D(
            img_size=img_size_audio,
            patch_size=patch_size_audio,
            in_chans=encoder_in_chans_audio,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim_audio,
            depth=encoder_depth_audio,
            num_heads=encoder_num_heads_audio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb
        )

        self.decoder_audio = PretrainVisionTransformerDecoder(
            patch_size=patch_size_audio,
            num_patches=self.encoder_audio.patch_embed.num_patches,
            num_classes=decoder_num_classes_audio,
            embed_dim=decoder_embed_dim_audio,
            depth=decoder_depth_audio,
            num_heads=decoder_num_heads_audio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=1 / 48,  # no meaning, just to scape 'assert'
        )

        self.encoder_to_decoder_audio = nn.Linear(encoder_embed_dim_audio, decoder_embed_dim_audio, bias=False)

        self.mask_token_audio = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim_audio))

        self.pos_embed_audio = get_sinusoid_encoding_table(self.encoder_audio.patch_embed.num_patches,
                                                           decoder_embed_dim_audio)
        trunc_normal_(self.mask_token_audio, std=.02)
        self.phoneme_viseme_mlp_v = Mlp(in_features=512, hidden_features=2048, out_features=512, act_layer=nn.GELU,
                                        drop=0.)
        self.own_pair_mlp_v = Mlp(in_features=512, hidden_features=2048, out_features=512, act_layer=nn.GELU, drop=0.)
        self.phoneme_viseme_mlp_a = Mlp(in_features=512, hidden_features=2048, out_features=512, act_layer=nn.GELU,
                                        drop=0.)
        self.own_pair_mlp_a = Mlp(in_features=512, hidden_features=2048, out_features=512, act_layer=nn.GELU, drop=0.)

        # cross-modal fusion encoder
        self.encoder_fusion = PretrainVisionTransformerEncoderForFusion(
            embed_dim=encoder_embed_dim,  # for video
            embed_dim_audio=encoder_embed_dim_audio,  # for audio
            depth=encoder_fusion_depth,
            num_heads=encoder_fusion_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
        )

        self.inter_contrastive_temperature = inter_contrastive_temperature

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask, x_audio, mask_audio, padded_video, padded_audio):
        # encoder: video
        x_vis, x_vis_inter_features, vis_padded_video, invis_padded_video = self.encoder(x, mask,
                                                                                         self.return_intermediate_features,
                                                                                         padded_video)  # [B, N_vis, C_e]
        # encoder: audio
        x_vis_audio, x_vis_audio_inter_features, vis_padded_audio, invis_padded_audio = self.encoder_audio(x_audio,
                                                                                                           mask_audio,
                                                                                                           self.return_intermediate_features,
                                                                                                           padded_audio)
        # hcmcl
        x_vis_inter = x_vis_inter_features[-1]
        x_vis_audio_inter = x_vis_audio_inter_features[-1]
        x_vis_inter[(vis_padded_video > 0)] = 0
        vis_video_num = (vis_padded_video == 0)[:, :, 0].sum(dim=1)[:, None]
        video_features_inter = x_vis_inter.sum(dim=1) / vis_video_num  # (B, C)
        x_vis_audio_inter[(vis_padded_audio > 0)] = 0
        vis_audio_num = (vis_padded_audio == 0)[:, :, 0].sum(dim=1)[:, None]
        audio_features_inter = x_vis_audio_inter.sum(dim=1) / vis_audio_num  # (B, C)
        video_features_inter_p_v = self.phoneme_viseme_mlp_v(video_features_inter)
        video_features_inter_own = self.own_pair_mlp_v(video_features_inter)
        audio_features_inter_p_v = self.phoneme_viseme_mlp_a(audio_features_inter)
        audio_features_inter_own = self.own_pair_mlp_a(audio_features_inter)

        video_features_inter_p_v = video_features_inter_p_v / video_features_inter_p_v.norm(dim=1, keepdim=True)
        audio_features_inter_p_v = audio_features_inter_p_v / audio_features_inter_p_v.norm(dim=1, keepdim=True)

        logits_per_video_inter_p_v = video_features_inter_p_v @ audio_features_inter_p_v.t() / self.inter_contrastive_temperature
        logits_per_audio_inter_p_v = logits_per_video_inter_p_v.t()

        video_features_inter_own = video_features_inter_own / video_features_inter_own.norm(dim=1, keepdim=True)
        audio_features_inter_own = audio_features_inter_own / audio_features_inter_own.norm(dim=1, keepdim=True)

        logits_per_video_inter_own = video_features_inter_own @ audio_features_inter_own.t() / self.inter_contrastive_temperature
        logits_per_audio_inter_own = logits_per_video_inter_own.t()

        # encoder: fusion
        x_vis, x_vis_audio = self.encoder_fusion(x_vis, x_vis_audio, vis_padded_video[:, :, 0],
                                                 vis_padded_audio[:, :, 0])
        # decoder: video
        x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accordingly.
        _, pos_N = mask.shape
        expand_pos_embed = self.pos_embed[:, :pos_N, :].expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)  # [B, N, C_d]
        padded_video_full = torch.cat([vis_padded_video[:, :, 0], invis_padded_video[:, :, 0]], dim=1)
        x_vis_inter_features = [self.encoder_to_decoder(feature) + pos_emd_vis for feature in x_vis_inter_features]
        x = self.decoder(x_full, pos_emd_mask.shape[1], x_vis_inter_features, padded_video_full,
                         vis_padded_video[:, :, 0])  # [B, N_mask, 2 * 3 * 16 * 16]
        # decoder: audio
        x_vis_audio = self.encoder_to_decoder_audio(x_vis_audio)  # [B, N_vis, C_d]
        B_audio, N_audio, C_audio = x_vis_audio.shape
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accordingly.
        expand_pos_embed_audio = self.pos_embed_audio.expand(B_audio, -1, -1).type_as(x_audio).to(
            x_audio.device).clone().detach()
        pos_emd_vis_audio = expand_pos_embed_audio[~mask_audio].reshape(B_audio, -1, C_audio)
        pos_emd_mask_audio = expand_pos_embed_audio[mask_audio].reshape(B_audio, -1, C_audio)
        x_full_audio = torch.cat([x_vis_audio + pos_emd_vis_audio, self.mask_token_audio + pos_emd_mask_audio],
                                 dim=1)  # [B, N, C_d]
        padded_audio_full = torch.cat([vis_padded_audio[:, :, 0], invis_padded_audio[:, :, 0]], dim=1)
        x_vis_audio_inter_features = [self.encoder_to_decoder_audio(feature) + pos_emd_vis_audio for feature in
                                      x_vis_audio_inter_features]
        x_audio = self.decoder_audio(x_full_audio, pos_emd_mask_audio.shape[1],
                                     x_vis_audio_inter_features, padded_audio_full,
                                     vis_padded_audio[:, :, 0])  # [B, N_mask, 1 * 16 * 16]

        return x, x_audio, logits_per_video_inter_p_v, logits_per_audio_inter_p_v, logits_per_video_inter_own, logits_per_audio_inter_own, invis_padded_video[
                                                                                                                                           :,
                                                                                                                                           :,
                                                                                                                                           0], invis_padded_audio[
                                                                                                                                               :,
                                                                                                                                               :,
                                                                                                                                               0]


@register_model
def pretrain_hicmae_dim384_patch16_160_a256(pretrained=False, **kwargs):
    model = PretrainAudioVisionTransformer(
        img_size=160,
        patch_size=16,
        encoder_embed_dim=384,
        encoder_num_heads=6,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=192,
        decoder_num_heads=3,
        # audio
        img_size_audio=(64, 128),  # (T, F)
        patch_size_audio=16,
        encoder_embed_dim_audio=384,
        encoder_num_heads_audio=6,
        decoder_num_classes_audio=16,
        decoder_embed_dim_audio=192,
        decoder_num_heads_audio=3,
        # fusion
        encoder_fusion_num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        max_length=16,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_hicmae_dim512_patch16_160_a256(pretrained=False, **kwargs):
    model = PretrainAudioVisionTransformer(
        img_size=160,
        patch_size=16,
        encoder_embed_dim=512,
        encoder_num_heads=8,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        # audio
        img_size_audio=(64, 128),  # (T, F)
        patch_size_audio=16,
        encoder_embed_dim_audio=512,
        encoder_num_heads_audio=8,
        decoder_num_classes_audio=16,
        decoder_embed_dim_audio=384,
        decoder_num_heads_audio=6,
        # fusion
        encoder_fusion_num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        max_length=5,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_hicmae_dim512_patch16_160_a256_no_mae(pretrained=False, **kwargs):
    model = PretrainAudioVisionTransformerNoMAE(
        img_size=160,
        patch_size=16,
        encoder_embed_dim=512,
        encoder_num_heads=8,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        # audio
        img_size_audio=(64, 128),  # (T, F)
        patch_size_audio=16,
        encoder_embed_dim_audio=512,
        encoder_num_heads_audio=8,
        decoder_num_classes_audio=16,
        decoder_embed_dim_audio=384,
        decoder_num_heads_audio=6,
        # fusion
        encoder_fusion_num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        max_length=5,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_hicmae_dim512_patch16_160_a256_prob(pretrained=False, **kwargs):
    model = ProbPretrainAudioVisionTransformer(
        img_size=160,
        patch_size=16,
        encoder_embed_dim=512,
        encoder_num_heads=8,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        # audio
        img_size_audio=(64, 128),  # (T, F)
        patch_size_audio=16,
        encoder_embed_dim_audio=512,
        encoder_num_heads_audio=8,
        decoder_num_classes_audio=16,
        decoder_embed_dim_audio=384,
        decoder_num_heads_audio=6,
        # fusion
        encoder_fusion_num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        max_length=16,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_hicmae_dim512_patch16_160_a256_multitask(pretrained=False, **kwargs):
    model = MultiTaskPretrainAudioVisionTransformer(
        img_size=160,
        patch_size=16,
        encoder_embed_dim=512,
        encoder_num_heads=8,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        # audio
        img_size_audio=(64, 128),  # (T, F)
        patch_size_audio=16,
        encoder_embed_dim_audio=512,
        encoder_num_heads_audio=8,
        decoder_num_classes_audio=16,
        decoder_embed_dim_audio=384,
        decoder_num_heads_audio=6,
        # fusion
        encoder_fusion_num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        max_length=16,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_hicmae_dim512_patch16_160_a256_mesh(pretrained=False, **kwargs):
    model = PretrainAudioVisionTransformerMesh(
        vertex_size=254 * 3,
        patch_size=254 * 3,
        encoder_embed_dim=512,
        encoder_num_heads=8,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        # audio
        img_size_audio=(64, 128),  # (T, F)
        patch_size_audio=16,
        encoder_embed_dim_audio=512,
        encoder_num_heads_audio=8,
        decoder_num_classes_audio=16,
        decoder_embed_dim_audio=384,
        decoder_num_heads_audio=6,
        # fusion
        encoder_fusion_num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        max_length=5,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_hicmae_dim512_patch16_160_a256_mesh_no_mae(pretrained=False, **kwargs):
    model = PretrainAudioVisionTransformerMeshNoMAE(
        vertex_size=254 * 3,
        patch_size=254 * 3,
        encoder_embed_dim=512,
        encoder_num_heads=8,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        # audio
        img_size_audio=(64, 128),  # (T, F)
        patch_size_audio=16,
        encoder_embed_dim_audio=512,
        encoder_num_heads_audio=8,
        decoder_num_classes_audio=16,
        decoder_embed_dim_audio=384,
        decoder_num_heads_audio=6,
        # fusion
        encoder_fusion_num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        max_length=5,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_hicmae_dim512_patch16_160_a256_mesh_no_mae_full_mesh(pretrained=False, **kwargs):
    model = PretrainAudioVisionTransformerMeshNoMAE(
        vertex_size=5023 * 3,
        patch_size=5023 * 3,
        encoder_embed_dim=512,
        encoder_num_heads=8,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        # audio
        img_size_audio=(64, 128),  # (T, F)
        patch_size_audio=16,
        encoder_embed_dim_audio=512,
        encoder_num_heads_audio=8,
        decoder_num_classes_audio=16,
        decoder_embed_dim_audio=384,
        decoder_num_heads_audio=6,
        # fusion
        encoder_fusion_num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        max_length=5,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_hicmae_dim512_patch16_160_a256_conformer_full_mesh(pretrained=False, **kwargs):
    model = PretrainAudioVisionTransformerMeshConformer(
        vertex_size=5023 * 3,
        patch_size=5023 * 3,
        encoder_embed_dim=768,
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        # audio
        img_size_audio=(64, 128),  # (T, F)
        patch_size_audio=16,
        encoder_embed_dim_audio=512,
        encoder_num_heads_audio=8,
        decoder_num_classes_audio=16,
        decoder_embed_dim_audio=384,
        decoder_num_heads_audio=6,
        # fusion
        encoder_fusion_num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        max_length=5,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def pretrain_hicmae_dim512_patch16_160_a256_full_mesh(pretrained=False, **kwargs):
    model = PretrainAudioVisionTransformerMesh(
        vertex_size=5023 * 3,
        patch_size=5023 * 3,
        encoder_embed_dim=512,
        encoder_num_heads=8,
        encoder_num_classes=0,
        decoder_num_classes=768,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        # audio
        img_size_audio=(64, 128),  # (T, F)
        patch_size_audio=16,
        encoder_embed_dim_audio=512,
        encoder_num_heads_audio=8,
        decoder_num_classes_audio=16,
        decoder_embed_dim_audio=384,
        decoder_num_heads_audio=6,
        # fusion
        encoder_fusion_num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        max_length=5,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model