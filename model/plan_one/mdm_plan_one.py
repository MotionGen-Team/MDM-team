"""方案一总装模型。"""

from __future__ import annotations

from typing import Any, Callable, Dict

import torch
import torch.nn as nn
import clip

from model.BERT.BERT_encoder import load_bert
from model.mdm import PositionalEncoding, TimestepEmbedder
from model.rotation2xyz import Rotation2xyz

from .global_branch.global_branch import GlobalBranch
from .local_branch.local_branch import LocalBranch
from .shared_embedding.shared_embedding import SharedEmbeddingBlock
from .tcn_refine.fusion import FusionBlock
from .tcn_refine.heads import RefineHeads
from .tcn_refine.residual_tcn import CoordinationResidualTCN


class PlanOneMDM(nn.Module):
    """
    方案一总装模型。

    当前版本负责把已经实现的四块模块串起来：
    1. Shared Embedding
    2. Local Branch
    3. Global Branch
    4. Refine

    说明：
    - 外部接口仍保持 `forward(x_t, timesteps, y)` 风格。
    - 当前先支持时间嵌入，以及外部直接传入的 `text_embed`。
    - raw text 的编码暂不在这个类里实现，后面如果要接官方文本编码器，再单独补。
    """

    def __init__(
        self,
        njoints: int,
        nfeats: int,
        data_rep: str = 'hml_vec',
        struct_joints_num: int = 22,
        d_model: int = 512,
        d_struct: int = 256,
        dropout: float = 0.1,
        global_layers: int = 2,
        global_ff_mult: int = 4,
        global_query_heads: int = 8,
        global_kv_heads: int = 2,
        pos_embed_max_len: int = 5000,
        text_cond_dim: int | None = None,
        cond_mode: str = 'no_cond',
        cond_mask_prob: float = 0.0,
        dataset: str = 'humanml',
        text_encoder_type: str = 'clip',
        clip_version: str = 'ViT-B/32',
        all_goal_joint_names: list[str] | None = None,
        translation: bool = True,
        pose_rep: str = 'rot6d',
        glob: bool = True,
        glob_rot: bool = True,
    ):
        super().__init__()
        self.njoints = njoints
        self.nfeats = nfeats
        self.data_rep = data_rep
        self.struct_joints_num = struct_joints_num
        self.d_model = d_model
        self.d_struct = d_struct
        self.out_dim = njoints * nfeats
        self.cond_mode = cond_mode
        self.cond_mask_prob = cond_mask_prob
        self.dataset = dataset
        self.text_encoder_type = text_encoder_type
        self.clip_version = clip_version
        self.translation = translation
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.all_goal_joint_names = all_goal_joint_names or []

        # 条件嵌入沿用官方的时间步嵌入形式。
        self.sequence_pos_encoder = PositionalEncoding(d_model, dropout, max_len=pos_embed_max_len)
        self.embed_timestep = TimestepEmbedder(d_model, self.sequence_pos_encoder)

        self.text_cond_dim = text_cond_dim
        self._text_encoder: Callable[[Any], torch.Tensor] | None = None
        self.text_encoder_name = 'external'
        self.clip_model = None

        if 'action' in self.cond_mode:
            raise NotImplementedError('PlanOneMDM currently supports only text or no_cond modes.')

        if 'text' in self.cond_mode:
            if self.text_encoder_type == 'clip':
                self.clip_model = self.load_and_freeze_clip(self.clip_version)
                self._text_encoder = self.clip_encode_text
                self.text_encoder_name = 'clip'
                if text_cond_dim is None:
                    self.text_cond_dim = 512
            elif self.text_encoder_type == 'bert':
                bert_model_path = 'distilbert/distilbert-base-uncased'
                self.clip_model = load_bert(bert_model_path)
                self._text_encoder = self.bert_encode_text
                self.text_encoder_name = 'bert'
                if text_cond_dim is None:
                    self.text_cond_dim = 768
            else:
                raise ValueError(f'Unsupported text_encoder_type: {self.text_encoder_type}')

        # 文本条件支持两种接法：
        # 1. 直接传入 text_embed
        # 2. 先给模型挂一个外部文本编码器，再传 raw text
        #
        # 对 clip/bert 这类已知编码维度的情况，直接构造普通 Linear，
        # 避免训练启动阶段在参数统计时碰到未初始化的 Lazy 参数。
        if self.text_cond_dim is None:
            self.text_proj = nn.LazyLinear(d_model)
        else:
            self.text_proj = nn.Linear(self.text_cond_dim, d_model)
        self.cond_norm = nn.LayerNorm(d_model)

        self.shared_embedding = SharedEmbeddingBlock(
            data_rep=data_rep,
            joints_num=struct_joints_num,
            d_model=d_model,
            d_struct=d_struct,
            dropout=dropout,
        )
        self.local_branch = LocalBranch(
            d_struct=d_struct,
            d_model=d_model,
            dropout=dropout,
        )
        self.global_branch = GlobalBranch(
            d_struct=d_struct,
            d_model=d_model,
            num_layers=global_layers,
            ff_mult=global_ff_mult,
            num_query_heads=global_query_heads,
            num_kv_heads=global_kv_heads,
            dropout=dropout,
        )
        self.fusion = FusionBlock(d_model=d_model)
        self.refine_heads = RefineHeads(d_model=d_model, out_dim=self.out_dim)
        self.residual_tcn = CoordinationResidualTCN(d_model=d_model, out_dim=self.out_dim, dropout=dropout)
        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version: str):
        clip_model, _ = clip.load(clip_version, device='cpu', jit=False)
        clip.model.convert_weights(clip_model)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model

    def clip_encode_text(self, raw_text: Any) -> torch.Tensor:
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device)
            zero_pad = torch.zeros(
                [texts.shape[0], default_context_length - context_length],
                dtype=texts.dtype,
                device=texts.device,
            )
            texts = torch.cat([texts, zero_pad], dim=1)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device)
        return self.clip_model.encode_text(texts).float().unsqueeze(0)

    def bert_encode_text(self, raw_text: Any):
        enc_text, mask = self.clip_model(raw_text)
        enc_text = enc_text.permute(1, 0, 2)
        mask = ~mask
        return enc_text, mask

    def mask_cond(self, cond: torch.Tensor, force_mask: bool = False) -> torch.Tensor:
        bs = cond.shape[-2]
        if force_mask:
            return torch.zeros_like(cond)
        if self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob
            ).view(1, bs, 1)
            return cond * (1.0 - mask)
        return cond

    def set_text_encoder(self, text_encoder: Callable[[Any], torch.Tensor], encoder_name: str = 'external') -> None:
        """
        给 plan_one 挂一个外部文本编码器。

        这样采样/训练侧仍然可以沿用 `model.encode_text(raw_text)` 的调用方式，
        只是具体编码器由外部注入，而不是在这个类里绑死 CLIP/BERT。
        """
        self._text_encoder = text_encoder
        self.text_encoder_name = encoder_name

    def encode_text(self, raw_text: Any) -> torch.Tensor:
        """
        兼容官方调用方式的文本编码入口。

        如果外部已经通过 `set_text_encoder(...)` 挂了编码器，就直接调用；
        否则给出明确报错，提示上游改为传 `text_embed` 或先注入编码器。
        """
        if self._text_encoder is None:
            raise RuntimeError(
                'PlanOneMDM has no active text encoder. '
                'Please either pass y[\"text_embed\"] directly, '
                'or call set_text_encoder(...) before using raw text prompts.'
            )
        return self._text_encoder(raw_text)

    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        y: Dict[str, Any] | None = None,
        return_dict: bool = False,
    ) -> torch.Tensor | Dict[str, Any]:
        """
        输入:
            x_t: [B, J, F, T]
            timesteps: [B]
            y: 条件字典，当前支持：
                - `text_embed`: 外部预先计算好的文本特征
                - `c`: 若已经有完整条件嵌入 `[1, B, D]`，可直接传入

        输出:
            默认返回:
                y_pred [B, J, F, T]

            如果 `return_dict=True`:
                返回包含各阶段中间结果的字典
        """
        if x_t.ndim != 4:
            raise ValueError(f'Expected x_t with shape [B, J, F, T], got {tuple(x_t.shape)}')
        if timesteps.ndim != 1:
            raise ValueError(f'Expected timesteps with shape [B], got {tuple(timesteps.shape)}')

        batch_size, _, _, nframes = x_t.shape
        # 先把时间步/文本等外部条件整理成统一的 c [1, B, D]，
        # 后面 Shared Embedding 和 Global Branch 都直接复用这份条件表示。
        c = self.build_condition(timesteps, y)

        # 第一层：结构适配 + 共享嵌入。
        # 输出结构化表示 h_struct 和时序共享表示 h_global。
        shared_outputs = self.shared_embedding(x_t, c=c)

        # 第二层局部分支：围绕 body groups 做局部时序建模，输出 L_t。
        local_outputs = self.local_branch(shared_outputs['h_struct'])

        # 第二层全局分支：先从 h_struct 构造 group summary，再把 summary
        # 融入 h_global，最后通过 GQA Transformer 输出 G_t。
        global_outputs = self.global_branch(
            h_struct=shared_outputs['h_struct'],
            h_global=shared_outputs['h_global'],
            c=c,
        )

        # 第三层输出与修正：
        # 1. 融合 local/global 两路特征
        # 2. 给出基础预测 y_base_raw
        # 3. 再通过 residual TCN 预测修正量 delta_raw
        fusion_outputs = self.fusion(local_outputs['l_t'], global_outputs['g_t'])
        head_outputs = self.refine_heads(fusion_outputs['f_t'])

        r_in = torch.cat([fusion_outputs['f_t'], head_outputs['y_base_latent']], dim=-1)
        residual_outputs = self.residual_tcn(r_in)

        # 最终输出仍保持和官方 MDM 一致的 motion-space 形状 [B, J, F, T]，
        # 这样 diffusion 外围和训练/采样接口都不需要跟着改。
        y_raw = head_outputs['y_base_raw'] + residual_outputs['delta_raw']
        y_pred = self.restore_motion_shape(y_raw, batch_size=batch_size, nframes=nframes)

        if not return_dict:
            return y_pred

        return {
            'c': c,
            'shared': shared_outputs,
            'local': local_outputs,
            'global': global_outputs,
            'fusion': fusion_outputs,
            'heads': head_outputs,
            'residual_tcn': residual_outputs,
            'y_raw': y_raw,
            'y_pred': y_pred,
        }

    def build_condition(self, timesteps: torch.Tensor, y: Dict[str, Any] | None = None) -> torch.Tensor:
        """
        构造统一条件嵌入 `c [1, B, D]`。

        当前优先级：
        1. 如果外部直接传了 `y['c']`，则直接使用
        2. 如果存在 `text_embed`，则投影后与 `time_emb` 相加
        3. 如果只有 `text`，则尝试通过 `encode_text(...)` 先编码，再与 `time_emb` 相加
        4. 否则退回纯 `time_emb`
        """
        if y is not None and isinstance(y.get('c'), torch.Tensor):
            c = y['c']
            if c.ndim == 2:
                c = c.unsqueeze(0)
            if c.ndim != 3 or c.shape[0] != 1:
                raise ValueError(f'Expected c with shape [1, B, D] or [B, D], got {tuple(c.shape)}')
            return c

        time_emb = self.embed_timestep(timesteps)
        cond = time_emb

        force_mask = bool(y.get('uncond', False)) if y is not None else False
        text_embed = None
        if y is not None and isinstance(y.get('text_embed'), torch.Tensor):
            text_embed = y['text_embed']
        elif y is not None and 'text' in y:
            # 和官方入口保持同一习惯：如果上游给的是 raw text，
            # 就通过 model.encode_text(...) 走统一文本编码接口。
            text_embed = self.encode_text(y['text'])

        # 默认条件是时间步嵌入，这样即使没有文本条件也能直接跑 diffusion 主链。
        if isinstance(text_embed, tuple):
            text_embed = text_embed[0]

        if isinstance(text_embed, torch.Tensor):
            if text_embed.ndim == 2:
                text_embed = text_embed.unsqueeze(0)
            if text_embed.ndim != 3:
                raise ValueError(
                    f'Expected text_embed with shape [1, B, D_text] or [B, D_text], got {tuple(text_embed.shape)}'
                )
            # 这里先只支持外部传入的 text embedding。
            # 这样总装模型不需要绑死某一种文本编码器，后续接 CLIP/BERT 时只需要
            # 在外面准备好 text_embed，再统一投影到 d_model 即可。
            text_cond = self.text_proj(self.mask_cond(text_embed, force_mask=force_mask))
            cond = self.cond_norm(time_emb + text_cond)

        return cond

    def restore_motion_shape(self, y_raw: torch.Tensor, batch_size: int, nframes: int) -> torch.Tensor:
        """
        把 `[T, B, C]` 恢复成模型对外输出的 `[B, J, F, T]`。
        """
        if y_raw.ndim != 3:
            raise ValueError(f'Expected y_raw with shape [T, B, C], got {tuple(y_raw.shape)}')

        timesteps, batch_dim, channels = y_raw.shape
        if batch_dim != batch_size:
            raise ValueError(f'Batch mismatch: expected {batch_size}, got {batch_dim}')
        if timesteps != nframes:
            raise ValueError(f'Frame mismatch: expected {nframes}, got {timesteps}')
        if channels != self.out_dim:
            raise ValueError(f'Channel mismatch: expected {self.out_dim}, got {channels}')

        # y_raw 在 refine 结束后仍是 token 形式 [T, B, C]。
        # 这里把它恢复成 diffusion 期待的 [B, J, F, T]。
        y_pred = y_raw.view(timesteps, batch_size, self.njoints, self.nfeats)
        y_pred = y_pred.permute(1, 2, 3, 0).contiguous()
        return y_pred

    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)
