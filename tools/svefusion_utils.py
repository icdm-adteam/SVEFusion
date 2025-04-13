import torch
import torch.nn as nn
from functools import partial
from mamba_ssm.models.mixer_seq_simple import create_block
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn


class Permute(nn.Module):
    def __init__(self, *args):
        super(Permute, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.permute(self.shape)


class SVSO(nn.Module):
    def __init__(self, features_in, down_mlp_channels):
        super().__init__()
        
        assert len(down_mlp_channels) == 3
        assert down_mlp_channels[0] == features_in
        
        self.features_in = features_in
        
        self.down_mlp1 = nn.Sequential(
            nn.Conv1d(2 * self.features_in, down_mlp_channels[0], kernel_size=1),
        )
        
        self.down_mlp2 = nn.Sequential(
            nn.BatchNorm1d(down_mlp_channels[0]),
            nn.ReLU(),
            nn.Conv1d(down_mlp_channels[0], down_mlp_channels[1], kernel_size=1),
            nn.BatchNorm1d(down_mlp_channels[1]),
            nn.ReLU(),
            nn.Conv1d(down_mlp_channels[1], down_mlp_channels[2], kernel_size=1),
        )

    def forward(self, embeddings):
        assert embeddings.shape[2] == self.features_in
        
        batch_size, num_embeddings, features_in = embeddings.shape

        global_feat = embeddings.max(dim=1)[0]

        semantic_embeddings = torch.cat(
            [embeddings, global_feat.unsqueeze(1).repeat(1, num_embeddings, 1)], 
            dim=-1
        )

        mixed_semantic_embeddings = self.down_mlp1(semantic_embeddings.permute(0, 2, 1)).permute(0, 2, 1)
        scores = self.down_mlp2(mixed_semantic_embeddings.permute(0, 2, 1)).permute(0, 2, 1).squeeze(-1)

        sorted_indices = scores.argsort(dim=1, descending=True)
        
        sorted_embeddings = torch.gather(mixed_semantic_embeddings, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, features_in))

        return sorted_indices, sorted_embeddings
    

class MambaBlock(nn.Module):
    # single-directional mamba
    def __init__(self, 
                 d_model, 
                 ssm_cfg, 
                 norm_epsilon, 
                 rms_norm,
                 residual_in_fp32=True, 
                 fused_add_norm=True,
                 layer_idx=None,
                 device=None,
                 dtype=torch.float32):
        super().__init__()

        factory_kwargs = {'device': device, 'dtype':dtype}

        # mamba layer
        self.mamba_encoder_1 = create_block(
            d_model=d_model,
            ssm_cfg=ssm_cfg,
            norm_epsilon=norm_epsilon,
            rms_norm=rms_norm,
            residual_in_fp32=residual_in_fp32,
            fused_add_norm=fused_add_norm,
            layer_idx=layer_idx,
            **factory_kwargs,
        )

        norm_cls = partial(
            nn.LayerNorm, eps=norm_epsilon, **factory_kwargs
        )
        self.norm_input = norm_cls(d_model)
        self.norm_forward = norm_cls(d_model)
        
        self.x_silu = nn.SiLU()
        
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(
        self,
        x,
        pos_embed,
        debug=False,
        ):

        mamba_layer1 = self.mamba_encoder_1

        batch_size, num_embeddings, features = x.shape

        # Pos Embedding
        pos_embed_coords = torch.zeros([batch_size, num_embeddings, 9], device=x.device, dtype=torch.float32)
        pos_embed_coords[:, :, 0] = torch.arange(num_embeddings, device=x.device).float() / num_embeddings
        pos_embed_coords[:, :, 1:3] = (torch.arange(num_embeddings, device=x.device).float().view(-1, 1) // 12) / (num_embeddings // 12 + 1)
        pos_embed_coords[:, :, 3:5] = (torch.arange(num_embeddings, device=x.device).float().view(-1, 1) % 12) / 12.0
        pos_embed_coords[:, :, 5:7] = ((torch.arange(num_embeddings, device=x.device).float().view(-1, 1) + 6) // 12) / (num_embeddings // 12 + 1)
        pos_embed_coords[:, :, 7:9] = ((torch.arange(num_embeddings, device=x.device).float().view(-1, 1) + 6) % 12) / 12.0
        pos_embed = pos_embed(pos_embed_coords.float())

        x_pos_embed = x + pos_embed

        x_norm = self.norm_input(x_pos_embed)
        x_silu = self.x_silu(x_norm)

        # Forward SSMs
        out_feat_3d = torch.zeros_like(x)
        for i in range(batch_size):
            feat = x_norm[i][None]
            out_feat = mamba_layer1(feat, None)
            out_feat_3d[i] = out_feat[0].squeeze(0)

        out_feat_3d_norm = self.norm_forward(out_feat_3d)

        x_forward = torch.concat([out_feat_3d_norm, x_silu], dim=-1)
        mamba_encode = x_forward
        
        x = self.proj(mamba_encode) + x

        return x