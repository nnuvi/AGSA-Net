import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class SWiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SWiGLU, self).__init__()
        self.fc = nn.Linear(input_dim, 2 * hidden_dim)

    def forward(self, x):
        out = self.fc(x)
        x_out, gate = out.chunk(2, dim=-1)
        return x_out * torch.sigmoid(gate) + x_out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0)) 

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]
    

# Abundance Guided Self Attention

class AbundanceGuidedAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0., lambda_init=1.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Learnable scalar to weight the abundance guidance
        self.lambda_abundance = nn.Parameter(torch.tensor(lambda_init))

    def forward(self, x, abundance=None):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Base Attention Scores (Spectral Similarity)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Abundance Guidance Injection
        if abundance is not None:

            # Normalize abundance for cosine similarity
            abn_norm = F.normalize(abundance, p=2, dim=-1)

            # Compute Affinity Matrix
            abn_affinity = abn_norm @ abn_norm.transpose(-2, -1)

            # Pad this affinity matrix to match the CLS token
            N_tokens = N
            N_pixels = abn_affinity.shape[1]

            # Create full guidance matrix initialized to 0
            guidance = torch.zeros(B, N_tokens, N_tokens, device=x.device)
            guidance[:, 1:, 1:] = abn_affinity
            guidance = guidance.unsqueeze(1)

            # Add guidance to attention scores
            attn = attn + (self.lambda_abundance * guidance)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# 3. Transformer Layer

class GuidedTransformerLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = AbundanceGuidedAttention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, abundance=None):
        # Pre-Norm Architecture
        x = x + self.attn(self.norm1(x), abundance)
        x = x + self.mlp(self.norm2(x))
        return x


# AGSA-Net 

class AGSANet(nn.Module):
    """
    Dual-branch Network:
    Branch 1: CNN Unmixer (Encoder-Decoder)
    Branch 2: Abundance-Guided DiffFormer (Transformer)
    """
    def __init__(self, band, num_classes, patch_size, depth, num_heads, basic_cls_name='conv2d_unmix'):
        super(AGSANet, self).__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.depth = depth
        self.num_heads = num_heads
        self.basic_cls_name = basic_cls_name


        # --- Unmixing Branch ---

        self.unmix_encoder = nn.Sequential(
            nn.Conv2d(band, band//2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(band//2),
            nn.ReLU(),
            nn.Conv2d(band//2, band//4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(band//4),
            nn.ReLU(),
            nn.Conv2d(band//4, num_classes, kernel_size=1, stride=1, padding=0)
        )
        self.unmix_decoder = nn.Sequential(
            nn.Conv2d(num_classes, band*2, kernel_size=1, stride=1, bias=False),
            nn.ReLU()
        )
        self.unmix_decoder_nonlinear = nn.Sequential(
            nn.Conv2d(band*2, band, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
            nn.Conv2d(band, band, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid()
        )

        # --- Transformer Branch ---

        # Transformer Config
        self.embed_dim = 64
        self.mlp_ratio = 2.0

        # Patch Embedding
        self.patch_embed = nn.Linear(band, self.embed_dim)

        # CLS Token & Pos Embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = PositionalEncoding(self.embed_dim, max_len=(patch_size**2) + 1)
        self.pos_drop = nn.Dropout(p=0.1)

        # Transformer Encoder
        self.blocks = nn.ModuleList([
            GuidedTransformerLayer(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                drop=0.1,
                attn_drop=0.1
            )
            for _ in range(self.depth)
        ])
        self.norm = nn.LayerNorm(self.embed_dim)

        # --- Fusion & Head ---

        # Abundance Feature processing
        self.abu_conv = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(),
        )

        self.abu_flat_dim = self._get_abundance_flattened_size()
        self.head = nn.Linear(self.embed_dim + self.abu_flat_dim, num_classes)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _get_abundance_flattened_size(self):
        with torch.no_grad():
            # Dummy forward to check size
            x = torch.zeros((1, self.num_classes, self.patch_size, self.patch_size))
            x = self.abu_conv(x)
            return x.numel()

    def forward(self, x):
        B, C, H, W = x.shape

        abu = self.unmix_encoder(x)  

        re_unmix = self.unmix_decoder(abu)
        re_unmix_nonlinear = self.unmix_decoder_nonlinear(re_unmix)

        abu_constrained = abu.abs()
        abu_constrained = abu_constrained / (abu_constrained.sum(dim=1, keepdim=True) + 1e-8)

        abu_guidance = abu_constrained.flatten(2).transpose(1, 2)

        embed_dim=self.embed_dim
        depth=self.depth
        num_heads=self.num_heads

        x_tokens = x.flatten(2).transpose(1, 2)
        x_embed = self.patch_embed(x_tokens) 

        cls_token = self.cls_token.expand(B, -1, -1)
        x_embed = torch.cat((cls_token, x_embed), dim=1) 

        x_embed = self.pos_embed(x_embed)
        x_embed = self.pos_drop(x_embed)

        for blk in self.blocks:
            x_embed = blk(x_embed, abundance=abu_guidance)

        x_out = self.norm(x_embed)
        cls_out = x_out[:, 0] 


        # Process Abundance map spatially for concatenation
        abu_feat = self.abu_conv(abu_constrained)
        abu_feat = abu_feat.view(B, -1)

        # Concatenate Transformer features + Abundance features
        fused = torch.cat((cls_out, abu_feat), dim=1)

        # Classification
        output_cls = self.head(fused)

        return re_unmix_nonlinear, re_unmix, output_cls, embed_dim, depth, num_heads