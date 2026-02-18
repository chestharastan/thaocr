"""
ThaoNet — OCR Recognition Model

Architecture:  Image → [Backbone] → [Neck] → [Head] → Text

Each component has a registry. To add a new backbone/neck/head:
  1. Write your class in this file (or import it here)
  2. Decorate it with @register_backbone("name"), @register_neck("name"),
     or @register_head("name")
  3. Set the `type` field in config.yml — done!

No need to edit train.py, config.py, or any other file.
"""
import math
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════
#  Component Registries
# ═══════════════════════════════════════════════════════════════
BACKBONE_REGISTRY = {}
NECK_REGISTRY = {}
HEAD_REGISTRY = {}


def register_backbone(name: str):
    """Decorator to register a backbone class."""
    def wrapper(cls):
        BACKBONE_REGISTRY[name] = cls
        return cls
    return wrapper


def register_neck(name: str):
    """Decorator to register a neck class."""
    def wrapper(cls):
        NECK_REGISTRY[name] = cls
        return cls
    return wrapper


def register_head(name: str):
    """Decorator to register a head class."""
    def wrapper(cls):
        HEAD_REGISTRY[name] = cls
        return cls
    return wrapper


# ═══════════════════════════════════════════════════════════════
#  Building Blocks
# ═══════════════════════════════════════════════════════════════
class ConvBlock(nn.Module):
    """Conv2d + BatchNorm + ReLU."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Simple residual block: two conv layers with skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """x: [B, T, D] → x + PE[:, :T, :]"""
        return x + self.pe[:, :x.size(1), :]


# ═══════════════════════════════════════════════════════════════
#  Backbones — feature extraction from image
# ═══════════════════════════════════════════════════════════════
#
#  Each backbone takes [B, 1, H, W] and returns [B, C, H', W']
#  Must define: out_channels, h_divisor (how much H shrinks)

@register_backbone("resnet")
class ResNetBackbone(nn.Module):
    """
    ResNet-style CNN backbone for OCR.

    4 stages with residual blocks. Reduces H by 16x, W by 4x.

    Pooling schedule:
        Layer 1: c1 + MaxPool(2,2)  → H/2,  W/2
        Layer 2: c2 + MaxPool(2,2)  → H/4,  W/4
        Layer 3: c3 + MaxPool(2,1)  → H/8,  W/4
        Layer 4: c4 + MaxPool(2,1)  → H/16, W/4
    """
    h_divisor = 16   # H is divided by this
    w_divisor = 4    # W is divided by this

    def __init__(self, channels=(64, 128, 256, 256), **kwargs):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.out_channels = c4

        self.layer1 = nn.Sequential(
            ConvBlock(1, c1), ResBlock(c1), nn.MaxPool2d(2, 2),
        )
        self.layer2 = nn.Sequential(
            ConvBlock(c1, c2), ResBlock(c2), nn.MaxPool2d(2, 2),
        )
        self.layer3 = nn.Sequential(
            ConvBlock(c2, c3), ResBlock(c3), nn.MaxPool2d((2, 1)),
        )
        self.layer4 = nn.Sequential(
            ConvBlock(c3, c4), ResBlock(c4), nn.MaxPool2d((2, 1)),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


@register_backbone("lightweight")
class LightweightBackbone(nn.Module):
    """
    Lightweight CNN backbone — fewer layers, faster training.

    3 stages (no residual blocks). Reduces H by 8x, W by 4x.
    Good for quick experiments or smaller datasets.
    """
    h_divisor = 8
    w_divisor = 4

    def __init__(self, channels=(64, 128, 256, 256), **kwargs):
        super().__init__()
        c1, c2, c3 = channels[0], channels[1], channels[2]
        self.out_channels = c3

        self.layer1 = nn.Sequential(
            ConvBlock(1, c1), nn.MaxPool2d(2, 2),        # H/2, W/2
        )
        self.layer2 = nn.Sequential(
            ConvBlock(c1, c2), nn.MaxPool2d(2, 2),       # H/4, W/4
        )
        self.layer3 = nn.Sequential(
            ConvBlock(c2, c3), nn.MaxPool2d((2, 1)),     # H/8, W/4
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# ═══════════════════════════════════════════════════════════════
#  Necks — project backbone features to model dimension
# ═══════════════════════════════════════════════════════════════
#
#  Each neck takes [B, T, feat_dim] and returns [B, T, d_model]

@register_neck("linear")
class LinearNeck(nn.Module):
    """Linear projection + LayerNorm + Dropout."""

    def __init__(self, feat_dim: int, d_model: int, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.proj(x)


# ═══════════════════════════════════════════════════════════════
#  Heads — sequence modeling + classification
# ═══════════════════════════════════════════════════════════════
#
#  Each head takes [B, T, d_model] and returns [T, B, vocab_size] (CTC format)

@register_head("transformer_ctc")
class TransformerCTCHead(nn.Module):
    """Transformer Encoder + CTC classification head."""

    def __init__(self, d_model: int, vocab_size: int, nhead: int = 4,
                 num_layers: int = 4, dim_feedforward: int = 1024,
                 dropout: float = 0.1, **kwargs):
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        """x: [B, T, d_model] → [T, B, V]"""
        z = self.pos_enc(x)
        h = self.encoder(z)
        logits = self.fc(h)           # [B, T, V]
        return logits.permute(1, 0, 2)  # [T, B, V] for CTC


@register_head("bilstm_ctc")
class BiLSTMCTCHead(nn.Module):
    """Bidirectional LSTM + CTC classification head.

    Lighter and faster than Transformer. Good for simpler scripts
    or when training data is limited.
    """

    def __init__(self, d_model: int, vocab_size: int, num_layers: int = 2,
                 dropout: float = 0.1, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(d_model, vocab_size)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        """x: [B, T, d_model] → [T, B, V]"""
        h, _ = self.lstm(x)
        logits = self.fc(h)           # [B, T, V]
        return logits.permute(1, 0, 2)  # [T, B, V] for CTC


# ═══════════════════════════════════════════════════════════════
#  ThaoNet — Main Model (assembles components from registry)
# ═══════════════════════════════════════════════════════════════

class OCRRecModel(nn.Module):
    """
    ThaoNet — OCR Recognition Model

    Assembles backbone + neck + head from the registry based on config.
    To add a new component, just write a class and decorate it with
    @register_backbone, @register_neck, or @register_head.

    Input:  [B, 1, H, W]
    Output: [T, B, V]  (CTC format)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        cnn_channels=(64, 128, 256, 256),
        target_h: int = 48,
        neck_dropout: float = None,
        backbone_type: str = "resnet",
        neck_type: str = "linear",
        head_type: str = "transformer_ctc",
    ):
        super().__init__()
        if neck_dropout is None:
            neck_dropout = dropout

        # ─── Backbone ─────────────────────────────────────────
        backbone_cls = BACKBONE_REGISTRY.get(backbone_type)
        if backbone_cls is None:
            raise ValueError(
                f"Unknown backbone type: '{backbone_type}'. "
                f"Available: {list(BACKBONE_REGISTRY.keys())}"
            )
        self.backbone = backbone_cls(channels=cnn_channels)

        # Calculate feature dim after backbone
        cnn_out_ch = self.backbone.out_channels
        h_after_cnn = target_h // self.backbone.h_divisor
        feat_dim = cnn_out_ch * h_after_cnn

        # ─── Neck ─────────────────────────────────────────────
        neck_cls = NECK_REGISTRY.get(neck_type)
        if neck_cls is None:
            raise ValueError(
                f"Unknown neck type: '{neck_type}'. "
                f"Available: {list(NECK_REGISTRY.keys())}"
            )
        self.neck = neck_cls(feat_dim=feat_dim, d_model=d_model, dropout=neck_dropout)

        # ─── Head ─────────────────────────────────────────────
        head_cls = HEAD_REGISTRY.get(head_type)
        if head_cls is None:
            raise ValueError(
                f"Unknown head type: '{head_type}'. "
                f"Available: {list(HEAD_REGISTRY.keys())}"
            )
        self.head = head_cls(
            d_model=d_model,
            vocab_size=vocab_size,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    @classmethod
    def from_config(cls, model_cfg, vocab_size: int):
        """Create model from a ModelConfig object."""
        return cls(
            vocab_size=vocab_size,
            d_model=model_cfg.head.d_model,
            nhead=model_cfg.head.nhead,
            num_layers=model_cfg.head.num_layers,
            dim_feedforward=model_cfg.head.dim_feedforward,
            dropout=model_cfg.head.dropout,
            cnn_channels=tuple(model_cfg.backbone.channels),
            target_h=model_cfg.target_h,
            neck_dropout=model_cfg.neck.dropout,
            backbone_type=model_cfg.backbone.type,
            neck_type=model_cfg.neck.type,
            head_type=model_cfg.head.type,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, H, W]

        Returns:
            logits: [T, B, V]  (CTC format: time-first)
        """
        # Backbone: extract features
        f = self.backbone(x)              # [B, C, H', W']
        B, C, H, W = f.shape

        # Reshape: collapse height into channel, sequence along width
        f = f.permute(0, 3, 1, 2).contiguous()   # [B, W', C, H']
        f = f.view(B, W, C * H)                  # [B, T, feat_dim]

        # Neck: project to d_model
        z = self.neck(f)                          # [B, T, d_model]

        # Head: sequence modeling + classification
        return self.head(z)                       # [T, B, V]

    def count_params(self) -> dict:
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "total_M": total / 1e6}
