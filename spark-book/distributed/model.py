"""Offline BERT-style masked-token model for communication experiments.

This is a new teaching model, not a pretrained BERT or a replacement for
the WordPiece BERT trained in Chapter 53. Tokens are synthetic integer IDs.
"""
import torch
from torch import nn

SEED = 271026


class TinyBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.words = nn.Embedding(64, 32)
        self.positions = nn.Embedding(12, 32)
        self.segments = nn.Embedding(2, 32)
        self.norm = nn.LayerNorm(32)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(32, 4, 64, dropout=0.0,
                activation="gelu", batch_first=True, norm_first=False)
            for _ in range(2)
        ])
        self.head = nn.Sequential(nn.Linear(32, 32), nn.GELU(),
                                  nn.LayerNorm(32), nn.Linear(32, 64))

    def forward(self, ids):
        pos = torch.arange(ids.shape[1], device=ids.device)
        seg = (pos >= 6).long().expand_as(ids)
        x = self.norm(self.words(ids) + self.positions(pos) + self.segments(seg))
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def make_model(device="cpu", dtype=torch.float64):
    torch.manual_seed(SEED)
    return TinyBert().to(device=device, dtype=dtype)


def batch(step, device="cpu"):
    # Eight global examples; exactly two supervised tokens per example.
    g = torch.Generator().manual_seed(SEED + 100 + step)
    ids = torch.randint(4, 64, (8, 12), generator=g)
    labels = torch.full_like(ids, -100)
    labels[:, [2, 8]] = ids[:, [2, 8]]
    ids[:, [2, 8]] = 1
    return ids.to(device), labels.to(device)


def loss(model, ids, labels):
    return nn.functional.cross_entropy(model(ids).flatten(0, 1), labels.flatten())


def flat(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
