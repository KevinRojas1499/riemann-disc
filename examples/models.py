from __future__ import annotations
import torch
from torch import nn, Tensor
from flow_matching.utils.manifolds import Manifold
from typing import Tuple



# Activation class
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * x


# Model class
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        time_dim: int = 1,
        cond_dim: int = 1,
        hidden_dim: int = 256,
        vocab_size: int = 4,
        context_len : int = 1,
        depth=3,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.input_layer = nn.Linear(input_dim + time_dim + cond_dim * context_len, hidden_dim)

        self.main = nn.Sequential(
            *[
                layer for i in range(depth - 1) for layer in (Swish(),
                nn.Linear(hidden_dim, hidden_dim)) 
            ]
        )
        self.v_out = nn.Sequential(
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.d_out = nn.Sequential(
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, context_len * vocab_size),
        )

    def forward(self, x: Tensor, t: Tensor, y : Tensor, s : Tensor) -> Tensor:
        B, N = y.shape
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim).float()

        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        h = torch.cat([x, t, y.view(-1,1).float()], dim=1)
        h = self.input_layer(h)
        output = self.main(h)
        v_out = self.v_out(output)
        d_out = self.d_out(output)

        return v_out.reshape(*sz), d_out.reshape(B, N, self.vocab_size)


class ProjectToTangent(nn.Module):
    """Projects a vector field onto the tangent plane at the input."""

    def __init__(self, vecfield: nn.Module, manifold: Manifold):
        super().__init__()
        self.vecfield = vecfield
        self.manifold = manifold

    def forward(self, x: Tensor, y : Tensor, t: Tensor, s=None) -> Tensor:
        if s is None:
            s = t.clone()
        x = self.manifold.projx(x)
        v, w = self.vecfield(x=x, y=y, t=t, s=s)
        v = self.manifold.proju(x, v)
        return v, w

def get_model(name, manifold, dim, hidden_dim, depth, vocab_size, context_len):
    if name == 'mlp':
        return ProjectToTangent(  # Ensures we can just use Euclidean divergence.
            MLP(  # Vector field in the ambient space.
                input_dim=dim,
                hidden_dim=hidden_dim,
                depth=depth,
                vocab_size=vocab_size,
                context_len=context_len
            ),
            manifold=manifold,
        )