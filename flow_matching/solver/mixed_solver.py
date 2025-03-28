# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import math
from contextlib import nullcontext
from math import ceil
from typing import Callable, Optional, Union

import torch
from torch import Tensor

from torch.nn import functional as F

from flow_matching.path import MixtureDiscreteProbPath

from flow_matching.solver.solver import Solver
from flow_matching.utils import categorical, ModelWrapper
from flow_matching.utils.manifolds import geodesic, Manifold
from .utils import get_nearest_times

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class RiemannianDiscreteODESolver(Solver):
    r"""Riemannian ODE solver
    Initialize the ``RiemannianODESolver``.

    Args:
        manifold (Manifold): the manifold to solve on.
        velocity_model (ModelWrapper): a velocity field model receiving :math:`(x,t)`
            and returning :math:`u_t(x)` which is assumed to lie on the tangent plane at `x`.
    """

    def __init__(self, manifold: Manifold, velocity_model: ModelWrapper):
        super().__init__()
        self.manifold = manifold
        self.velocity_model = velocity_model

    def sample(
        self,
        x_init: Tensor,
        y_init: Tensor,
        step_size: float,
        projx: bool = True,
        proju: bool = True,
        method: str = "euler",
        time_grid: Tensor = torch.tensor([0.0, 1.0]),
        return_intermediates: bool = False,
        verbose: bool = False,
        enable_grad: bool = False,
        div_free: Union[float, Callable[[float], float]] = 0.0,
        **model_extras,
    ) -> Tensor:
        step_fns = {
            "midpoint": _midpoint_step,
        }
        assert method in step_fns.keys(), f"Unknown method {method}"
        step_fn = step_fns[method]

        def velocity_func(x, t):
            return self.velocity_model(x=x, t=t, **model_extras)

        # --- Factor this out.
        time_grid = torch.sort(time_grid.to(device=x_init.device)).values

        # If step_size is float then t discretization is uniform with step size set by step_size.
        t_init = time_grid[0].item()
        t_final = time_grid[-1].item()
        assert (
            t_final - t_init
        ) > step_size, f"Time interval [min(time_grid), max(time_grid)] must be larger than step_size. Got a time interval [{t_init}, {t_final}] and step_size {step_size}."

        n_steps = math.ceil((t_final - t_init) / step_size)
        t_discretization = torch.tensor(
            [step_size * i for i in range(n_steps)] + [t_final],
            device=x_init.device,
        )
        # ---
        t0s = t_discretization[:-1]

        if verbose:
            if not TQDM_AVAILABLE:
                raise ImportError(
                    "tqdm is required for verbose mode. Please install it."
                )
            t0s = tqdm(t0s)

        if return_intermediates:
            xts = []
            yts = []
            i_ret = 0

        with torch.set_grad_enabled(enable_grad):
            xt = x_init
            yt = y_init
            for i, (t0, t1) in enumerate(zip(t0s, t_discretization[1:])):
                dt = t1 - t0
                xt_next = step_fn(
                    velocity_func,
                    xt,
                    t0,
                    dt,
                    manifold=self.manifold,
                    projx=projx,
                    proju=proju,
                )

                # Discrete step

                p_1t = self.model(x=yt, t=t0.repeat(xt.shape[0]), **model_extras)
                y1 = categorical(p_1t.to(dtype=torch.float32))

                if i == n_steps - 1:
                    yt = y1
                else:
                    # Compute u_t(x|x_t,x_1)
                    scheduler_output = self.path.scheduler(t=t0)

                    k_t = scheduler_output.alpha_t
                    d_k_t = scheduler_output.d_alpha_t

                    delta_1 = F.one_hot(y1, num_classes=self.vocabulary_size).to(
                        k_t.dtype
                    )
                    u = d_k_t / (1 - k_t) * delta_1

                    # Add divergence-free part
                    div_free_t = div_free(t0) if callable(div_free) else div_free

                    if div_free_t > 0:
                        p_0 = self.source_distribution_p[(None,) * yt.dim()]
                        u = u + div_free_t * d_k_t / (k_t * (1 - k_t)) * (
                            (1 - k_t) * p_0 + k_t * delta_1
                        )

                    # Set u_t(x_t|x_t,x_1) = 0
                    delta_t = F.one_hot(yt, num_classes=self.vocabulary_size)
                    u = torch.where(
                        delta_t.to(dtype=torch.bool), torch.zeros_like(u), u
                    )

                    # Sample x_t ~ u_t( \cdot |x_t,x_1)
                    intensity = u.sum(dim=-1)  # Assuming u_t(xt|xt,x1) := 0
                    mask_jump = torch.rand(
                        size=yt.shape, device=yt.device
                    ) < 1 - torch.exp(-dt * intensity)

                    if mask_jump.sum() > 0:
                        yt[mask_jump] = categorical(
                            u[mask_jump].to(dtype=torch.float32)
                        )

                if return_intermediates:
                    while (
                        i_ret < len(time_grid)
                        and t0 <= time_grid[i_ret]
                        and time_grid[i_ret] <= t1
                    ):
                        xts.append(
                            interp(self.manifold, xt, xt_next, t0, t1, time_grid[i_ret])
                        )
                        i_ret += 1
                        yts.append(yt)
                xt = xt_next

        if return_intermediates:
            return torch.stack(xts, dim=0)
        else:
            return xt


def interp(manifold, xt, xt_next, t, t_next, t_ret):
    return geodesic(manifold, xt, xt_next)(
        (t_ret - t) / (t_next - t).reshape(1)
    ).reshape_as(xt)


def _midpoint_step(
    velocity_model: Callable,
    xt: Tensor,
    t0: Tensor,
    dt: Tensor,
    manifold: Manifold,
    projx: bool = True,
    proju: bool = True,
) -> Tensor:
    r"""Perform a midpoint step on a manifold.

    Args:
        velocity_model (Callable): the velocity model
        xt (Tensor): tensor containing the state at time t0
        t0 (Tensor): the time at which this step is taken
        dt (Tensor): the step size
        manifold (Manifold): a manifold object
        projx (bool, optional): whether to project the state onto the manifold. Defaults to True.
        proju (bool, optional): whether to project the velocity onto the tangent plane. Defaults to True.

    Returns:
        Tensor: tensor containing the state after the step
    """
    velocity_fn = lambda x, t: (
        manifold.proju(x, velocity_model(x, t)) if proju else velocity_model(x, t)
    )
    projx_fn = lambda x: manifold.projx(x) if projx else x

    half_dt = 0.5 * dt
    vt = velocity_fn(xt, t0)
    x_mid = xt + half_dt * vt
    x_mid = projx_fn(x_mid)

    xt = xt + dt * velocity_fn(x_mid, t0 + half_dt)

    return projx_fn(xt)
