# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Domain randomization events specific to the balance bot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_rigid_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    scale_range: tuple[float, float],
):
    """Scale the inertia tensor of the selected bodies by one random factor per environment (on top of the mass
    randomization, which scales inertia with mass): how hard the robot is to rotate is not known exactly either."""
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()
    body_ids = (
        torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
        if asset_cfg.body_ids == slice(None)
        else torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")
    )
    inertias = asset.root_physx_view.get_inertias()
    scale = torch.empty(len(env_ids), 1, 1, device="cpu").uniform_(*scale_range)
    inertias[env_ids[:, None], body_ids] *= scale
    asset.root_physx_view.set_inertias(inertias, env_ids)
