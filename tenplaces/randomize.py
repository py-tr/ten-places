"""Seeded domain randomisation. Every episode is fully determined by its integer seed.

Axes: object placement, weight, friction, shape, lighting, background.
"""
from dataclasses import dataclass, field, asdict

import numpy as np


@dataclass
class SceneParams:
    seed: int
    # spoon (the hand-off object): centre on arm A's side, yaw about vertical
    spoon_xy: tuple = (-0.09, 0.0)
    spoon_yaw: float = 0.0
    spoon_mass: float = 0.04
    spoon_length: float = 0.16
    spoon_width: float = 0.018
    # target where arm B must set the spoon down, on B's side
    target_xy: tuple = (0.09, 0.0)
    friction: float = 1.0
    light_dir: tuple = (0.0, 0.0, -1.0)
    light_diffuse: float = 0.7
    table_rgb: tuple = (0.55, 0.42, 0.30)
    floor_rgb: tuple = (0.25, 0.28, 0.32)
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def sample(seed: int, nominal: bool = False) -> SceneParams:
    """Draw scene parameters for one episode. `nominal=True` returns the unrandomised scene."""
    if nominal:
        return SceneParams(seed=seed)
    rng = np.random.default_rng(seed)
    u = rng.uniform
    return SceneParams(
        seed=seed,
        spoon_xy=(u(-0.12, -0.06), u(-0.07, 0.07)),
        spoon_yaw=u(-0.35, 0.35),
        spoon_mass=u(0.02, 0.07),
        spoon_length=u(0.15, 0.17),
        spoon_width=u(0.016, 0.022),
        target_xy=(u(0.07, 0.12), u(-0.07, 0.07)),
        friction=u(0.6, 1.4),
        light_dir=tuple(np.array([u(-0.5, 0.5), u(-0.5, 0.5), -1.0])),
        light_diffuse=u(0.4, 0.9),
        table_rgb=tuple(u(0.2, 0.8, 3)),
        floor_rgb=tuple(u(0.1, 0.5, 3)),
    )
