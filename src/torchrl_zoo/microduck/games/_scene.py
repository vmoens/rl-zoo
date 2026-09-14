# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license in the repository root.
"""Internal robot attachment and camera geometry shared by the game scenes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mujoco


def _yaw_quaternion(yaw: float) -> list[float]:
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def _quaternion_product(a: Sequence[float], b: Sequence[float]) -> list[float]:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def _camera_axes(position: Sequence[float], target: Sequence[float]) -> list[float]:
    """Return the ``xyaxes`` of a camera at ``position`` looking at ``target``."""
    forward = [t - p for t, p in zip(target, position)]
    norm = math.sqrt(sum(v * v for v in forward))
    forward = [v / norm for v in forward]
    up = (0.0, 0.0, 1.0)
    right = [
        forward[1] * up[2] - forward[2] * up[1],
        forward[2] * up[0] - forward[0] * up[2],
        forward[0] * up[1] - forward[1] * up[0],
    ]
    norm = math.sqrt(sum(v * v for v in right))
    right = [v / norm for v in right]
    camera_up = [
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    ]
    return right + camera_up


def _share_meshes(spec: mujoco.MjSpec) -> None:
    """Keep one copy of each mesh that was attached under several prefixes.

    :meth:`mujoco.MjSpec.attach` copies the child's meshes under the player's
    prefix, so a team of five carries five copies of every robot mesh. Point
    the other players' geoms at the first copy and delete the rest; the
    compiled model is unchanged apart from its mesh tables.
    """
    first: dict[str, str] = {}
    replacement: dict[str, str] = {}
    for mesh in spec.meshes:
        if "/" not in mesh.name:
            continue
        stem = mesh.name.split("/", 1)[1]
        kept = first.setdefault(stem, mesh.name)
        if kept != mesh.name:
            replacement[mesh.name] = kept
    for geom in spec.geoms:
        if geom.meshname in replacement:
            geom.meshname = replacement[geom.meshname]
    for mesh in list(spec.meshes):
        if mesh.name in replacement:
            spec.delete(mesh)


def _attach_duck(spec, robot_scene, prefix, frame, color):
    # MuJoCo is an optional dependency of the shared TorchRL components.
    import mujoco

    child = mujoco.MjSpec.from_file(str(robot_scene))
    for geom in list(child.worldbody.geoms):
        child.delete(geom)
    for light in list(child.worldbody.lights):
        child.delete(light)
    for key in list(child.keys):
        child.delete(key)
    for material in list(child.materials):
        if any(material.textures):
            child.delete(material)
        elif "shell" in material.name:
            material.rgba = color
    for texture in list(child.textures):
        child.delete(texture)
    spec.attach(child, prefix=prefix, frame=frame)
