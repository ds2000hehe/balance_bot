# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare the balance bot USD for training.

The USD exported from the URDF importer has a few issues that hurt RL training. This script writes a
cleaned, flattened copy into the extension's ``assets/data`` folder:

1. ``metersPerUnit`` is tagged as 0.01 although the geometry is authored in meters. Isaac Sim's metrics
   assembler can rescale such a reference by 100x, so the tag is reset to 1.0.
2. (optional, on by default) The convex-hull wheel colliders are replaced by sphere colliders of the same
   radius. A convex hull of a wheel mesh is a polygon (max 64 vertices) and rolls bumpily; a sphere rolls
   perfectly smoothly and gives one clean contact point. ``--wheel_radius`` sets the radius of the real wheels
   instead (e.g. 0.0825 m for 6.5" hoverboard wheels); the wheel visuals are scaled radially to match.
3. Mass properties are made physically plausible. Masses you set are kept, but:

   * massless helper links (``base_link``, ``part_7``, ...) get a tiny explicit mass; otherwise PhysX
     substitutes a 1 kg default for each of them.
   * inertias still sized for the original CAD mass (e.g. 1e-6 kg.m^2 on a 3 kg wheel) are recomputed
     from the collision shapes, scaled to the authored mass.

Usage:

.. code-block:: bash

    ./isaaclab.sh -p scripts/tools/prepare_usd.py ~/Desktop/balance_bot1.usd

"""

import argparse
import os

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

DEFAULT_OUT = os.path.join(
    os.path.dirname(__file__), "..", "..", "source", "balance_bot", "balance_bot", "assets", "data", "balance_bot.usd"
)

parser = argparse.ArgumentParser(description="Clean up the balance bot USD for Isaac Lab training.")
parser.add_argument("input", type=str, help="Path to the source USD.")
parser.add_argument("--output", type=str, default=DEFAULT_OUT, help="Path of the cleaned USD.")
parser.add_argument("--robot_prim", type=str, default="/Root/balance_bot_urdf", help="Robot prim in the source USD.")
parser.add_argument("--keep_mesh_wheels", action="store_true", help="Keep the convex-hull wheel colliders.")
parser.add_argument("--wheel_radius", type=float, default=None, help="Real wheel radius [m] (default: from the mesh).")
parser.add_argument("--helper_mass", type=float, default=1e-3, help="Mass [kg] given to massless helper links.")
args = parser.parse_args()

WHEELS = ["left_wheel", "right_wheel"]


def _has_colliders(prim: Usd.Prim) -> bool:
    for child in Usd.PrimRange(prim):
        if child.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Get():
            return True
    return False


def main():
    src = Usd.Stage.Open(os.path.expanduser(args.input))
    # flatten so the output is self-contained (no dangling sublayers / payloads)
    layer = src.Flatten()
    stage = Usd.Stage.Open(layer)

    # 1. units
    print(f"[prepare_usd] metersPerUnit: {UsdGeom.GetStageMetersPerUnit(stage)} -> 1.0")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    robot = stage.GetPrimAtPath(args.robot_prim)
    assert robot.IsValid(), f"Robot prim '{args.robot_prim}' not found"

    # drop leftovers from the Physics Inspector / any physics scene: Isaac Lab creates its own scene
    for prim in list(stage.Traverse()):
        rel = prim.GetRelationship("physics:simulationOwner")
        if rel and rel.HasAuthoredTargets():
            rel.ClearTargets(True)
        if prim.IsA(UsdPhysics.Scene):
            print(f"[prepare_usd] removing physics scene {prim.GetPath()}")
            stage.RemovePrim(prim.GetPath())

    # 2. wheel colliders (before the mass pass, so wheel inertia is computed from the spheres)
    if not args.keep_mesh_wheels:
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        for name in WHEELS:
            wheel = stage.GetPrimAtPath(f"{args.robot_prim}/{name}")
            assert wheel.IsValid(), f"Wheel prim '{name}' not found"
            # disable existing mesh colliders
            for prim in Usd.PrimRange(wheel):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            # wheel extents in its own frame; the axle is the shortest axis, the radius the largest half-extent
            box = bbox_cache.ComputeRelativeBound(wheel, wheel).ComputeAlignedRange()
            center, half = box.GetMidpoint(), box.GetSize() / 2.0
            radius = max(half)
            if args.wheel_radius is not None:
                # scale the visuals radially (not along the axle) about the wheel center
                axle = min(range(3), key=lambda i: half[i])
                s = args.wheel_radius / radius
                scale = Gf.Vec3d(*[1.0 if i == axle else s for i in range(3)])
                pivot = Gf.Vec3d(*[0.0 if i == axle else center[i] * (1.0 - s) for i in range(3)])
                for child in wheel.GetChildren():
                    if child.GetName() == "visuals":
                        xform = UsdGeom.Xformable(child)
                        xform.AddTranslateOp(opSuffix="wheelRadius").Set(pivot)
                        xform.AddScaleOp(opSuffix="wheelRadius").Set(scale)
                print(f"[prepare_usd] {name}: radius {radius:.4f} -> {args.wheel_radius:.4f} m (visuals scaled x{s:.3f})")
                radius = args.wheel_radius
            sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(f"{wheel.GetPath()}/collision_sphere"))
            sphere.CreateRadiusAttr(radius)
            sphere.AddTranslateOp().Set(Gf.Vec3d(center))
            sphere.CreatePurposeAttr(UsdGeom.Tokens.guide)  # invisible in renders
            UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
            print(f"[prepare_usd] {name}: sphere collider r={radius:.4f} m at {tuple(round(c, 4) for c in center)}")

    # 3. mass properties
    for prim in Usd.PrimRange(robot):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        name = prim.GetName()
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass = mass_api.GetMassAttr().Get()
        if mass is None or mass < args.helper_mass:
            # massless CAD helper link (base_link, mate connectors, ...): PhysX would otherwise substitute 1 kg
            print(f"[prepare_usd] {name}: mass {mass} -> {args.helper_mass} kg")
            mass_api.CreateMassAttr(args.helper_mass)
            mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-6, 1e-6, 1e-6))
            continue
        # a mass edited by hand keeps the inertia of the original CAD mass; detect inertias that are implausibly
        # small for the mass (radius of gyration below 5 mm)
        inertia = mass_api.GetDiagonalInertiaAttr().Get()
        if inertia is not None and max(inertia) >= mass * 0.005**2:
            continue
        if _has_colliders(prim):
            # zero inertia = let PhysX compute it from the collision shapes, scaled to the authored mass
            print(f"[prepare_usd] {name}: inertia {inertia} inconsistent with {mass} kg -> computed from colliders")
            mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(0.0, 0.0, 0.0))
            prim.GetAttribute("physics:principalAxes").Clear()
        else:
            # no geometry: treat it as a small solid sphere (r = 1 cm)
            value = 0.4 * mass * 0.01**2
            print(f"[prepare_usd] {name}: inertia {inertia} inconsistent with {mass} kg -> point-like {value:.2e}")
            mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(value, value, value))

    out = os.path.abspath(os.path.expanduser(args.output))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    stage.GetRootLayer().Export(out)
    print(f"[prepare_usd] wrote {out}")


if __name__ == "__main__":
    main()
