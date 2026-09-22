"""WCSPH dam/sphere simulation with live rendering and PLY export.

Examples:
    python main.py --scene sphere --seconds 1 --live
    python main.py --scene sphere --steps 300 --output-dir results
"""
import argparse
import json
from pathlib import Path

import numpy as np

from warpSph import *
from visualization import LiveParticleRenderer, export_ply, save_particle_frame


def parse_args():
    parser = argparse.ArgumentParser(description="GPU WCSPH free-surface demo")
    parser.add_argument("--scene", choices=("cube", "sphere"), default="sphere")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=None,
                        help="Override duration with an exact number of solver steps")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--viscosity", type=float, default=1.0e-6,
                        help="Kinematic viscosity used by the Morris SPH term")
    parser.add_argument("--artificial-viscosity-alpha", type=float, default=0.1,
                        help="Approaching-pair artificial-viscosity coefficient")
    parser.add_argument("--live", action="store_true", help="Show an updating Matplotlib viewport")
    parser.add_argument("--usd", action="store_true", help="Also save a Warp USD point animation")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Synchronize density every step and write exact peak diagnostics")
    parser.add_argument("--motion-diagnostics", action="store_true",
                        help="Write per-frame RMS speed and kinetic-energy diagnostics")
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def build_fluid(scene, config):
    if scene == "sphere":
        # A solid sphere, initially suspended above the pool floor.
        return SphereData(span=config.partSize, center=vecxf([-1.0, 0.8, 0.0]), radius=1.15)
    return CubeData(span=config.partSize, dim=config.dim,
                    lb=vecxf([-2.0, -1.8, -1.2]), rt=vecxf([0.0, 0.2, 1.2]))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    wp.set_device(device)
    config = Config(dim=dim, partSize=wp.float32(0.2),
                    kViscosity=wp.float32(args.viscosity),
                    artificialViscosityAlpha=wp.float32(args.artificial_viscosity_alpha),
                    lb=vecxf(-5.0), rt=vecxf(5.0))
    sph = SPH(config)

    fluid_data = build_fluid(args.scene, config)
    pool_data = PoolData(containerHeight=wp.float32(6), containerWidth=wp.float32(6),
                         containerLayer=wp.int32(2), fluidHeight=wp.float32(0.1),
                         span=config.partSize)
    fluid_positions = wp.from_numpy(fluid_data.partPos, dtype=vecxf, device=device)
    boundary_positions = wp.from_numpy(pool_data.poolPartPos, dtype=vecxf, device=device)

    fluid = Particle(fluid_data.partNum, config)
    boundary = Particle(pool_data.poolPartNum, config)
    wp.launch(copy, dim=fluid.shape, inputs=[fluid.position, fluid_positions], device=device)
    wp.launch(copy, dim=boundary.shape, inputs=[boundary.position, boundary_positions], device=device)
    particle_mass = config.restDensity * config.partVolume
    wp.launch(set, dim=fluid.shape, inputs=[fluid.mass, particle_mass], device=device)
    wp.launch(set, dim=boundary.shape, inputs=[boundary.mass, particle_mass], device=device)

    fluid_grid = wp.HashGrid(dim_x=128, dim_y=128, dim_z=128, device=device)
    boundary_grid = wp.HashGrid(dim_x=128, dim_y=128, dim_z=128, device=device)
    viewport = LiveParticleRenderer(enabled=args.live)

    usd_renderer = None
    if args.usd:
        try:
            import warp.render as wpr
            usd_renderer = wpr.UsdRenderer(str(output_dir / "fluid.usd"), fps=args.fps)
        except Exception as exc:
            print(f"USD renderer unavailable ({exc}); PNG/PLY output continues.")

    max_steps = args.steps if args.steps is not None else int(args.seconds / float(config.dt))
    sim_time = 0.0
    frame = 0
    next_frame_time = 0.0
    peak_density = float(config.restDensity)
    peak_iteration = 0
    motion_samples = []
    print(f"device={device}, scene={args.scene}, fluid={fluid.shape}, boundary={boundary.shape}, "
          f"sound_speed={float(config.soundSpeed):.1f}, dt={float(config.dt):.6g}, "
          f"cfl_limit={float(config.cflDt):.6g}, viscosity={float(config.kViscosity):.6g}, "
          f"artificial_alpha={float(config.artificialViscosityAlpha):.6g}, "
          f"steps={max_steps}")

    for iteration in range(max_steps):
        fluid_grid.build(fluid.position, sph.h)
        boundary_grid.build(boundary.position, sph.h)
        stepWCSPH([fluid, boundary], [True, False], [fluid_grid, boundary_grid], config, sph)
        sim_time += float(config.dt)
        if args.diagnostics:
            step_max_density = float(fluid.sphDensity.numpy().max())
            if step_max_density > peak_density:
                peak_density = step_max_density
                peak_iteration = iteration + 1

        # Sample on physical frame times (0, 1/fps, ..., duration), independent
        # of the CFL-controlled solver step. This avoids duplicate end frames.
        if sim_time + 0.5 * float(config.dt) >= next_frame_time:
            positions = fluid.position.numpy()
            viewport.update(positions, sim_time)
            save_particle_frame(positions, frames_dir / f"frame_{frame:04d}.png", sim_time)
            export_ply(positions, output_dir / f"fluid_{frame:04d}.ply")
            if usd_renderer is not None:
                usd_renderer.begin_frame(sim_time)
                usd_renderer.render_points("fluid", fluid.position, radius=0.11,
                                           colors=(0.08, 0.45, 0.95))
                usd_renderer.end_frame()
            density = fluid.sphDensity.numpy()
            if args.motion_diagnostics:
                velocities = fluid.velocity.numpy()
                horizontal_speed = np.linalg.norm(velocities[:, (0, 2)], axis=1)
                speed_squared = np.sum(velocities * velocities, axis=1)
                motion_samples.append({
                    "frame": frame,
                    "time_s": sim_time,
                    "speed_rms_m_per_s": float(np.sqrt(speed_squared.mean())),
                    "horizontal_speed_rms_m_per_s": float(np.sqrt(np.mean(horizontal_speed ** 2))),
                    "horizontal_speed_p95_m_per_s": float(np.percentile(horizontal_speed, 95)),
                    "kinetic_energy_j": float(0.5 * float(particle_mass) * speed_squared.sum()),
                    "center_m": positions.mean(axis=0).tolist(),
                    "span_m": np.ptp(positions, axis=0).tolist(),
                })
            print(f"frame={frame:03d} t={sim_time:.4f}s "
                  f"rho=[{density.min():.2f}, {density.max():.2f}]")
            frame += 1
            next_frame_time += 1.0 / args.fps

    final_positions = fluid.position.numpy()
    export_ply(final_positions, output_dir / "fluid_final.ply")
    if args.diagnostics:
        final_density = fluid.sphDensity.numpy()
        diagnostics = {
            "scene": args.scene,
            "device": device,
            "steps": max_steps,
            "sim_time_s": sim_time,
            "dt_s": float(config.dt),
            "cfl_limit_s": float(config.cflDt),
            "sound_speed_m_per_s": float(config.soundSpeed),
            "kinematic_viscosity": float(config.kViscosity),
            "artificial_viscosity_alpha": float(config.artificialViscosityAlpha),
            "rest_density_kg_per_m3": float(config.restDensity),
            "peak_density_kg_per_m3": peak_density,
            "peak_density_error_percent": 100.0 * (peak_density / float(config.restDensity) - 1.0),
            "peak_iteration": peak_iteration,
            "final_density_min_kg_per_m3": float(final_density.min()),
            "final_density_max_kg_per_m3": float(final_density.max()),
            "positions_finite": bool(np.isfinite(final_positions).all()),
            "position_bounds_min": final_positions.min(axis=0).tolist(),
            "position_bounds_max": final_positions.max(axis=0).tolist(),
        }
        (output_dir / "diagnostics.json").write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"exact_peak_density={peak_density:.4f} kg/m^3 "
              f"error={diagnostics['peak_density_error_percent']:.4f}% "
              f"at_iteration={peak_iteration}")
    if args.motion_diagnostics:
        (output_dir / "motion_diagnostics.json").write_text(
            json.dumps({
                "scene": args.scene,
                "viscosity": float(config.kViscosity),
                "artificial_viscosity_alpha": float(config.artificialViscosityAlpha),
                "samples": motion_samples,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
    if usd_renderer is not None:
        usd_renderer.save()
    viewport.close()
    print(f"Finished. Results: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
