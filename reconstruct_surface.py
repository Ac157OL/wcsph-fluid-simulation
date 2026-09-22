"""Reconstruct an SPH surface from a PLY point cloud via Gaussian field + marching cubes."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
import trimesh


def load_ascii_ply(path):
    with open(path, "r", encoding="ascii") as stream:
        line = stream.readline().strip()
        if line != "ply":
            raise ValueError("Not a PLY file")
        vertex_count = None
        while True:
            line = stream.readline().strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            if line == "end_header":
                break
        if vertex_count is None:
            raise ValueError("PLY has no vertex element")
        return np.loadtxt(stream, dtype=np.float32, max_rows=vertex_count, usecols=(0, 1, 2))


def reconstruct(points, resolution=96):
    padding = 0.45
    lower, upper = points.min(0) - padding, points.max(0) + padding
    extent = upper - lower
    spacing = float(extent.max() / (resolution - 1))
    shape = np.ceil(extent / spacing).astype(int) + 1
    field = np.zeros(tuple(shape), dtype=np.float32)
    indices = np.rint((points - lower) / spacing).astype(int)
    np.add.at(field, tuple(indices.T), 1.0)
    nearest = cKDTree(points).query(points, k=2)[0][:, 1]
    particle_spacing = float(np.median(nearest))
    # Overlap adjacent particle kernels to obtain one coherent liquid surface.
    sigma_voxels = max(1.5, 0.8 * particle_spacing / spacing)
    field = gaussian_filter(field, sigma=sigma_voxels)
    level = float(field.max() * 0.30)
    vertices, faces, normals, _ = marching_cubes(field, level=level, spacing=(spacing,) * 3)
    vertices += lower
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals, process=True)
    trimesh.repair.fix_normals(mesh)
    return mesh


def render_water(mesh, filename):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    # Keep surface rendering responsive for dense meshes.
    stride = max(1, len(faces) // 32000)
    display_faces = faces[::stride]
    triangles = vertices[display_faces][:, :, [0, 2, 1]]
    face_normals = np.asarray(mesh.face_normals)[::stride]
    light = LightSource(azdeg=320, altdeg=42)
    intensity = np.clip(light.shade_normals(face_normals), 0, 1)
    base = np.array([0.05, 0.48, 0.92, 0.78])
    colors = np.tile(base, (len(display_faces), 1))
    diffuse = colors[:, :3] * (0.48 + 0.52 * intensity[:, None])
    specular = 0.42 * np.power(intensity[:, None], 14.0)
    colors[:, :3] = np.clip(diffuse + specular, 0.0, 1.0)

    fig = plt.figure(figsize=(10, 7), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    poly = Poly3DCollection(triangles, facecolors=colors, edgecolor="none", shade=False)
    ax.add_collection3d(poly)
    mins, maxs = triangles.reshape(-1, 3).min(0), triangles.reshape(-1, 3).max(0)
    center, radius = (mins + maxs) / 2, max((maxs - mins).max() * 0.58, 0.5)
    ax.set(xlim=(center[0]-radius, center[0]+radius),
           ylim=(center[1]-radius, center[1]+radius),
           zlim=(center[2]-radius, center[2]+radius))
    ax.set_box_aspect((1, 1, 0.82)); ax.view_init(elev=19, azim=-57)
    ax.set_axis_off(); ax.set_facecolor("#03111f"); fig.patch.set_facecolor("#03111f")
    ax.set_title("Reconstructed water surface", color="white", fontsize=15, pad=12)
    fig.tight_layout(); fig.savefig(filename, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="ASCII particle PLY")
    parser.add_argument("--output-dir", default="results/reconstruction")
    parser.add_argument("--resolution", type=int, default=96)
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    points = load_ascii_ply(args.input)
    mesh = reconstruct(points, args.resolution)
    mesh.export(out / "fluid_surface.ply")
    render_water(mesh, out / "fluid_surface_render.png")
    print(f"points={len(points)}, vertices={len(mesh.vertices)}, faces={len(mesh.faces)}")


if __name__ == "__main__":
    main()
