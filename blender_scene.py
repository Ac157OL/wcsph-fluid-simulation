"""Build and render a fluid surface scene in Blender.

Run with:
    blender -b --python blender_scene.py -- \
      results/reconstruction/fluid_surface.ply results/blender results/fluid_final.ply
"""
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def cli_paths():
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    source = Path(args[0] if args else "results/reconstruction/fluid_surface.ply").resolve()
    output = Path(args[1] if len(args) > 1 else "results/blender").resolve()
    point_cloud = Path(args[2] if len(args) > 2 else "results/fluid_final.ply").resolve()
    output.mkdir(parents=True, exist_ok=True)
    return source, output, point_cloud


def look_at(obj, point):
    obj.rotation_euler = (Vector(point) - obj.location).to_track_quat("-Z", "Y").to_euler()


def material(name, base, metallic=0.0, roughness=0.35):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*base, 1.0)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def set_if_present(node, names, value):
    for name in names:
        if name in node.inputs:
            node.inputs[name].default_value = value
            return name
    return None


def add_area(name, location, energy, color, size, target=(-0.3, 0.0, -1.8)):
    data = bpy.data.lights.new(name, "AREA")
    data.energy, data.color, data.shape, data.size = energy, color, "DISK", size
    obj = bpy.data.objects.new(name, data); bpy.context.collection.objects.link(obj)
    obj.location = location; look_at(obj, target)
    return obj


def render(scene, path):
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def import_ply(path):
    """Import a PLY with the Blender 4.x operator and a legacy fallback."""
    try:
        bpy.ops.wm.ply_import(filepath=str(path))
    except Exception:
        bpy.ops.import_mesh.ply(filepath=str(path))
    return bpy.context.active_object


def add_point_display_modifier(obj, point_material):
    """Render imported PLY vertices as small spheres without changing source data."""
    group = bpy.data.node_groups.new("Imported PLY point display", "GeometryNodeTree")
    group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = group.nodes, group.links
    group_in = nodes.new("NodeGroupInput")
    group_out = nodes.new("NodeGroupOutput")
    ico = nodes.new("GeometryNodeMeshIcoSphere")
    ico.inputs["Radius"].default_value = 0.055
    ico.inputs["Subdivisions"].default_value = 1
    instance = nodes.new("GeometryNodeInstanceOnPoints")
    realize = nodes.new("GeometryNodeRealizeInstances")
    set_material = nodes.new("GeometryNodeSetMaterial")
    set_material.inputs["Material"].default_value = point_material
    links.new(group_in.outputs["Geometry"], instance.inputs["Points"])
    links.new(ico.outputs["Mesh"], instance.inputs["Instance"])
    links.new(instance.outputs["Instances"], realize.inputs["Geometry"])
    links.new(realize.outputs["Geometry"], set_material.inputs["Geometry"])
    links.new(set_material.outputs["Geometry"], group_out.inputs["Geometry"])
    modifier = obj.modifiers.new("Display imported particles", "NODES")
    modifier.node_group = group


def add_native_surface_reconstruction_modifier(obj):
    """Keep an auditable Blender-native points-to-volume-to-mesh pipeline."""
    group = bpy.data.node_groups.new("Blender native fluid reconstruction", "GeometryNodeTree")
    group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = group.nodes, group.links
    group_in = nodes.new("NodeGroupInput")
    group_out = nodes.new("NodeGroupOutput")
    mesh_to_points = nodes.new("GeometryNodeMeshToPoints")
    points_to_volume = nodes.new("GeometryNodePointsToVolume")
    points_to_volume.resolution_mode = "VOXEL_SIZE"
    points_to_volume.inputs["Voxel Size"].default_value = 0.07
    points_to_volume.inputs["Radius"].default_value = 0.32
    volume_to_mesh = nodes.new("GeometryNodeVolumeToMesh")
    volume_to_mesh.inputs["Threshold"].default_value = 0.12
    volume_to_mesh.inputs["Adaptivity"].default_value = 0.03
    links.new(group_in.outputs["Geometry"], mesh_to_points.inputs["Mesh"])
    links.new(mesh_to_points.outputs["Points"], points_to_volume.inputs["Points"])
    links.new(points_to_volume.outputs["Volume"], volume_to_mesh.inputs["Volume"])
    links.new(volume_to_mesh.outputs["Mesh"], group_out.inputs["Geometry"])
    modifier = obj.modifiers.new("Points to Volume to Mesh", "NODES")
    modifier.node_group = group


def main():
    source, output, point_cloud = cli_paths()
    if not source.exists():
        raise FileNotFoundError(source)
    if not point_cloud.exists():
        raise FileNotFoundError(point_cloud)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x, scene.render.resolution_y = 1100, 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.look = "AgX - Medium High Contrast"
    world = bpy.data.worlds.new("Dark studio world")
    world.color = (0.004, 0.008, 0.018)
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.004, 0.008, 0.018, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.12
    scene.world = world
    scene["description"] = "SPH surface reconstruction and water rendering"
    scene["source_ply"] = str(source)
    scene["source_point_cloud_ply"] = str(point_cloud)
    scene["reconstruction"] = "Gaussian implicit field + Marching Cubes"

    # Preserve the actual simulation PLY in the professional-software scene.
    # The object remains a vertex-only point cloud; Geometry Nodes only supplies
    # a non-destructive viewport/render representation of each imported point.
    cloud = import_ply(point_cloud)
    cloud.name = "WCSPH_Source_PointCloud_784"
    cloud.rotation_euler.x = math.radians(90.0)
    cloud["role"] = "Source particle point cloud"
    cloud["particle_count"] = len(cloud.data.vertices)
    cloud_mat = material("Imported particle cyan", (0.01, 0.34, 0.8), metallic=0.05, roughness=0.22)
    add_point_display_modifier(cloud, cloud_mat)
    native_surface = cloud.copy()
    native_surface.data = cloud.data.copy()
    native_surface.modifiers.clear()
    bpy.context.collection.objects.link(native_surface)
    native_surface.name = "Blender_Native_PointCloud_Reconstruction"
    native_surface["method"] = "Mesh to Points -> Points to Volume -> Volume to Mesh"
    add_native_surface_reconstruction_modifier(native_surface)
    # Retain this procedural reconstruction in the editable file as proof, while
    # using the fixed Marching Cubes mesh below for deterministic final rendering.
    native_surface.hide_render = True
    native_surface.hide_viewport = True

    # PLY stores simulation Y as up; rotate it into Blender's Z-up coordinates.
    water = import_ply(source)
    water.name = "WCSPH_Reconstructed_Surface"
    water.rotation_euler.x = math.radians(90.0)
    for polygon in water.data.polygons:
        polygon.use_smooth = True
    water["source_particle_count"] = 784
    water["surface_vertex_count"] = len(water.data.vertices)
    water["water_ior"] = 1.333

    bevel = water.modifiers.new("Micro bevel", "BEVEL")
    bevel.width, bevel.segments = 0.012, 2

    # Studio floor.
    bpy.ops.mesh.primitive_plane_add(size=30.0, location=(0.0, 0.0, -2.62))
    floor = bpy.context.object; floor.name = "Wet studio floor"
    floor.data.materials.append(material("Wet charcoal", (0.025, 0.045, 0.065), metallic=0.12, roughness=0.2))

    # Camera and a three-point photographic lighting rig.
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data); bpy.context.collection.objects.link(camera)
    camera.location = (6.8, -9.0, 3.8); camera_data.lens = 56
    look_at(camera, (-0.35, 0.0, -1.65)); scene.camera = camera
    add_area("Key softbox", (3.8, -4.0, 7.0), 1450, (0.76, 0.9, 1.0), 4.5)
    add_area("Fill softbox", (-5.0, -2.0, 2.8), 1050, (0.22, 0.5, 1.0), 3.5)
    add_area("Warm rim", (1.0, 4.0, 5.8), 1550, (1.0, 0.52, 0.25), 3.0)

    # First proof image: the original vertex-only simulation PLY inside Blender.
    water.hide_render = True
    cloud.hide_render = False
    render(scene, output / "blender_pointcloud_import.png")
    cloud.hide_render = True
    water.hide_render = False

    # Reconstruction preview: opaque clay with shader-level triangle edges.
    preview_mat = material("Reconstruction clay", (0.03, 0.38, 0.72), metallic=0.12, roughness=0.28)
    preview_bsdf = preview_mat.node_tree.nodes.get("Principled BSDF")
    wire_node = preview_mat.node_tree.nodes.new("ShaderNodeWireframe")
    wire_node.inputs["Size"].default_value = 0.006
    wire_node.use_pixel_size = False
    mix_node = preview_mat.node_tree.nodes.new("ShaderNodeMixRGB")
    mix_node.blend_type = "MIX"
    mix_node.inputs[1].default_value = (0.015, 0.2, 0.42, 1.0)
    mix_node.inputs[2].default_value = (0.001, 0.003, 0.008, 1.0)
    preview_mat.node_tree.links.new(wire_node.outputs["Fac"], mix_node.inputs[0])
    preview_mat.node_tree.links.new(mix_node.outputs["Color"], preview_bsdf.inputs["Base Color"])
    water.data.materials.append(preview_mat)
    render(scene, output / "blender_reconstruction_preview.png")

    # Final physically inspired water material.
    water.data.materials.clear()
    water_mat = bpy.data.materials.new("Physical water — IOR 1.333")
    water_mat.use_nodes = True
    nodes, links = water_mat.node_tree.nodes, water_mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.008, 0.11, 0.19, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.045
    bsdf.inputs["IOR"].default_value = 1.333
    transmission_socket = set_if_present(bsdf, ("Transmission Weight", "Transmission"), 0.96)
    set_if_present(bsdf, ("Coat Weight", "Coat"), 0.14)
    volume = nodes.new("ShaderNodeVolumeAbsorption")
    volume.inputs["Color"].default_value = (0.015, 0.24, 0.48, 1.0)
    volume.inputs["Density"].default_value = 0.07
    links.new(volume.outputs["Volume"], nodes.get("Material Output").inputs["Volume"])
    water_mat["ior"] = 1.333
    water_mat["transmission_socket"] = transmission_socket or "unavailable"
    water.data.materials.append(water_mat)

    # Mild glare in the compositor makes the softbox reflections legible.
    scene.use_nodes = True
    tree = scene.node_tree; tree.nodes.clear()
    layers = tree.nodes.new("CompositorNodeRLayers")
    glare = tree.nodes.new("CompositorNodeGlare")
    glare.glare_type, glare.quality, glare.threshold, glare.size = "FOG_GLOW", "HIGH", 0.9, 6
    composite = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(layers.outputs["Image"], glare.inputs["Image"])
    tree.links.new(glare.outputs["Image"], composite.inputs["Image"])
    # Use Cycles for the final frame so refraction/volume absorption are evaluated
    # by a path tracer rather than the realtime preview rasterizer.
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 256
    scene.cycles.use_denoising = False  # Debian build omits OpenImageDenoise
    scene.cycles.max_bounces = 10
    scene.cycles.transmission_bounces = 8
    render(scene, output / "blender_water_final.png")
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "wcsph_water_scene.blend"))
    print("BLENDER_TASK4_OK", len(cloud.data.vertices), len(water.data.vertices),
          len(water.data.polygons), output)


if __name__ == "__main__":
    main()
