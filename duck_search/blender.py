"""Build an editable .blend and original STL workshop props from room.json.

Run with the separate Python 3.11 bpy environment; no training packages needed.
Robot meshes are CC BY-NC-SA 4.0, retained in their own named collection.
"""

import json

import bpy
from mathutils import Matrix, Vector

from .scene import COLORS, ROOT


def material(name, color, roughness=0.5, metallic=0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    node = mat.node_tree.nodes.get("Principled BSDF")
    node.inputs["Base Color"].default_value = color
    node.inputs["Roughness"].default_value = roughness
    node.inputs["Metallic"].default_value = metallic
    return mat


def point(obj, target):
    obj.rotation_euler = (
        (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()
    )


def export():
    scene = json.loads((ROOT / "assets/room.json").read_text())
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    sc.unit_settings.scale_length = 1
    original = bpy.data.collections.new("Original room assets — Apache-2.0")
    sc.collection.children.link(original)
    robot_col = bpy.data.collections.new("Pollen robot — CC BY-NC-SA 4.0")
    sc.collection.children.link(robot_col)
    mats = {}
    stldir = ROOT / "assets/stl"
    stldir.mkdir(exist_ok=True)
    for part in scene["parts"]:
        name = part["name"]
        color = tuple(part["color"])
        if color not in mats:
            mats[color] = material(
                f"Surface {len(mats)}", color, metallic=0.3 if "latch" in name else 0
            )
        bpy.ops.mesh.primitive_cube_add(size=1, location=part["pos"])
        obj = bpy.context.object
        obj.name = name
        obj.dimensions = part["size"]
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        for collection in list(obj.users_collection):
            collection.objects.unlink(obj)
        original.objects.link(obj)
        obj.data.materials.append(mats[color])
        bevel = obj.modifiers.new("Manufactured edge radius", "BEVEL")
        bevel.width = min(0.012, min(part["size"]) * 0.12)
        bevel.segments = 3
        obj.modifiers.new("Weighted normals", "WEIGHTED_NORMAL")
        obj["collision_enabled"] = part["collision"]
        obj["full_dimensions_metres"] = part["size"]
        obj["source"] = "assets/room.json"
        if part["category"] not in ["wall", "floor"] and part["collision"]:
            # Export each prop centred on its own origin, in metres, and restore
            # the scene position. README specifies print scaling explicitly.
            loc = obj.location.copy()
            obj.location = (0, 0, 0)
            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.wm.stl_export(
                filepath=str(stldir / f"{name}.stl"), export_selected_objects=True
            )
            obj.location = loc
    # Add non-colliding visual floor tile seams, matching room floor extents.
    seam = material("Floor grout", (0.20, 0.23, 0.25, 1))
    for axis in [0, 1]:
        for x in range(-4, 5):
            pos = [0, 0, 0.0007]
            pos[axis] = x * 0.5
            size = [5.0, 4.0, 0.001]
            size[axis] = 0.005
            bpy.ops.mesh.primitive_cube_add(size=1, location=pos)
            obj = bpy.context.object
            obj.name = f"Visual floor seam {axis}-{x}"
            obj.dimensions = size
            obj.data.materials.append(seam)
    for i, (x, y) in enumerate(scene["targets"]):
        if i == 0:
            bpy.ops.mesh.primitive_uv_sphere_add(
                segments=32, ring_count=16, radius=0.045, location=(x, y, 0.045)
            )
        elif i == 1:
            bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, 0.04))
            bpy.context.object.dimensions = (0.10, 0.07, 0.08)
        else:
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=32, radius=0.035, depth=0.11, location=(x, y, 0.055)
            )
        obj = bpy.context.object
        obj.name = f"Search target {i}"
        obj.data.materials.append(material(f"Target {i}", COLORS[i], 0.3))
    robotpath = ROOT / "assets/robot_blender_meshes.json"
    if robotpath.exists():
        for k, p in enumerate(json.loads(robotpath.read_text())):
            mesh = bpy.data.meshes.new(p["name"])
            mesh.from_pydata(p["vertices"], [], p["faces"])
            mesh.update()
            obj = bpy.data.objects.new(p["name"], mesh)
            robot_col.objects.link(obj)
            transform = Matrix(p["rotation"]).to_4x4()
            transform.translation = Vector(p["position"])
            obj.matrix_world = transform
            obj.data.materials.append(
                material(f"Robot material {k}", tuple(p["rgba"]), 0.4, 0.15)
            )
            obj["attribution"] = "Pollen Robotics / microduck_rl; CC BY-NC-SA 4.0"
    bpy.ops.object.camera_add(location=(0.4, -7.0, 6.0))
    cam = bpy.context.object
    cam.name = "Workshop overview"
    point(cam, (0, 0, 0.1))
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 7.6
    sc.camera = cam
    for name, loc, power, size in [
        ("Key", (0, -1, 5), 500, 5),
        ("Fill", (-3, 1, 4), 250, 4),
        ("Rim", (3, 3, 4), 350, 3),
    ]:
        bpy.ops.object.light_add(type="AREA", location=loc)
        lamp = bpy.context.object
        lamp.name = name
        lamp.data.energy = power
        lamp.data.shape = "DISK"
        lamp.data.size = size
        point(lamp, (0, 0, 0))
    sc.world.color = (0.25, 0.30, 0.36)
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 24
    sc.cycles.use_denoising = True
    sc.render.resolution_x = 1400
    sc.render.resolution_y = 1000
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.filepath = str(ROOT / "outputs/blender_room.png")
    sc["description"] = (
        "MicroDuck object search workshop. Original room assets; robot meshes have a separate noncommercial licence."
    )
    sc["physics_source"] = "assets/room.xml; Blender beauty modifiers are visual only"
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT / "assets/workshop_search.blend"))
    bpy.ops.render.render(write_still=True)
    print("BLENDER_DONE", sc.render.filepath, flush=True)
