"""blender -b -P tools/blender_glb_check.py -- FILE.glb : import a GLB headless and print mesh counts (user-level check)."""
import sys
import bpy

path = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=path)
for o in bpy.data.objects:
    if o.type == "MESH":
        m = o.data
        print(f"BLENDER_MESH {o.name} verts={len(m.vertices)} faces={len(m.polygons)} color_attrs={len(m.color_attributes)}")
