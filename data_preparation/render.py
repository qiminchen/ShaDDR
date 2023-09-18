import bpy, sys, os
from math import pi

parser = argparse.ArgumentParser()
parser.add_argument("class_id", type=str, help="shapenet category id")
parser.add_argument("share_id", type=int, help="id of the share [0]")
parser.add_argument("share_total", type=int, help="total num of shares [1]")
FLAGS = parser.parse_args()

class_id = FLAGS.class_id
share_id = FLAGS.share_id
share_total = FLAGS.share_total

root_dir = './' + class_id
if not os.path.exists(root_dir):
    print("ERROR: this dir does not exist: " + root_dir)
    exit(-1)

obj_dir_set = os.listdir(root_dir)

obj_dir_set_ = []
for i in range(len(obj_dir_set)):
    if i % share_total == share_id:
        obj_dir_set_.append(obj_dir_set[i])
obj_dir_set = obj_dir_set_

for i, obj_dir in enumerate(obj_dir_set):

    print(i, obj_dir)
    
    fout = open("current_rendering" + str(share_id) + ".txt", 'w')
    fout.write(obj_dir)
    fout.close()

    #remove
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    for this_obj in bpy.data.objects:
        if this_obj.type=="MESH":
            this_obj.select_set(True)
            bpy.ops.object.delete(use_global=False, confirm=False)

    file_loc = root_dir + '/' + obj_dir + '/model_normalized.obj'
    imported_object = bpy.ops.import_scene.obj(filepath=file_loc, use_edges=False, use_smooth_groups=False)

    #load
    for this_obj in bpy.data.objects:
        if this_obj.type=="MESH":
            this_obj.select_set(True)
            bpy.context.view_layer.objects.active = this_obj
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.split_normals()

    bpy.ops.object.mode_set(mode='OBJECT')

    bpy.context.scene.render.engine = 'CYCLES'

    # render

    # top - 1
    cam = bpy.data.objects['Camera']
    cam.location.x = 0
    cam.location.y = 0
    cam.location.z = 1
    cam.rotation_euler[0] = 0
    cam.rotation_euler[1] = 0
    cam.rotation_euler[2] = 0
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.filepath = root_dir + '/' + obj_dir + '/0.png'
    bpy.ops.render.render(write_still=True)
    # break

    # down - 1
    cam = bpy.data.objects['Camera']
    cam.location.x = 0
    cam.location.y = 0
    cam.location.z = -1
    cam.rotation_euler[0] = pi
    cam.rotation_euler[1] = 0
    cam.rotation_euler[2] = 0
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.filepath = root_dir + '/' + obj_dir + '/1.png'
    bpy.ops.render.render(write_still=True)

    # left - 1
    cam = bpy.data.objects['Camera']
    cam.location.x = 0
    cam.location.y = -1
    cam.location.z = 0
    cam.rotation_euler[0] = 0.5*pi
    cam.rotation_euler[1] = 0
    cam.rotation_euler[2] = 0
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.filepath = root_dir + '/' + obj_dir + '/2.png'
    bpy.ops.render.render(write_still=True)

    # right - 1
    cam = bpy.data.objects['Camera']
    cam.location.x = 0
    cam.location.y = 1
    cam.location.z = 0
    cam.rotation_euler[0] = 0.5*pi
    cam.rotation_euler[1] = 0
    cam.rotation_euler[2] = pi
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.filepath = root_dir + '/' + obj_dir + '/3.png'
    bpy.ops.render.render(write_still=True)

    # front - 1
    cam = bpy.data.objects['Camera']
    cam.location.x = 1
    cam.location.y = 0
    cam.location.z = 0
    cam.rotation_euler[0] = 0.5*pi
    cam.rotation_euler[1] = 0
    cam.rotation_euler[2] = 0.5*pi
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.filepath = root_dir + '/' + obj_dir + '/4.png'
    bpy.ops.render.render(write_still=True)

    # back - 1
    cam = bpy.data.objects['Camera']
    cam.location.x = -1
    cam.location.y = 0
    cam.location.z = 0
    cam.rotation_euler[0] = 0.5*pi
    cam.rotation_euler[1] = 0
    cam.rotation_euler[2] = -0.5*pi
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.filepath = root_dir + '/' + obj_dir + '/5.png'
    bpy.ops.render.render(write_still=True)
