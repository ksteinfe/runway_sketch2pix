import math, os, pathlib, sys
import bpy, mathutils
from mathutils import Vector


PTH_OUT = r"C:\Users\ksteinfe\Desktop\TEMP"
pathlib.Path(PTH_OUT).mkdir(parents=False, exist_ok=True)


VIEW_COUNT = 512
IMG_SIZE = 512
HALF_SPHERE = True

def main(ctx):
    # get the current scene
    scn = ctx.scene
    #cam = scn.camera

    cfg = setup(scn)
    print(cfg)    
    
    pts = fibonacci_lattice_pts(VIEW_COUNT, half_sphere=HALF_SPHERE)
    
    for name_msh in cfg['names_msh']:
        msh = activate_mesh(name_msh, scn, cfg)
        loc_msh = msh.location.copy()
        
        name_msh_nice = name_msh.lower().replace('.','_').replace(' ','_')
        
        print("=============================\n{}".format(name_msh_nice))
        #print( "cam location is {}".format(scn.camera.location) )
        #print( "mesh location is {}".format(loc_msh) )
        
        for n,pt in enumerate(pts):
            print( "{} of {}".format(n,len(pts)) )
            print("---")
            
            for name_vlr in cfg['names_vlr']:
                activate_view_layer(name_vlr, scn, cfg)
                
                set_camera(pt, scn, cfg)
                set_light(loc_msh, scn, cfg) # set light after camera
                
                fname = "{}-{:03d}_{}.jpg".format(name_msh_nice, n,name_vlr)
                pth = os.path.join(cfg['pths_out'][name_vlr], fname )
                
                render_layer(pth)
        
        
    teardown(scn,cfg)
    
    
def setup(scn):
    cfg = {}
    
    # set object seleciton mode
    bpy.ops.object.mode_set(mode="OBJECT")
    
    # set render resolution
    scn.render.resolution_x = IMG_SIZE
    scn.render.resolution_y = IMG_SIZE
    
    # lens length
    scn.camera.data.lens = 80    
    
    # find all view layers
    cfg['names_vlr'] = [layer.name for layer in scn.view_layers]
    # turn off all view layers
    for layer in scn.view_layers: layer.use = False
    
    # create subdirs
    cfg['pths_out'] = {}
    for name_vlr in cfg['names_vlr']:
        cfg['pths_out'][name_vlr] = os.path.join(PTH_OUT, name_vlr)
        pathlib.Path(cfg['pths_out'][name_vlr]).mkdir(parents=False, exist_ok=True)
    
    
    # find mesh objects
    cfg['names_msh'] = [ob.name for ob in scn.objects if ob.type == "MESH"]
    cfg['names_msh'].sort()
    # hide all mesh object from render
    for ob in scn.objects:
        if ob.type == "MESH": ob.hide_render = True
    
    # find first light
    cfg['light'] = False
    for ob in scn.objects:
        if ob.type == "LIGHT": 
            cfg['light'] = ob
            continue
    if not cfg['light']: raise Exception("no light found.")
    
    
    return cfg
        
def teardown(scn, cfg):
    print("teardown")
    # turn on all view layers
    for layer in scn.view_layers: layer.use = True
    
    # display all mesh objects in render
    for ob in scn.objects:
        if ob.type == "MESH": ob.hide_render = False


def activate_view_layer(name, scn, cfg):
    #print("activate_layer {}".format(name))
    for layer in scn.view_layers: 
        if layer.name == name: layer.use = True
        else: layer.use = False
    
def activate_mesh(name, scn, cfg):
    #print("activate_mesh {}".format(name))
    ret = False
    for ob in scn.objects:
        if ob.type == "MESH": 
            if ob.name == name:
                ob.hide_render = False
                ob.select_set(state=True)
                ret = ob
            else:
                ob.hide_render = True
                ob.select_set(state=False)
    
    if not ret: raise Exception("no mesh with name {} found.".format(name))
    return ret

def set_light(loc_msh, scn, cfg):
    loc_cam = scn.camera.location
    
    #print( "cam {}".format(scn.camera.location) )
    #print( "msh {}".format(loc_msh) )
    
    vec_z = loc_msh - loc_cam # vec in direction of camera
    dist = vec_z.length # use dist from cam to msh as base distance
    vec_z.normalize()
    vec_z *= dist *0.5
    
    #print("dist {}".format(dist))
    
    vec_x = vec_z.cross(Vector((0,0,1))) # vec to left of screen
    vec_x.normalize()
    vec_x *= -dist *0.5
    
    vec_y = vec_z.cross(vec_x) # vec to top of screen
    vec_y.normalize()
    vec_y *= dist *0.5
    
    loc_lgt = loc_cam.copy() # start at camera location
    loc_lgt += vec_z # move away from cam
    loc_lgt += vec_x # move to the left
    loc_lgt += vec_y # move to the top
    
    cfg['light'].location = loc_lgt


def set_camera(pos, scn, cfg):
    cam = scn.camera
    cam.location = pos
    
    looking_direction = cam.location - mathutils.Vector((0,0,0))
    rot_quat = looking_direction.to_track_quat('Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()
    
    
    cam.data.lens = cam.data.lens + 5
    bpy.ops.view3d.camera_to_view_selected()
    cam.data.lens = cam.data.lens - 5


def render_layer(pth_save):
    bpy.context.scene.render.filepath = pth_save
    
    # redirect output to log file
    logfile = os.path.join(PTH_OUT,"log.txt")
    open(logfile, 'a').close()
    old = os.dup(1)
    sys.stdout.flush()
    os.close(1)
    os.open(logfile, os.O_WRONLY)

    # do the rendering
    bpy.ops.render.render(write_still=True)

    # disable output redirection
    os.close(1)
    os.dup(old)
    os.close(old)
    
    

def fibonacci_lattice_pts(cnt, rad=1.0, half_sphere=True):
    if half_sphere: cnt *= 2
    phi = ( 1.0 + math.sqrt ( 5.0 ) ) / 2.0
    i2 = [ 2*i-(cnt-1) for i in range(cnt) ]
    theta = [ 2.0*math.pi*float(i2[i])/phi for i in range(cnt) ]
    sphi = [ float(i2[i])/float(cnt) for i in range(cnt) ]
    cphi = [ math.sqrt(float(cnt+i2[i])*float(cnt-i2[i])) / float(cnt) for i in range(cnt) ]
    crds = [ (cphi[i]*math.sin(theta[i])*rad , cphi[i]*math.cos(theta[i])*rad , sphi[i]*rad ) for i in range(cnt) ]    
    if half_sphere: crds = filter(lambda pt: pt[2] >= 0, crds )
    return list(crds)


main(bpy.context)