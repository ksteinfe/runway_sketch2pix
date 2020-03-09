import os, sys, math, copy, StringIO, datetime, time, shutil, uuid, json
from System.IO import Directory
import Rhino, System
import scriptcontext as sc
import rhinoscriptsyntax as rs

DEBUG = False
DEFAULT_VIEW_COUNT = 3

DEFAULT_SAVE_PATH = r"C:\Users\ksteinfe\Desktop\TEMP"
REQUIRED_LAYERS = ["rndr", "line"]
FILL_LAYER_NAME = "fill"

DO_CAPTURE_VIEW = True

LINE_CURV_COLOR = '32,32,32'   # color of Rhino curve objects on the 'line' layer
LINE_CURV_THICK = 2         # thickness of Rhino curve objects on the 'line' layer
LINE_SILH_COLOR = '1,1,1'   # color of mesh and surface silhouettes on the 'line' layer
LINE_SILH_THICK = 4         # thickness of mesh and surface silhouettes on the 'line' layer
LINE_EDGE_COLOR = '30,30,30'# color of mesh and surface fold lines on the 'line' layer
LINE_EDGE_THICK = 2         # thickness of mesh and surface fold lines on the 'line' layer
FILL_CURV_THICK = 10        # thickness of Rhino curve objects on the 'fill' layer
FILL_COLR = '224,224,224'   # color of all fills on the 'fill' layer

MODE_NAME_STR = "SCRIPT GENERATED {} - DELETE THIS"
MODE_NAME_LINE = MODE_NAME_STR.format("LINE")
MODE_NAME_FILL = MODE_NAME_STR.format("FILL")
MODE_NAME_RNDR = "Rendered"

def main():
    cfg = setup()
    rs.CurrentView(cfg['view'].ActiveViewportID)
    # MAIN LOOP
    
    msg0 = "{} views at {} zoom levels across {} xforms of {} objects.".format( cfg['view_count'], len(cfg['obj_bbox_pads']), cfg['xform_count'], len(cfg['groups_info']) ) 
    msg1 = "{} images will result.".format( cfg['total_image_count'] ) 
    print(msg0)
    print(msg1)
    if rs.MessageBox("This script will plot {}\n{}\nThe folder {} has been created for this purpose.\nShall we proceed?".format(msg0,msg1,cfg['pth_save']), 4, "Ready to plot {} images?".format(cfg['total_image_count'])) != 6:
        teardown(cfg)
        exit()
    
    cfg['tmp_obj_ids'] = False
    cfg['rot_group'] = False
    cfg['tot_rot'] = False
    
    try:
        #raise Exception("nope")
        
        for g,group_info in enumerate(cfg['groups_info']):
            print("##### {} ({} of {})".format(group_info['name'].lower(), g+1,len(cfg['groups_info'])))
            #print("{} objects in this group".format(len(group_info['obj_ids'])))
            #set_camera(group_info['bbox'], cfg)
            isolate_group(g,cfg)
            
            deg = 360.0/cfg['view_count']
            rxf = rs.XformRotation2(deg, (0,0,1), group_info['bbox'].Center)
            
            cfg['rot_group'] = group_info
            cfg['tot_rot'] = 0
            
            for r in range(cfg['view_count']):
            
                for x,xf in enumerate( xforms_to_apply(group_info['bbox'], cfg, DEBUG) ):
                    
                    all_layers_on(cfg)
                    cfg['tmp_obj_ids'] = apply_xf(xf,group_info['obj_ids']) # apply this transformation
                    rs.RemoveObjectsFromGroup(cfg['tmp_obj_ids'],group_info['name'])
                    rs.HideGroup(group_info['name'])
                    
                    for p,pad in enumerate(cfg['obj_bbox_pads']):
                        xbbox = bbox_of_objects(cfg['tmp_obj_ids'])
                        set_camera(xbbox, pad, cfg)
                        
                        name = "{}_r{:03}_x{:02}_p{:02}_{}".format(group_info['name'].lower(),r,x,p,cfg['layer_info']['parent'].Name.lower())
                        is_first = (r==0 and x==0)
                        do_capture(name,is_first, cfg) # capture view
                    
                    isolate_group(g,cfg)
                    all_layers_on(cfg)
                    rs.DeleteObjects(cfg['tmp_obj_ids'])
                    cfg['tmp_obj_ids'] = False
                    
                    Rhino.RhinoApp.Wait()
                    if (sc.escape_test(False)): raise Exception('Esc key caught in main()')
                    
                rs.TransformObjects(group_info['obj_ids'], rxf)
                cfg['tot_rot'] += deg
            
            cfg['rot_group'] = False
            cfg['tot_rot'] = 0
        
                    
    except Exception as e:
        print("!!!! SCRIPT STOPPED !!!!")
        print e
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    finally:
        teardown(cfg)
        
def setup():
    cfg = {}
    cfg['do_capture_fill'] = True
    delete_residual_display_modes()
    
    ## properties dialog
    #
    props = [
        ("view_count",DEFAULT_VIEW_COUNT),
        ("image_size",512),
        ("zoom_padding_percent","-10,10,75"),
        ("iso (NE, NW, SE, or SW)","SE"),
        ("do_scale_1d", "y"),
        ("do_scale_2d", "y"),
        ("do_shear", "n"),
        ("render_or_capture", "capture"),
        ("worms_eye?", "n")
    ]
    prop_box_results = False
    if DEBUG: prop_box_results = [p[1] for p in props]
    if not prop_box_results:
        itms, vals = [p[0] for p in props], [p[1] for p in props]
        prop_box_results = rs.PropertyListBox(itms, vals, "Please set the following properties.", "Rhino Batch Render")
        if prop_box_results is None: exit()
    
    try:
        cfg['view_count']            = int(prop_box_results[0])
        cfg['size']                  = int(prop_box_results[1])
        cfg['obj_bbox_pads']          = [int(s)*0.01 for s in prop_box_results[2].split(',')]
        cfg['iso_select']            = str(prop_box_results[3]).lower()
        cfg["do_scale_1d"]           = str(prop_box_results[4]).lower() in ("y", "yes", "true", "t", "1")
        cfg["do_scale_2d"]           = str(prop_box_results[5]).lower() in ("y", "yes", "true", "t", "1")
        cfg["do_shear"]              = str(prop_box_results[6]).lower() in ("y", "yes", "true", "t", "1")
        cfg["do_render_via_view_cap"]= str(prop_box_results[7]).lower().startswith("c")
        cfg["do_worms_eye"]          = str(prop_box_results[8]).lower() in ("y", "yes", "true", "t", "1")
        
    except Exception as e:
        big_problem("There was a problem parsing the values given in the properties dialog.\n{}".format(e))
        
    
    ## groups to draw
    #
    setup_groups(cfg)
        
        
    ## layers
    #
    setup_layers(cfg)
    
    ## xforms
    #
    fake_xf = xforms_to_apply(cfg['groups_info'][0]['bbox'], cfg, DEBUG)
    cfg['xform_count'] = len(fake_xf)
    
    ## root path
    #
    pth_root = False
    if DEBUG: pth_root = DEFAULT_SAVE_PATH
    if not pth_root: pth_root = rs.BrowseForFolder(message="where should we save this?", title="Where to Save?")
    dir_cfg = initialize_directory(pth_root, cfg['do_capture_fill'])
    cfg.update(dir_cfg)
    
    
    ## view
    #
    cam_height = 0.75
    if cfg["do_worms_eye"]: cam_height *= -1
    poss = {"ne":(1,1,cam_height),"nw":(-1,1,cam_height),"se":(1,-1,cam_height),"sw":(-1,-1,cam_height)}
        
    if cfg["iso_select"] not in poss:
        big_problem("There was a problem with the selected isometric view.\n'{}' is not a valid selection.".format(cfg["iso_select"]))        
    cfg["iso_cam_pos"] = poss[cfg["iso_select"]]
    
    ## tidy up and save settings JSON
    #
    cfg['total_image_count'] = cfg['view_count'] * len(cfg['obj_bbox_pads']) * cfg['xform_count'] * len(cfg['groups_info'])
    with open(os.path.join(cfg['pth_save'],'settings.json'), 'w') as json_file:
        json.dump({k:"{}".format(str(v)) for k,v in cfg.items() if k != "groups_info"}, json_file, sort_keys=True, indent=4)
    
    
    ## SETUP RHINO
    #
    rs.UnselectAllObjects()
    setup_display_modes(cfg)
    setup_floating_viewport(cfg)
    setup_render_settings(cfg)
    
    # padding objects
    wbbox = False
    for gp in cfg['groups_info']:
        if not wbbox: wbbox = gp['bbox']
        else: wbbox.Union(gp['bbox'])
    
    wbbox = pad_bbox(wbbox,10)
    cfg['pad_obj_ids'] = []
    cfg['pad_obj_ids'].append(rs.AddPoint(wbbox.Min))
    cfg['pad_obj_ids'].append(rs.AddPoint(wbbox.Max))
    return cfg
    
def teardown(cfg):
    print("teardown")
    cfg['view'].Close()
    """
    activate_display_mode(cfg['view_restore']['disp_mode'], cfg)
    cfg['view'].Maximized = cfg['view_restore']['maximized']
    cfg['view'].Size = cfg['view_restore']['size']
    """
    if 'pad_obj_ids' in cfg and cfg['pad_obj_ids']: 
        print("... deleting padding objects")
        rs.DeleteObjects(cfg['pad_obj_ids'])
    
    if 'tmp_obj_ids' in cfg and cfg['tmp_obj_ids']: 
        print("... deleting xformed objects")
        rs.DeleteObjects(cfg['tmp_obj_ids'])
        
    if 'rot_group' in cfg and cfg['rot_group']:
        print("... rotating objects back to their starting position")
        rxf = rs.XformRotation2(-cfg['tot_rot'], (0,0,1), cfg['rot_group']['bbox'].Center)
        rs.TransformObjects(cfg['rot_group']['obj_ids'], rxf)
    
    show_all_groups(cfg)
    sc.doc.RenderSettings = cfg['render_settings']
    
    for mode in cfg['display_modes'].keys():
        if mode == 'rndr': continue
        if not Rhino.Display.DisplayModeDescription.DeleteDiplayMode(cfg['display_modes'][mode].Id): 
            print("Temporary display mode {} was not deleted. Consider removing this yourself.".format(cfg['display_modes'][mode].EnglishName))
    
    delete_residual_display_modes()
    return
    
def delete_residual_display_modes():
    dmds = Rhino.Display.DisplayModeDescription.GetDisplayModes()
    for dmd in dmds:
        if (MODE_NAME_LINE in dmd.EnglishName) or (MODE_NAME_FILL in dmd.EnglishName):
            print("Deleting residual display mode {}".format(dmd.EnglishName))
            if not Rhino.Display.DisplayModeDescription.DeleteDiplayMode(dmd.Id): 
                print("Residual display mode {} was not deleted. Consider removing this yourself.".format(dmd.EnglishName))
    
def initialize_directory(pth_root, init_fill_dir, debug=False):
    print("Initializing save path: {}".format(pth_root))
    dir_cfg = {}
    if not Directory.Exists(pth_root): big_problem("!!! Path does not exist.\nSelect a valid folder.\n{}".format(pth_root))
    
    filename = "unsavedfile"
    try:
        filename = os.path.splitext(sc.doc.Name)[0].lower().replace(" ","_")
    except:
        pass
    
    success = False
    for char in 'abcdefghijkmnpqrstuvwxyz':
        dstmp = datetime.date.today().strftime('%y%m%d')
        dir_cfg['pth_save'] = os.path.join(pth_root,'{}{}-{}'.format(dstmp, char, filename))
        if not Directory.Exists(dir_cfg['pth_save']):
            Directory.CreateDirectory(dir_cfg['pth_save'])
            success = True
            break
            
    if not success:
        big_problem("!!!! failed to initalize save path.\nClear out the following path by hand.\n{}".format(pth_root))
    
    dir_cfg['pth_save_rndr'] = os.path.join(dir_cfg['pth_save'],'rndr')
    dir_cfg['pth_save_line'] = os.path.join(dir_cfg['pth_save'],'line')
    dir_cfg['pth_save_fill'] = os.path.join(dir_cfg['pth_save'],'fill')
    Directory.CreateDirectory(dir_cfg['pth_save_rndr'])
    Directory.CreateDirectory(dir_cfg['pth_save_line'])
    if init_fill_dir: Directory.CreateDirectory(dir_cfg['pth_save_fill'])
    
    return dir_cfg
    
def big_problem(msg):
    rs.MessageBox(msg, 16)
    raise Exception(msg)
    
####################################################

def setup_groups(cfg):
    cfg['groups_info'] = []
    try:
        obj_guids = rs.GetObjects("Select groups of objects to draw.", group=True)
        if obj_guids is None: raise Exception("Why are you messing with me? Select some groups please.")
        name_grps = set()
        for id in obj_guids: 
            gnms = rs.ObjectGroups(id)
            if len(gnms)==0: raise Exception("One or more of the selected objects was not in a group.\nEverything must be grouped!")
            if len(gnms)>1: raise Exception("One or more of the selected objects was in more than one group.\nEverything must be in exactly one group!")
            name_grps.update(gnms)
            
        for name_grp in list(name_grps):
            ids = rs.ObjectsByGroup(name_grp)
            bbox = bbox_of_objects(ids)
            cfg['groups_info'].append( {"name":name_grp, "bbox":bbox, "obj_ids":ids} )
        
        if len(cfg['groups_info'])==0:
                raise Exception("No valid groups selected.")
        
    except Exception as e:
        big_problem("There was a problem with the selection of groups.\n{}".format(e))    

def isolate_group(idx, cfg):
    for g, group_info in enumerate(cfg['groups_info']):
        if idx==g: rs.ShowGroup(group_info['name'])
        else: rs.HideGroup(group_info['name'])
        
def show_all_groups(cfg):
    for g, group_info in enumerate(cfg['groups_info']):
        rs.ShowGroup(group_info['name'])
        
def hide_all_groups(cfg):
    for g, group_info in enumerate(cfg['groups_info']):
        rs.HideGroup(group_info['name'])
    
def select_objects(cfg):
    for obj_id in cfg['obj_guids']:
        rhobj = Rhino.RhinoDoc.ActiveDoc.Objects.Find(obj_id)
        rhobj.Select(True)

####################################################

def set_camera(bbox, pad, cfg):
    pos = Rhino.Geometry.Point3d(*cfg["iso_cam_pos"])
    tar = Rhino.Geometry.Point3d(0,0,0)
    cfg['view'].ActiveViewport.SetCameraLocations(tar, pos)
    cfg['view'].ActiveViewport.ZoomBoundingBox( pad_bbox(bbox,pad) )
    cfg['view'].Redraw()
    
def bbox_of_objects(guids, pad=0.05):
    bbox = False
    for obj_id in guids:
        rhobj = Rhino.RhinoDoc.ActiveDoc.Objects.Find(obj_id)
        bx = rhobj.Geometry.GetBoundingBox(True)
        if not bbox: bbox = bx
        else: bbox.Union(bx)
    
    bbox = pad_bbox(bbox, pad)
    return bbox

def pad_bbox(box, amt):
    bbox = Rhino.Geometry.BoundingBox(box.Min, box.Max) # copy box
    dx = (bbox.Max.X - bbox.Min.X) * amt
    dy = (bbox.Max.Y - bbox.Min.Y) * amt
    dz = (bbox.Max.Z - bbox.Min.Z) * amt
    bbox.Inflate(dx, dy, dz);
    return bbox

####################################################

def do_capture(name, is_first, cfg):    
    Rhino.RhinoApp.Wait()
    if (sc.escape_test(False)): raise Exception('Esc key caught pre-render in do_capture()')  
    
    if DO_CAPTURE_VIEW: do_view_capture(name, is_first, cfg)
    
    Rhino.RhinoApp.Wait()
    if (sc.escape_test(False)): raise Exception('Esc key caught post-render in do_capture()')     
    
def do_view_capture(fname, is_first, cfg):
    # https://discourse.mcneel.com/t/viewcapture-displayed-lineweights-bug/67610/9
    def view_cap():
        vc = Rhino.Display.ViewCapture()
        vc.Width = cfg['size']
        vc.Height = cfg['size']
        vc.ScaleScreenItems = True
        vc.DrawAxes = False
        vc.DrawGrid = False
        vc.DrawGridAxes = False
        vc.TransparentBackground = True
        vc.RealtimeRenderPasses = 0
        return vc
        
    isolate_layer_line(cfg)
    activate_display_mode(cfg['display_modes']['line'], cfg)
    bmp = cfg['view'].CaptureToBitmap(System.Drawing.Size(cfg['size'],cfg['size']))
    #bmp = view_cap().CaptureToBitmap(cfg['view'])
    bmp.MakeTransparent() # this is rotten. https://discourse.mcneel.com/t/capturetobitmap-with-transparency/4905/2
    bmp.Save( os.path.join(cfg['pth_save_line'], "{}.png".format(fname) ) )
    
    if cfg['do_capture_fill']:
        isolate_layer_fill(cfg)
        activate_display_mode(cfg['display_modes']['fill'], cfg)
        #bmp = cfg['view'].CaptureToBitmap(System.Drawing.Size(cfg['size'],cfg['size']))
        bmp = view_cap().CaptureToBitmap(cfg['view'])
        bmp.Save( os.path.join(cfg['pth_save_fill'], "{}.png".format(fname) ) )    
        
        
    isolate_layer_rndr(cfg)
    if cfg["do_render_via_view_cap"]:
        activate_display_mode(cfg['display_modes']['rndr'], cfg)
        bmp = view_cap().CaptureToBitmap(cfg['view'])
        bmp.Save( os.path.join(cfg['pth_save_rndr'], "{}.png".format(fname) ) ) 
        #if is_first: bmp.Save( os.path.join(cfg['pth_save'], "{}.png".format(fname) ) )
    else:    
        do_render(fname, cfg)

def do_render(fname, cfg):
    rs.CurrentView(cfg['view'].ActiveViewportID)
    rs.Command("_-Render")
    rs.Command('_-SaveRenderWindowAs "{}"'.format( os.path.join(cfg['pth_save_rndr'],"{}.png".format(fname) ) ) )
    rs.Command("_-CloseRenderWindow")


####################################################

def setup_layers(cfg):
    parent_layer = False
    if DEBUG: parent_layer = "DEBUG"
    if not parent_layer: parent_layer = rs.GetLayer("Please select the 'parent' layer")
    if parent_layer is None: exit()
    cfg['layer_info'] = get_layer_info(parent_layer)
    try:
        cfg['do_capture_fill'] = False
        if len(cfg['layer_info']['children'])==0: raise Exception("The selected parent layer has no sublayers. Please select a layer that has the proper sublayers defined.")
        for rlyr in REQUIRED_LAYERS:
            if rlyr not in [lyr.Name for lyr in cfg['layer_info']['children']]: raise Exception("The selected parent layer does not contain the required sublayer '{}'.".format(rlyr))
            cfg['layer_info'][rlyr] = False
            for lyr in cfg['layer_info']['children']:
                if lyr.Name==rlyr:
                    cfg['layer_info'][rlyr] = lyr
                    break
            
        if FILL_LAYER_NAME in [lyr.Name for lyr in cfg['layer_info']['children']]:
            cfg['do_capture_fill'] = True
            cfg['layer_info'][FILL_LAYER_NAME] = False
            for lyr in cfg['layer_info']['children']:
                if lyr.Name==FILL_LAYER_NAME:
                    cfg['layer_info'][FILL_LAYER_NAME] = lyr
                    break
    except Exception as e:
        big_problem("There was a problem with the selected layer.\n{}".format(e))
        
    cfg['layer_info']['parent'].IsVisible = True
    sc.doc.ActiveDoc.Layers.SetCurrentLayerIndex(cfg['layer_info']['parent'].Index, True)
    return

def get_layer_info(root_layer_name):
    lay_root = sc.doc.Layers.FindName(root_layer_name, 0)
    if lay_root is None : return False
    ret = {"parent": lay_root, "children": []}
    if lay_root.GetChildren() is not None:
        for lay_child in lay_root.GetChildren():
            if not rs.IsLayerEmpty(lay_child.Id):
                ret["children"].append( lay_child )
    
    return ret

def isolate_layer_rndr(cfg):
    cfg['layer_info']['rndr'].IsVisible = True
    cfg['layer_info']['rndr'].SetPersistentVisibility(True)  
    cfg['layer_info']['line'].IsVisible = False
    cfg['layer_info']['line'].SetPersistentVisibility(False)    
    if cfg['do_capture_fill']:
        cfg['layer_info']['fill'].IsVisible = False
        cfg['layer_info']['fill'].SetPersistentVisibility(False)

def isolate_layer_line(cfg):
    cfg['layer_info']['rndr'].IsVisible = False
    cfg['layer_info']['rndr'].SetPersistentVisibility(False)  
    cfg['layer_info']['line'].IsVisible = True
    cfg['layer_info']['line'].SetPersistentVisibility(True)    
    if cfg['do_capture_fill']:
        cfg['layer_info']['fill'].IsVisible = False
        cfg['layer_info']['fill'].SetPersistentVisibility(False)
        
def isolate_layer_fill(cfg):
    cfg['layer_info']['rndr'].IsVisible = False
    cfg['layer_info']['rndr'].SetPersistentVisibility(False)  
    cfg['layer_info']['line'].IsVisible = False
    cfg['layer_info']['line'].SetPersistentVisibility(False)    
    if cfg['do_capture_fill']:
        cfg['layer_info']['fill'].IsVisible = True
        cfg['layer_info']['fill'].SetPersistentVisibility(True)        

def all_layers_on(cfg):
    cfg['layer_info']['rndr'].IsVisible = True
    cfg['layer_info']['rndr'].SetPersistentVisibility(True)  
    cfg['layer_info']['line'].IsVisible = True
    cfg['layer_info']['line'].SetPersistentVisibility(True)    
    if cfg['do_capture_fill']:
        cfg['layer_info']['fill'].IsVisible = True
        cfg['layer_info']['fill'].SetPersistentVisibility(True) 


####################################################

def xforms_to_apply(bbox, cfg, do_limit):
    s = 1.618
    s2 = s/2.0
    ctr = bbox.Center
    ctr.Z = 0    
    
    ret = [Rhino.Geometry.Transform.Identity]
    #if do_limit: return ret
    if cfg["do_scale_1d"]: 
        ret.extend([
        xf_scale(ctr, (1, 1, s)),
        xf_scale(ctr, (1, s, 1)),
        xf_scale(ctr, (s, 1, 1))
    ])
    if cfg["do_scale_2d"]: 
        ret.extend([
        xf_scale(ctr, (1, s, s)),
        xf_scale(ctr, (s, 1, s)),
        xf_scale(ctr, (s, s, 1))
    ]),
    if cfg["do_shear"]:
        ret.extend([
        xf_shear(ctr, vz=(s2,0,1)),
        xf_shear(ctr, vz=(0,s2,1)),
        xf_shear(ctr, vz=(-s2,0,1)),
    ])
    
    
    
    return ret
    
def xf_scale(ctr, scale=(1,1,1)):
    if 0 in scale: raise Exception("Attempted to scale by 0%")
    pln = rs.MovePlane(rs.WorldXYPlane(), ctr)
    xs,ys,zs = scale
    xform = Rhino.Geometry.Transform.Scale(pln, xs, ys, zs)
    return xform
    
def xf_shear(ctr, vx=(1,0,0),vy=(0,1,0),vz=(0,0,1)):
    pln = rs.MovePlane(rs.WorldXYPlane(), ctr)
    xform = Rhino.Geometry.Transform.Shear(pln, rs.CreateVector(vx), rs.CreateVector(vy), rs.CreateVector(vz))
    return xform 
    
def apply_xf(xf,obj_ids):
    new_obj_ids = []
    for guid in obj_ids:
        obj_ref = rs.coerceguid(guid, False)
        id = sc.doc.Objects.Transform(obj_ref, xf, False)
        if id!=System.Guid.Empty: new_obj_ids.append(id)
    #if new_obj_ids: sc.doc.Views.Redraw()
    return(new_obj_ids)
    
####################################################



def setup_display_modes(cfg):
    cfg['display_modes'] = {}
        
    # RNDR
    #
    cfg['display_modes']['rndr'] = False
    for display_mode_desc in Rhino.Display.DisplayModeDescription.GetDisplayModes():
        if MODE_NAME_RNDR == display_mode_desc.EnglishName:
            cfg['display_modes']['rndr'] = display_mode_desc
        
    if not cfg['display_modes']['rndr']:
        big_problem("Could not find rendered display mode.\nPlease define a display mode named '{}'".format(MODE_NAME_RNDR))
        
        
    # LINE
    # 
    disp_param_line = {
        'Name':MODE_NAME_LINE,
        'GUID':uuid.uuid4(),
        'PipelineId':"e1eb7363-87f2-4a2b-a861-256e77835369",
        #'GroundPlaneUsage':1,
        'CurveColor':LINE_CURV_COLOR, # CurveColor color
        'CurveThickness':LINE_CURV_THICK, # Curve thickness 
        
        'TechnicalMask':46, # 14 shows mesh "edges" (described as "creases"); 6 hides mesh "edges"; 46 shows intersections?
        'TechnicalUsageMask': 6,
        
        'TEThickness':LINE_EDGE_THICK, # Edge thickness
        'TSiThickness':LINE_SILH_THICK, # Silhouette thickness
        'TEColor':LINE_EDGE_COLOR, # technical edge color
        'TSiColor':LINE_SILH_COLOR, # technical silhouette color
    }
    
    pth_ini_line = cfg['pth_save'] + r"\line.ini"
    f = open(pth_ini_line,"w")
    disp_param = disp_param_default
    disp_param.update(disp_param_line)
    f.write(disp_mode_str.format(**disp_param))
    f.close()
    
    guid = Rhino.Display.DisplayModeDescription.ImportFromFile(pth_ini_line)
    dst_display_mode_desc = Rhino.Display.DisplayModeDescription.GetDisplayMode(guid)
    cfg['display_modes']['line'] = dst_display_mode_desc
    
    # FILL
    # 
    disp_param_fill = {
        'Name':MODE_NAME_FILL,
        'GUID':uuid.uuid4(),
        'PipelineId':"952b2830-ce8a-4b4f-935a-8cd570d162c7",
        'FrontMaterialOverrideObjectColor':'y',
        
        'CurveColor':FILL_COLR, # CurveColor color
        'CurveThickness':FILL_CURV_THICK, # Curve thickness 
        
        'ShadeSurface': 'y', # y or n
        'FrontMaterialDiffuse':FILL_COLR,
        
        'SurfacesShowEdges':'n',
        'SurfacesShowTangentEdges':'n',
        'SurfacesShowTangentSeams':'n',
        'ShowMeshWires':'n'
    }
    
    pth_ini_fill = cfg['pth_save'] + r"\fill.ini"
    f = open(pth_ini_fill,"w")
    disp_param = disp_param_default
    disp_param.update(disp_param_fill)
    f.write(disp_mode_str.format(**disp_param))
    f.close()
    
    guid = Rhino.Display.DisplayModeDescription.ImportFromFile(pth_ini_fill)
    dst_display_mode_desc = Rhino.Display.DisplayModeDescription.GetDisplayMode(guid)
    cfg['display_modes']['fill'] = dst_display_mode_desc
    
    if not DEBUG:
        os.remove(pth_ini_line)
        os.remove(pth_ini_fill)
    
def setup_floating_viewport(cfg):
    x,y = 100, 200 # position of floating window relative to the screen (not Rhino)
    cfg['view'] = sc.doc.Views.Add("SCRIPT",Rhino.Display.DefinedViewportProjection.Top,System.Drawing.Rectangle(x,y,cfg['size'],cfg['size']),False)
    """
    cfg['view'] = sc.doc.Views.ActiveView
    cfg['view_restore'] = {
        "maximized" : cfg['view'].Maximized,
        "disp_mode" : cfg['view'].ActiveViewport.DisplayMode,
        "size" : cfg['view'].Size
    }
    cfg['view'].Maximized = False
    cfg['view'].Size = System.Drawing.Size(cfg['size'],cfg['size'])
    """
    set_camera(cfg['groups_info'][0]['bbox'], 0, cfg)
    isolate_group(0,cfg)
    activate_display_mode(cfg['display_modes']['rndr'], cfg)

def setup_render_settings(cfg):
    rset = sc.doc.RenderSettings
    cfg['render_settings'] = copy.deepcopy(rset)
    rset.ImageSize = System.Drawing.Size(cfg['size'],cfg['size'])
    rset.TransparentBackground = True
    rset.UseViewportSize = False      
    
def activate_display_mode(disp_mode, cfg):
    cfg['view'].ActiveViewport.DisplayMode = disp_mode
    cfg['view'].Redraw()
    sc.doc.Views.Redraw()


disp_param_default = {
    'GroundPlaneUsage':0,
    'CustomGroundPlaneShow':'y',
    'CustomGroundPlaneAltitude':0,
    'CustomGroundPlaneAutomaticAltitude':'n',
    'FillMode': 2, # background fill: 2 (solid color?) or 7 (transparent)
    'SolidColor': "255,255,255", # background fill color if solid
    'FrontMaterialOverrideObjectColor':'n',
    
    'TechnicalMask':47, # 47 is off? 
    'TechnicalUsageMask': 0, # don't know
    'TEThickness':1, # technical edge thickness
    'TSiThickness':1, # technical Silhouette thickness
    'TEColor':'128,128,128', # technical edge color
    'TSiColor':'0,0,255', # technical silhouette color    
    
    'ShadeSurface': 'n', # y or n
    'SurfacesShowEdges':'y',
    'SurfacesShowTangentEdges':'y',
    'SurfacesShowTangentSeams':'y',
    'ShowMeshWires':'n',
    'SurfacesEdgeThickness':1,
    'FrontMaterialDiffuse':'128,128,128',
}


disp_mode_str = r"""
[DisplayMode\{GUID}]
SupportsShadeCmd=y
SupportsShading=y
SupportsStereo=y
AddToMenu=y
AllowObjectAssignment=n
ShadedPipelineRequired=y
WireframePipelineRequired=y
PipelineLocked=y
Order=15
DerivedFrom=b46ab226-05a0-4568-b454-4b1ab721c675
Name={Name}
XrayAllObjects=n
IgnoreHighlights=n
DisableConduits=n
DisableTransparency=n
BBoxMode=0
RealtimeDisplayId=00000000-0000-0000-0000-000000000000
PipelineId={PipelineId}
[DisplayMode\{GUID}\View settings]
UseDocumentGrid=n
DrawGrid=n
DrawAxes=n
DrawZAxis=n
DrawWorldAxes=n
ShowGridOnTop=n
ShowTransGrid=n
BlendGrid=n
GridTrans=60
DrawTransGridPlane=n
GridPlaneTrans=90
PlaneVisibility=0
AxesPercentage=100
PlaneUsesGridColor=n
GridPlaneColor=0,0,0
WorldAxesColor=0
WxColor=150,75,75
WyColor=75,150,75
WzColor=0,0,150
GroundPlaneUsage={GroundPlaneUsage}
CustomGroundPlaneShow={CustomGroundPlaneShow}
CustomGroundPlaneAltitude={CustomGroundPlaneAltitude}
CustomGroundPlaneAutomaticAltitude={CustomGroundPlaneAutomaticAltitude}
CustomGroundPlaneShadowOnly=y
LinearWorkflowUsage=0
CustomLinearWorkflowPreProcessColors=y
CustomLinearWorkflowPreProcessTextures=y
CustomLinearWorkflowPostProcessFrameBuffer=n
CustomLinearWorkflowPreProcessGamma=2.200000047683716
CustomLinearWorkflowPostProcessGamma=2.200000047683716
FillMode={FillMode}
SolidColor={SolidColor}
GradTopLeft=200,200,200
GradBotLeft=140,140,140
GradTopRight=200,200,200
GradBotRight=140,140,140
BackgroundBitmap=
StereoModeEnabled=0
StereoSeparation=1
StereoParallax=1
AGColorMode=0
AGViewingMode=0
FlipGlasses=n
ShowClippingPlanes=n
ClippingShowXSurface=y
ClippingShowXEdges=y
ClippingClipSelected=y
ClippingShowCP=y
ClippingSurfaceUsage=0
ClippingEdgesUsage=0
ClippingCPUsage=0
ClippingCPTrans=95
ClippingEdgeThickness=3
ClippingSurfaceColor=128,128,128
ClippingEdgeColor=0,0,0
ClippingCPColor=255,255,255
HorzScale=1
VertScale=1
[DisplayMode\{GUID}\Shading]
CullBackfaces=n
ShadeVertexColors=n
SingleWireColor=n
WireColor=0,0,0
ShadeSurface={ShadeSurface}
UseObjectMaterial=n
UseObjectBFMaterial=n
BakeTextures=y
ShowDecals=y
SurfaceColorWriting=y
ShadingEffect=0
ParallelLineWidth=2
ParallelLineSeparation=3
ParallelLineRotation=0
[DisplayMode\{GUID}\Shading\Material]
UseBackMaterial=n
FrontIsCustom=n
BackIsCustom=n
[DisplayMode\{GUID}\Shading\Material\Front Material]
FlatShaded=n
OverrideObjectColor={FrontMaterialOverrideObjectColor}
OverrideObjectTransparency=y
OverrideObjectReflectivity=y
Diffuse={FrontMaterialDiffuse}
Shine=128
Specular=255,255,255
Transparency=0
Reflectivity=0
ShineIntensity=100
Luminosity=0
[DisplayMode\{GUID}\Shading\Material\Back Material]
FlatShaded=n
OverrideObjectColor=y
OverrideObjectTransparency=y
OverrideObjectReflectivity=y
Diffuse=126,126,126
Shine=0
Specular=255,255,255
Transparency=0
Reflectivity=0
ShineIntensity=100
Luminosity=0
[DisplayMode\{GUID}\Lighting]
ShowLights=n
UseHiddenLights=n
UseLightColor=n
LightingScheme=0
Luminosity=0
AmbientColor=0,0,0
LightCount=0
CastShadows=y
ShadowMapSize=2048
NumSamples=11
ShadowMapType=2
ShadowBitDepth=32
ShadowColor=0,0,0
ShadowBias=2,12,0
ShadowBlur=1
TransparencyTolerance=40
PerPixelLighting=n
ShadowClippingUsage=0
ShadowClippingRadius=0
[DisplayMode\{GUID}\Objects]
CPSolidLines=n
CPSingleColor=n
CPHidePoints=n
CPHideSurface=n
CPHighlight=y
CPHidden=n
nCPWireThickness=1
nCVSize=3
eCVStyle=102
CPColor=0,0,0
GhostLockedObjects=n
LockedTrans=50
LockedUsage=2
LockedColor=100,100,100
LockedObjectsBehind=n
LayersFollowLockUsage=n
[DisplayMode\{GUID}\Objects\Surfaces]
SurfaceKappaHair=n
HighlightSurfaces=n
ShowIsocurves=n
IsoThicknessUsed=n
IsocurveThickness=1
IsoUThickness=1
IsoVThickness=1
IsoWThickness=1
SingleIsoColor=n
IsoColor=0,0,0
IsoColorsUsed=n
IsoUColor=0,0,0
IsoVColor=0,0,0
IsoWColor=0,0,0
IsoPatternUsed=n
IsocurvePattern=-1
IsoUPattern=-1
IsoVPattern=-1
IsoWPattern=-1
ShowEdges={SurfacesShowEdges}
ShowNakedEdges=n
ShowTangentEdges={SurfacesShowTangentEdges}
ShowTangentSeams={SurfacesShowTangentSeams}
ShowNonmanifoldEdges=n
ShowEdgeEndpoints=n
EdgeThickness={SurfacesEdgeThickness}
EdgeColorUsage=0
NakedEdgeOverride=0
NakedEdgeThickness=1
NakedEdgeColorUsage=0
EdgeColorReduction=0
NakedEdgeColorReduction=0
EdgeColor=0,0,0
NakedEdgeColor=0,0,0
NonmanifoldEdgeColor=0,0,0
EdgePattern=-1
NakedEdgePattern=-1
NonmanifoldEdgePattern=-1
[DisplayMode\{GUID}\Objects\Meshes]
HighlightMeshes=n
SingleMeshWireColor=n
MeshWireColor=0,0,0
MeshWireThickness=1
MeshWirePattern=-1
ShowMeshWires={ShowMeshWires}
ShowMeshVertices=n
MeshVertexSize=0
ShowMeshEdges=n
ShowMeshNakedEdges=n
ShowMeshNonmanifoldEdges=n
MeshEdgeThickness=0
MeshNakedEdgeThickness=0
MeshNonmanifoldEdgeThickness=0
MeshEdgeColorReduction=0
MeshNakedEdgeColorReduction=0
MeshNonmanifoldEdgeColorReduction=0
MeshEdgeColor=0,0,0
MeshNakedEdgeColor=0,0,0
MeshNonmanifoldEdgeColor=0,0,0
[DisplayMode\{GUID}\Objects\Curves]
ShowCurvatureHair=n
ShowCurves=y
SingleCurveColor=y
CurveColor={CurveColor}
CurveThickness={CurveThickness}
CurveTrans=0
CurvePattern=-1
LineEndCapStyle=0
LineJoinStyle=0
[DisplayMode\{GUID}\Objects\Points]
PointSize=3
PointStyle=102
ShowPoints=n
ShowPointClouds=n
PCSize=2
PCStyle=50
PCGripSize=2
PCGripStyle=102
[DisplayMode\{GUID}\Objects\Annotations]
ShowText=n
ShowAnnotations=n
DotTextColor=-1
DotBorderColor=-1
[DisplayMode\{GUID}\Objects\Technical]
TechnicalMask={TechnicalMask}
TechnicalUsageMask={TechnicalUsageMask}
THThickness=2
TEThickness={TEThickness}
TSiThickness={TSiThickness}
TCThickness=1
TSThickness=1
TIThickness=1
THColor=0,0,0
TEColor={TEColor}
TSiColor={TSiColor}
TCColor=0,0,0
TSColor=0,0,0
TIColor=0,0,0
"""





if __name__ == "__main__": 
    main()
    
    
    
    
