import os, sys, math, copy, StringIO, datetime, time, shutil
from System.IO import Directory
import Rhino, System
import scriptcontext as sc
import rhinoscriptsyntax as rs


ROOT_SAVE_PATH = r"C:\Users\ksteinfe\Desktop\TEMP"
PARENT_LAYER = "CAPTURE"
DEBUG = False

light_intensity = 1.0
light_color = (255,244,229) # warm white fluorecent

disp_param = {
    'TEThickness':'1', # Edge thickness
    'TSiThickness':'3', # Silhouette thickness
    'TEColor':'128,128,128', # Edge color
    'TSiColor':'0,0,0', # Silhouette color
    'CurveColor':'128,128,128', # CurveColor color
    'CurveThickness':'1', # Curve thickness 
    'TechnicalMask':6 # 14 shows mesh "edges" (described as "creases"); 6 hides mesh "edges"
}
VIEW_COUNT = 64
IMG_SIZE = 512
DO_CAPTURE_Z_BUFFER = False
#contour_curve_dist = False #False means don't contour; 0.2, 0.4 are good values for blob

XFORM_OTPS = { # always 1 identity xforms
    "do_scale_1d": True, # +3 xforms
    "do_scale_2d": True, # +3 xforms
    "do_shear": True# +3 xforms
}

def main():
    cfg = setup(PARENT_LAYER, VIEW_COUNT, IMG_SIZE, DEBUG, DO_CAPTURE_Z_BUFFER)    
    # MAIN LOOP
    print("plottting {} layers taken from {} views across {} xforms.".format( len(cfg['layer_info']['children']), len(cfg['view_locs']), len(cfg['xforms']) ) )
    print("{} images will result.".format( len(cfg['layer_info']['children']) * len(cfg['view_locs']) * len(cfg['xforms']) ) )
    try:
        for l,layer in enumerate(cfg['layer_info']['children']):
            isolate_layer(cfg['layer_info'], l)
            for x,xf in enumerate(cfg['xforms']):
                print("##### xform {} of {}".format(x+1,len(cfg['xforms'])))
                apply_xf_to_all_objects(xf) # apply transformation
                for c,cam in enumerate(cfg['view_locs']): 
                    do_capture(cam, "{}_{:04}_{:02}".format(layer.Name,c,x), cfg) # capture view
                apply_xf_to_all_objects( rs.XformInverse(xf) ) # restore transformation
                
                Rhino.RhinoApp.Wait()
                if (sc.escape_test(False)): raise Exception('Esc key caught in main()')
                    
    except Exception as e:
        print e
    finally:
        teardown(cfg)
    

def do_capture(loc, name, cfg):
    #print("view {} of {}".format(m+1,len(locs)))
    set_camera(cfg, loc)
    
    """
    contour_crv_ids = []
    if contour_curve_dist: contour_crv_ids = contour_all_meshes(loc, contour_curve_dist)
    """
    light_id = light_from_upper_left(light_intensity,light_color) 

    Rhino.RhinoApp.Wait()
    if (sc.escape_test(False)): raise Exception('Esc key caught pre-render in do_capture()')  
    
    do_view_capture(cfg, name)
    do_render(cfg, name)
    
    Rhino.RhinoApp.Wait()
    if (sc.escape_test(False)): raise Exception('Esc key caught post-render in do_capture()')     
    
    delete_light_by_id(light_id)
    # rs.DeleteObjects(contour_crv_ids)
    

def set_camera(cfg, crds):
    pos = Rhino.Geometry.Point3d(*crds)
    tar = Rhino.Geometry.Point3d(0,0,0)
    cfg['view'].ActiveViewport.SetCameraLocations(tar, pos)
    cfg['view'].ActiveViewport.ZoomExtents()
    
def do_view_capture(cfg, fname):
    #rs.Command("_-ViewCaptureToFile LockAspectRatio=No width={w} height={h} {pth}\{fname}_line.jpg".format(w=cfg['size'],h=cfg['size'],pth=cfg['pth_save'],fname=fname))
    bmp = cfg['view'].CaptureToBitmap(System.Drawing.Size(cfg['size'],cfg['size']))
    bmp.Save( os.path.join(cfg['pth_save_line'], "{}.jpg".format(fname) ) )
    if cfg['do_capture_z_buffer']:
        rs.Command("ShowZBuffer")
        bmp = cfg['view'].CaptureToBitmap(System.Drawing.Size(cfg['size'],cfg['size']))
        bmp.Save( os.path.join(cfg['pth_save_depth'], "{}.jpg".format(fname) ) )
        rs.Command("ShowZBuffer")
    
def do_render(cfg, fname):
    rs.Command("_-Render")
    rs.Command("_-SaveRenderWindowAs {}".format( os.path.join(cfg['pth_save_render'],"{}.png".format(fname) ) ) )
    rs.Command("_-CloseRenderWindow")

def setup(parent_layer, view_count, image_size, debug, do_capture_z_buffer=False):
    cfg = {}
    cfg['debug'] = debug
    cfg['do_capture_z_buffer'] = do_capture_z_buffer
    cfg['view_locs'] = fibonacci_lattice_pts(VIEW_COUNT)
    cfg['xforms'] = xforms_to_apply() # 12 xforms per view
    
    cfg['layer_info'] = get_layer_info(parent_layer)
    if not cfg['layer_info']:
        print("!!!!! ABORTING RENDER\ncould not find parent layer!")
        teardown(cfg)
        exit()
    
    if cfg['debug']: 
        cfg['view_locs'] = fibonacci_lattice_pts(1)
        cfg['xforms'] = xforms_to_apply(True)    
        cfg['layer_info']['children'] = [cfg['layer_info']['children'][0]]
    
    dir_cfg = initialize_directory(ROOT_SAVE_PATH, cfg['do_capture_z_buffer'], cfg['debug'])
    cfg.update(dir_cfg)
    
    sc.doc.Lights.Sun.Enabled = False
    sc.doc.GroundPlane.Enabled = False
    
    """
    #messing with skylight didn't get us very far. better to set manually
    if not skylight: sc.doc.Lights.Skylight.Enabled = False
    else:
        print(sc.doc.CurrentEnvironment.ForLighting.Name)
        rcxt = sc.doc.CurrentEnvironment.ForLighting.BeginChange()
    """
        
    cfg['size'] = image_size
    
    create_display_mode(cfg)
    
    x,y = 100, 200 # position of floating window relative to the screen (not Rhino)
    cfg['view'] = sc.doc.Views.Add("ksteinfe",Rhino.Display.DefinedViewportProjection.Top,System.Drawing.Rectangle(x,y,cfg['size'],cfg['size']),True)
    cfg['view'].ActiveViewport.DisplayMode = cfg['display_mode_desc']
    cfg['view'].Redraw()
    sc.doc.Views.Redraw()
    
    rset = sc.doc.RenderSettings
    cfg['render_settings'] = copy.deepcopy(rset)
    rset.ImageSize = System.Drawing.Size(cfg['size'],cfg['size'])
    rset.TransparentBackground = True
    rset.UseViewportSize = False    
    
    return cfg
    
def initialize_directory(pth_root, init_z_buf_dir, debug=False):
    print("Initializing save path: {}".format(pth_root))
    dir_cfg = {}
    if not Directory.Exists(pth_root): raise("!!! ROOT_SAVE_PATH does not exist.\nSet a valid path at the top of this script.\n{}".format(pth_root))
    
    success = False
    for char in 'abcdefghijkmnpqrstuvwxyz':
        dstmp = datetime.date.today().strftime('%y%m%d')
        if debug: dstmp = "_debug_" + dstmp
        dir_cfg['pth_save'] = os.path.join(pth_root,'{}{}_{}'.format(dstmp, char, os.path.splitext(sc.doc.Name)[0]))
        if not Directory.Exists(dir_cfg['pth_save']):
            Directory.CreateDirectory(dir_cfg['pth_save'])
            success = True
            break
            
    if not success:
        print("!!!! failed to initalize save path.\nClear out the following path by hand.\n{}".format(pth_root))
        exit()
    
    dir_cfg['pth_save_render'] = os.path.join(dir_cfg['pth_save'],'rndr')
    dir_cfg['pth_save_line'] = os.path.join(dir_cfg['pth_save'],'line')
    dir_cfg['pth_save_depth'] = os.path.join(dir_cfg['pth_save'],'dpth')
    Directory.CreateDirectory(dir_cfg['pth_save_render'])
    Directory.CreateDirectory(dir_cfg['pth_save_line'])
    if init_z_buf_dir: Directory.CreateDirectory(dir_cfg['pth_save_depth'])
    
    return dir_cfg
    
def teardown(cfg):
    print("teardown")
    cfg['view'].Close()
    sc.doc.RenderSettings = cfg['render_settings']
    if not Rhino.Display.DisplayModeDescription.DeleteDiplayMode(cfg['display_mode_desc'].Id):
       print("Temporary display mode {} was not deleted. Consider removing this yourself.".format(cfg['display_mode_desc'].EnglishName))
    
    reset_layers(cfg['layer_info'])
    return
    
def create_display_mode(cfg):
    cfg['pth_ini'] = cfg['pth_save'] + r"\temp.ini"
    f = open(cfg['pth_ini'],"w")
    f.write(disp_mode_str)
    f.close()
    
    guid = Rhino.Display.DisplayModeDescription.ImportFromFile(cfg['pth_ini'])
    dst_display_mode_desc = Rhino.Display.DisplayModeDescription.GetDisplayMode(guid)
    cfg['display_mode_desc'] = dst_display_mode_desc
    
    os.remove(cfg['pth_ini'])
    
def fibonacci_lattice_pts(cnt, rad=1.0, half_sphere=True):
    if half_sphere: cnt *= 2
    phi = ( 1.0 + math.sqrt ( 5.0 ) ) / 2.0
    i2 = [ 2*i-(cnt-1) for i in range(cnt) ]
    theta = [ 2.0*math.pi*float(i2[i])/phi for i in range(cnt) ]
    sphi = [ float(i2[i])/float(cnt) for i in range(cnt) ]
    cphi = [ math.sqrt(float(cnt+i2[i])*float(cnt-i2[i])) / float(cnt) for i in range(cnt) ]
    crds = [ (cphi[i]*math.sin(theta[i])*rad , cphi[i]*math.cos(theta[i])*rad , sphi[i]*rad ) for i in range(cnt) ]    
    if half_sphere: crds = filter(lambda pt: pt[2] >= 0, crds )
    return crds

def xforms_to_apply(do_limit=False):
    s = 1.618
    s2 = s/2.0
    ret = [Rhino.Geometry.Transform.Identity]
    if do_limit: return ret
    if XFORM_OTPS["do_scale_1d"]: 
        ret.extend([
        xf_scale((1, 1, s)),
        xf_scale((1, s, 1)),
        xf_scale((s, 1, 1))
    ])
    if XFORM_OTPS["do_scale_2d"]: 
        ret.extend([
        xf_scale((1, s, s)),
        xf_scale((s, 1, s)),
        xf_scale((s, s, 1))
    ])
    if XFORM_OTPS["do_shear"]: 
        ret.extend([
        xf_shear(vx=(1,s2,s2)),
        #xf_shear(vx=(1,s2,-s2)), 
        xf_shear(vy=(1,s2,s2)),
        #xf_shear(vy=(1,s2,-s2))
        xf_shear(vz=(s2,s2,1)),
       #xf_shear(vz=(s2,-s2,1))
    ])
        
    return ret

def xf_scale(scale=(1,1,1)):
    if 0 in scale:
        print("you want to scale to 0%?")
        exit()
    xs,ys,zs = scale
    xform = Rhino.Geometry.Transform.Scale(rs.WorldXYPlane(), xs, ys, zs)
    return xform
    
def xf_shear(vx=(1,0,0),vy=(0,1,0),vz=(0,0,1)):
    xform = Rhino.Geometry.Transform.Shear(rs.WorldXYPlane(), rs.CreateVector(vx), rs.CreateVector(vy), rs.CreateVector(vz))
    return xform    

def apply_xf_to_all_objects(xf,copy=False):
    rc = []
    for object_id in rs.AllObjects():
        #object_id = rs.coerceguid(object_id, True)
        id = sc.doc.Objects.Transform(object_id, xf, not copy)
        if id!=System.Guid.Empty: rc.append(id)
    if rc: sc.doc.Views.Redraw()

def light_from_upper_left(intensity, color):
    vec_cam = vector_of_active_view_camera()
    vec_up = rs.VectorUnitize(rs.VectorCrossProduct(rs.VectorCrossProduct(vec_cam,(0,0,1)), vec_cam))
    if vec_up is None: vec_up = rs.VectorUnitize(rs.VectorCrossProduct(rs.VectorCrossProduct(vec_cam,(0,1,0)), vec_cam))
    vec_left = rs.VectorUnitize(rs.VectorCrossProduct(vec_up,vec_cam))
    vec_light = rs.VectorAdd(vec_up,vec_left)
    
    # turn a bit toward the camera
    vec_cam = rs.VectorScale( rs.VectorUnitize(vec_cam) , -0.25 )
    vec_light = rs.VectorAdd(vec_light,vec_cam)
    
    lid = add_light(vec_light,(0,0,0), intensity, color)
    
    #print("created a light {}".format(lid))
    return lid
    
def delete_light_by_id(id):
    for n, light in enumerate(sc.doc.Lights):
        if light.Id == id:
            if sc.doc.Lights.Delete(n,False):
                #print("light deleted")
                sc.doc.Views.Redraw()
                return True     
            else:
                print("light not deleted")
                return False
    print("light not found")
    return False
            
def add_light(loc,tar,intensity=1.0,color=(255,255,255),name="script_generated"):
    start = rs.coerce3dpoint(loc, True)
    end = rs.coerce3dpoint(tar, True)
    
    light = Rhino.Geometry.Light()
    light.Name = name
    light.LightStyle = Rhino.Geometry.LightStyle.WorldDirectional
    light.Location = start
    light.Direction = end-start
    light.Intensity = intensity
    
    #b, color = Rhino.UI.Dialogs.ShowColorDialog(light.Diffuse)
    #if b: light.Diffuse = color
    light.Diffuse = System.Drawing.Color.FromArgb(*color) 
    
    index = sc.doc.Lights.Add(light)
    if index<0: raise Exception("unable to add light to LightTable")
    rc = sc.doc.Lights[index].Id
    sc.doc.Views.Redraw()
    return rc

def vector_of_active_view_camera():
    loc = sc.doc.Views.ActiveView.ActiveViewport.CameraLocation
    tar = sc.doc.Views.ActiveView.ActiveViewport.CameraTarget
    vec_cam = rs.VectorUnitize(rs.VectorCreate(tar,loc))
    return(vec_cam)

"""
# should only take meshes on a given layer
def contour_all_meshes(vec, dist):
    all_ids = []
    mshs, boxs = [],[]
    for object_id in rs.AllObjects():
        if rs.ObjectType(object_id) == 32: # 32 is a mesh
            msh = rs.coercemesh(object_id)
            bbox = msh.GetBoundingBox(False)
            mshs.append(msh)
            boxs.append(bbox)
            
    bbox = boxs[0]
    for b in boxs[1:]: bbox.Union(b)
    vec = rs.VectorScale( rs.VectorUnitize(vec) , bbox.Min.DistanceTo(bbox.Max) )
    cpt = rs.VectorAdd(rs.VectorScale(rs.VectorCreate(bbox.Max,bbox.Min),0.5),bbox.Min)
    spt, ept = rs.VectorAdd(cpt,vec), rs.VectorAdd(cpt,-vec)
            
    for msh in mshs:
        curves = Rhino.Geometry.Mesh.CreateContourCurves(msh, rs.coerce3dpoint(spt), rs.coerce3dpoint(ept), dist)
        if len(curves)==0: 
            print("no curves created")
            continue
        ids = [sc.doc.Objects.AddCurve(curve) for curve in curves ]
        all_ids.extend(ids)
    
    return all_ids
"""

def get_layer_info(root_layer_name):
    lay_root = sc.doc.Layers.FindName(root_layer_name, 0)
    if lay_root is None : return False
    ret = {"parent": lay_root, "children": []}
    if lay_root.GetChildren() is None:
        ret["children"] = [lay_root]
        return ret
    for lay_child in lay_root.GetChildren():
        if not rs.IsLayerEmpty(lay_child.Id):
            ret["children"].append( lay_child )
    
    return ret
    
def isolate_layer(linfo, idx):
    lay_active = linfo["children"][idx]
    lay_active.IsVisible = True
    sc.doc.Layers.SetCurrentLayerIndex(lay_active.Index, True)
    for lay in linfo["children"]:
        if lay_active.Index != lay.Index: lay.IsVisible = False
    
def reset_layers(linfo):
    sc.doc.Layers.SetCurrentLayerIndex(linfo["parent"].Index, True)
    for lay in linfo["children"]: lay.IsVisible = True


disp_mode_str = r"""
[DisplayMode\076c260a-43c5-4225-a8ec-ca8a3dc2b3b6]
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
Name=SCRIPT GENERATED - DELETE THIS
XrayAllObjects=n
IgnoreHighlights=n
DisableConduits=n
DisableTransparency=n
BBoxMode=0
RealtimeDisplayId=00000000-0000-0000-0000-000000000000
PipelineId=e1eb7363-87f2-4a2b-a861-256e77835369
[DisplayMode\076c260a-43c5-4225-a8ec-ca8a3dc2b3b6\View settings]
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
GroundPlaneUsage=1
CustomGroundPlaneShow=n
CustomGroundPlaneAltitude=0
CustomGroundPlaneAutomaticAltitude=y
CustomGroundPlaneShadowOnly=y
LinearWorkflowUsage=0
CustomLinearWorkflowPreProcessColors=y
CustomLinearWorkflowPreProcessTextures=y
CustomLinearWorkflowPostProcessFrameBuffer=n
CustomLinearWorkflowPreProcessGamma=2.200000047683716
CustomLinearWorkflowPostProcessGamma=2.200000047683716
FillMode=2
SolidColor=255,255,255
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
[DisplayMode\076c260a-43c5-4225-a8ec-ca8a3dc2b3b6\Shading]
CullBackfaces=n
ShadeVertexColors=n
SingleWireColor=n
WireColor=0,0,0
ShadeSurface=n
UseObjectMaterial=n
UseObjectBFMaterial=n
BakeTextures=y
ShowDecals=y
SurfaceColorWriting=y
ShadingEffect=0
ParallelLineWidth=2
ParallelLineSeparation=3
ParallelLineRotation=0
[DisplayMode\076c260a-43c5-4225-a8ec-ca8a3dc2b3b6\Shading\Material]
UseBackMaterial=n
FrontIsCustom=n
BackIsCustom=n
[DisplayMode\076c260a-43c5-4225-a8ec-ca8a3dc2b3b6\Shading\Material\Front Material]
FlatShaded=n
OverrideObjectColor=n
OverrideObjectTransparency=y
OverrideObjectReflectivity=y
Diffuse=126,126,126
Shine=128
Specular=255,255,255
Transparency=0
Reflectivity=0
ShineIntensity=100
Luminosity=0
[DisplayMode\076c260a-43c5-4225-a8ec-ca8a3dc2b3b6\Shading\Material\Back Material]
FlatShaded=n
OverrideObjectColor=n
OverrideObjectTransparency=y
OverrideObjectReflectivity=y
Diffuse=126,126,126
Shine=0
Specular=255,255,255
Transparency=0
Reflectivity=0
ShineIntensity=100
Luminosity=0
[DisplayMode\076c260a-43c5-4225-a8ec-ca8a3dc2b3b6\Lighting]
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
[DisplayMode\076c260a-43c5-4225-a8ec-ca8a3dc2b3b6\Objects]
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
[DisplayMode\076c260a-43c5-4225-a8ec-ca8a3dc2b3b6\Objects\Surfaces]
SurfaceKappaHair=n
HighlightSurfaces=n
ShowIsocurves=y
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
ShowEdges=y
ShowNakedEdges=n
ShowTangentEdges=y
ShowTangentSeams=y
ShowNonmanifoldEdges=n
ShowEdgeEndpoints=n
EdgeThickness=1
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
[DisplayMode\076c260a-43c5-4225-a8ec-ca8a3dc2b3b6\Objects\Meshes]
HighlightMeshes=n
SingleMeshWireColor=n
MeshWireColor=0,0,0
MeshWireThickness=1
MeshWirePattern=-1
ShowMeshWires=y
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
[DisplayMode\076c260a-43c5-4225-a8ec-ca8a3dc2b3b6\Objects\Curves]
ShowCurvatureHair=n
ShowCurves=y
SingleCurveColor=y
CurveColor={CurveColor}
CurveThickness={CurveThickness}
CurveTrans=0
CurvePattern=-1
LineEndCapStyle=0
LineJoinStyle=0
[DisplayMode\076c260a-43c5-4225-a8ec-ca8a3dc2b3b6\Objects\Points]
PointSize=3
PointStyle=102
ShowPoints=y
ShowPointClouds=y
PCSize=2
PCStyle=50
PCGripSize=2
PCGripStyle=102
[DisplayMode\076c260a-43c5-4225-a8ec-ca8a3dc2b3b6\Objects\Annotations]
ShowText=y
ShowAnnotations=y
DotTextColor=-1
DotBorderColor=-1
[DisplayMode\076c260a-43c5-4225-a8ec-ca8a3dc2b3b6\Objects\Technical]
TechnicalMask={TechnicalMask}
TechnicalUsageMask=6
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
""".format(**disp_param)



if __name__ == "__main__": 
    main()