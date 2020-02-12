import os, sys, math, copy, StringIO, datetime, time, shutil, uuid
from System.IO import Directory
import Rhino, System
import scriptcontext as sc
import rhinoscriptsyntax as rs

DEBUG = True

DEFAULT_SAVE_PATH = r"C:\Users\ksteinfe\Desktop\TEMP"

DO_CAPTURE_VIEW = True

MODE_NAME_STR = "SCRIPT GENERATED {} - DELETE THIS"
MODE_NAME_LINE = MODE_NAME_STR.format("LINE")
MODE_NAME_FILL = MODE_NAME_STR.format("FILL")

def main():
    cfg = setup()
    rs.CurrentView(cfg['view'].ActiveViewportID)
    # MAIN LOOP
    print("plottting {} views".format( cfg['view_count'] ) )
    try:
        raise Exception("nope")
                    
    except Exception as e:
        print("!!!! SCRIPT STOPPED !!!!")
        print e
    finally:
        teardown(cfg)
        
def setup():
    cfg = {}
    cfg['do_capture_fill'] = True
    delete_residual_display_modes()
    
    cfg['obj_guids'] = rs.GetObjects("Select objects to draw.")
    
    ## properties dialog
    #
    props = [
        ("view_count",2),
        ("image_size",512),
        ("zoom_padding_percent",25)
    ]
    results = False
    if DEBUG: results = [p[1] for p in props]
    if not results:
        itms, vals = [p[0] for p in props], [p[1] for p in props]
        results = rs.PropertyListBox(itms, vals, "Please set the following properties.", "Rhino Batch Render")
        if results is None: exit()
    
    try:
        cfg["view_count"]            = int(results[0])
        cfg['size']                  = int(results[1])
        cfg['obj_bbox_pad']          = int(results[2])*0.01
        
    except Exception as e:
        big_problem("There was a problem parsing the values given in the properties dialog.")
        
    
    ## bounding box
    #
    
    cfg['obj_bbox'] = bbox_of_objects(cfg['obj_guids'], cfg['obj_bbox_pad'])
    
    ## root path
    #
    pth_root = False
    if DEBUG: pth_root = DEFAULT_SAVE_PATH
    if not pth_root: pth_root = rs.BrowseForFolder(message="message", title="title")
    dir_cfg = initialize_directory(pth_root, cfg['do_capture_fill'])
    cfg.update(dir_cfg)
    
    
    cfg["axon_select"] = "se"
    poss = {"ne":(1,1,1),"nw":(-1,1,1),"se":(1,-1,1),"sw":(-1,-1,1)}
    if cfg["axon_select"] not in poss:
        big_problem("There was a problem with the selected axon.")        
    cfg["axon_cam_pos"] = poss[cfg["axon_select"]]
    
    ## SETUP RHINO
    #
    rs.UnselectAllObjects()
    setup_display_modes(cfg)
    setup_floating_viewport(cfg)
    setup_render_settings(cfg)
    
    
    return cfg
    
def teardown(cfg):
    print("teardown")
    #cfg['view'].Close()
    sc.doc.RenderSettings = cfg['render_settings']
    
    for mode in cfg['display_modes'].keys():
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
    
    dir_cfg['pth_save_render'] = os.path.join(dir_cfg['pth_save'],'rndr')
    dir_cfg['pth_save_line'] = os.path.join(dir_cfg['pth_save'],'line')
    dir_cfg['pth_save_fill'] = os.path.join(dir_cfg['pth_save'],'fill')
    Directory.CreateDirectory(dir_cfg['pth_save_render'])
    Directory.CreateDirectory(dir_cfg['pth_save_line'])
    if init_fill_dir: Directory.CreateDirectory(dir_cfg['pth_save_fill'])
    
    return dir_cfg
    
    
def big_problem(msg):
    rs.MessageBox(msg, 16)
    raise Exception(msg)
    
####################################################

def select_objects(cfg):
    for obj_id in cfg['obj_guids']:
        rhobj = Rhino.RhinoDoc.ActiveDoc.Objects.Find(obj_id)
        rhobj.Select(True)

def set_camera(cfg, crds):
    pos = Rhino.Geometry.Point3d(*crds)
    tar = Rhino.Geometry.Point3d(0,0,0)
    cfg['view'].ActiveViewport.SetCameraLocations(tar, pos)
    cfg['view'].ActiveViewport.ZoomBoundingBox(cfg['obj_bbox'])
    cfg['view'].Redraw()
    
def bbox_of_objects(guids, pad=0.25):
    bbox = False
    for obj_id in guids:
        rhobj = Rhino.RhinoDoc.ActiveDoc.Objects.Find(obj_id)
        bx = rhobj.Geometry.GetBoundingBox(True)
        if not bbox: bbox = bx
        else: bbox.Union(bx)
        
    dx = (bbox.Max.X - bbox.Min.X) * pad
    dy = (bbox.Max.Y - bbox.Min.Y) * pad
    dz = (bbox.Max.Z - bbox.Min.Z) * pad
    bbox.Inflate(dx, dy, dz);
    return bbox

####################################################


def do_capture(loc, name, cfg):    
    Rhino.RhinoApp.Wait()
    if (sc.escape_test(False)): raise Exception('Esc key caught pre-render in do_capture()')  
    
    if DO_CAPTURE_VIEW: do_view_capture(cfg, name)
    
    Rhino.RhinoApp.Wait()
    if (sc.escape_test(False)): raise Exception('Esc key caught post-render in do_capture()')     
    
    # delete_light_by_id(light_id)


def do_view_capture(cfg, fname):
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
    activate_display_mode(cfg, "line")
    bmp = cfg['view'].CaptureToBitmap(System.Drawing.Size(cfg['size'],cfg['size']))
    #bmp = view_cap().CaptureToBitmap(cfg['view'])
    bmp.MakeTransparent() # this is rotten. https://discourse.mcneel.com/t/capturetobitmap-with-transparency/4905/2
    bmp.Save( os.path.join(cfg['pth_save_line'], "{}.png".format(fname) ) )
    
    if cfg['do_capture_fill']:
        isolate_layer_fill(cfg)
        activate_display_mode(cfg, "fill")
        #bmp = cfg['view'].CaptureToBitmap(System.Drawing.Size(cfg['size'],cfg['size']))
        bmp = view_cap().CaptureToBitmap(cfg['view'])
        bmp.Save( os.path.join(cfg['pth_save_fill'], "{}.png".format(fname) ) )    

    
####################################################


def setup_display_modes(cfg):
    cfg['display_modes'] = {}
        
    # LINE
    # 
    disp_param_line = {
        'Name':MODE_NAME_LINE,
        'GUID':uuid.uuid4(),
        'PipelineId':"e1eb7363-87f2-4a2b-a861-256e77835369",
        'CurveColor':'0,0,0', # CurveColor color
        'CurveThickness':2, # Curve thickness 
        
        'TechnicalMask':14, # 14 shows mesh "edges" (described as "creases"); 6 hides mesh "edges"
        'TechnicalUsageMask': 6,
        
        'TEThickness':1, # Edge thickness
        'TSiThickness':3, # Silhouette thickness
        'TEColor':'128,128,128', # technical edge color
        'TSiColor':'1,1,1', # technical silhouette color
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
        
        'CurveColor':'224,224,224', # CurveColor color
        'CurveThickness':3, # Curve thickness 
        
        'ShadeSurface': 'y', # y or n
        'FrontMaterialDiffuse':'224,224,224',
        
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
    cfg['view'] = sc.doc.Views.Add("ksteinfe",Rhino.Display.DefinedViewportProjection.Top,System.Drawing.Rectangle(x,y,cfg['size'],cfg['size']),True)
    set_camera(cfg,cfg["axon_cam_pos"])
    activate_display_mode(cfg, "line")

def setup_render_settings(cfg):
    rset = sc.doc.RenderSettings
    cfg['render_settings'] = copy.deepcopy(rset)
    rset.ImageSize = System.Drawing.Size(cfg['size'],cfg['size'])
    rset.TransparentBackground = True
    rset.UseViewportSize = False      
    
def activate_display_mode(cfg, disp_mode_name):
    cfg['view'].ActiveViewport.DisplayMode = cfg['display_modes'][disp_mode_name]
    cfg['view'].Redraw()
    sc.doc.Views.Redraw()


disp_param_default = {
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
GroundPlaneUsage=0
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
    
    
    
    
