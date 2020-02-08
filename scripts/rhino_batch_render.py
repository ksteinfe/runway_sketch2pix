import os, sys, math, copy, StringIO, datetime, time, shutil
from System.IO import Directory
import Rhino, System
import scriptcontext as sc
import rhinoscriptsyntax as rs

DEBUG = True

DEFAULT_SAVE_PATH = r"C:\Users\ksteinfe\Desktop\TEMP"
REQUIRED_LAYERS = ["rndr", "line"]
FILL_LAYER_NAME = "fill"


def main():
    cfg = setup()
    # MAIN LOOP
    print("plottting {} views across {} xforms.".format( len(cfg['view_locs']), len(cfg['xforms']) ) )
    print("{} images will result.".format( len(cfg['view_locs']) * len(cfg['xforms']) ) )
    try:
        #raise Exception("nope")
        
        for x,xf in enumerate(cfg['xforms']):
            print("##### xform {} of {}".format(x+1,len(cfg['xforms'])))
            #apply_xf_to_all_objects(xf) # apply transformation
            for c,cam in enumerate(cfg['view_locs']): 
                do_capture(cam, "{:04}_{:02}".format(c,x), cfg) # capture view
            #apply_xf_to_all_objects( rs.XformInverse(xf) ) # restore transformation
            
            Rhino.RhinoApp.Wait()
            if (sc.escape_test(False)): raise Exception('Esc key caught in main()')
                    
    except Exception as e:
        print e
    finally:
        teardown(cfg)
        
def setup():
    cfg = {}
    
    ## properties dialog
    #
    props = [
        ("view_count",2),
        ("image_size",512),
        ("do_scale_1d", True),
        ("do_scale_2d", True),
        ("do_shear", True)
    ]
    results = False
    if DEBUG: results = [p[1] for p in props]
    if not results:
        itms, vals = [p[0] for p in props], [p[1] for p in props]
        results = rs.PropertyListBox(itms, vals, "Please set the following properties.", "Rhino Batch Render")
        if results is None: exit()
    
    try:
        cfg["view_count"] =  int(results[0])
        cfg['size'] =  int(results[1])
        cfg["do_scale_1d"] = str(results[2]).lower() in ("y", "yes", "true", "t", "1")
        cfg["do_scale_2d"] = str(results[3]).lower() in ("y", "yes", "true", "t", "1")
        cfg["do_shear"] =    str(results[4]).lower() in ("y", "yes", "true", "t", "1")
        
    except Exception as e:
        print("There was a problem parsing the values given in the properties dialog.")
        print(e)
        exit()
    
    ## parent layer
    #
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
        if FILL_LAYER_NAME in [lyr.Name for lyr in cfg['layer_info']['children']]:
            cfg['do_capture_fill'] = True
        
    except Exception as e:
        print("There was a problem with the selected layer.")
        print(e)
        exit()
        
    
    ## root path
    #
    pth_root = False
    if DEBUG: pth_root = DEFAULT_SAVE_PATH
    if not pth_root: pth_root = rs.BrowseForFolder(message="message", title="title")
    dir_cfg = initialize_directory(pth_root, cfg['do_capture_fill'], parent_layer)
    cfg.update(dir_cfg)
    
    
    ## views
    #
    cfg['view_locs'] = fibonacci_lattice_pts(cfg["view_count"])
    
    ## xforms
    #
    cfg['xforms'] = xforms_to_apply(cfg, DEBUG)
    
    ## SETUP RHINO
    #
    setup_display_modes(cfg)
    setup_floating_viewport(cfg)
    setup_render_settings(cfg)
    
    
    return cfg
    
def teardown(cfg):
    print("teardown")
    cfg['view'].Close()
    sc.doc.RenderSettings = cfg['render_settings']
    
    
    if 'display_mode_desc_line' in cfg:
        if not Rhino.Display.DisplayModeDescription.DeleteDiplayMode(cfg['display_mode_desc_line'].Id):
           print("Temporary display mode {} was not deleted. Consider removing this yourself.".format(cfg['display_mode_desc_line'].EnglishName))
    
    if 'display_mode_desc_fill' in cfg:
        if not Rhino.Display.DisplayModeDescription.DeleteDiplayMode(cfg['display_mode_desc_fill'].Id):
           print("Temporary display mode {} was not deleted. Consider removing this yourself.".format(cfg['display_mode_desc_fill'].EnglishName))
    
    
    #reset_layers(cfg['layer_info'])
    return
    
    
####################################################

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

def get_layer_info(root_layer_name):
    lay_root = sc.doc.Layers.FindName(root_layer_name, 0)
    if lay_root is None : return False
    ret = {"parent": lay_root, "children": []}
    if lay_root.GetChildren() is not None:
        for lay_child in lay_root.GetChildren():
            if not rs.IsLayerEmpty(lay_child.Id):
                ret["children"].append( lay_child )
    
    return ret

def initialize_directory(pth_root, init_fill_dir, layername, debug=False):
    print("Initializing save path: {}".format(pth_root))
    dir_cfg = {}
    if not Directory.Exists(pth_root): raise("!!! Path does not exist.\nSelect a valid folder.\n{}".format(pth_root))
    
    filename = "unsavedfile"
    layername = layername.lower().replace(" ","_")
    try:
        filename = os.path.splitext(sc.doc.Name)[0].lower().replace(" ","_")
    except:
        pass
    
    success = False
    for char in 'abcdefghijkmnpqrstuvwxyz':
        dstmp = datetime.date.today().strftime('%y%m%d')
        dir_cfg['pth_save'] = os.path.join(pth_root,'{}{}-{}-{}'.format(dstmp, char, filename, layername))
        if not Directory.Exists(dir_cfg['pth_save']):
            Directory.CreateDirectory(dir_cfg['pth_save'])
            success = True
            break
            
    if not success:
        print("!!!! failed to initalize save path.\nClear out the following path by hand.\n{}".format(pth_root))
        exit()
    
    dir_cfg['pth_save_render'] = os.path.join(dir_cfg['pth_save'],'rndr')
    dir_cfg['pth_save_line'] = os.path.join(dir_cfg['pth_save'],'line')
    dir_cfg['pth_save_fill'] = os.path.join(dir_cfg['pth_save'],'fill')
    Directory.CreateDirectory(dir_cfg['pth_save_render'])
    Directory.CreateDirectory(dir_cfg['pth_save_line'])
    if init_fill_dir: Directory.CreateDirectory(dir_cfg['pth_save_fill'])
    
    return dir_cfg

def xforms_to_apply(cfg, do_limit):
    s = 1.618
    s2 = s/2.0
    ret = [Rhino.Geometry.Transform.Identity]
    if do_limit: return ret
    if cfg["do_scale_1d"]: 
        ret.extend([
        xf_scale((1, 1, s)),
        xf_scale((1, s, 1)),
        xf_scale((s, 1, 1))
    ])
    if cfg["do_scale_2d"]: 
        ret.extend([
        xf_scale((1, s, s)),
        xf_scale((s, 1, s)),
        xf_scale((s, s, 1))
    ])
    if cfg["do_shear"]: 
        ret.extend([
        xf_shear(vx=(1,s2,s2)),
        #xf_shear(vx=(1,s2,-s2)), 
        xf_shear(vy=(1,s2,s2)),
        #xf_shear(vy=(1,s2,-s2))
        xf_shear(vz=(s2,s2,1)),
       #xf_shear(vz=(s2,-s2,1))
    ])
        
    return ret

####################################################

def do_capture(loc, name, cfg):
    #print("view {} of {}".format(m+1,len(locs)))
    set_camera(cfg, loc)
    
    #light_id = light_from_upper_left(light_intensity,light_color) 
    
    Rhino.RhinoApp.Wait()
    if (sc.escape_test(False)): raise Exception('Esc key caught pre-render in do_capture()')  
    
    do_view_capture(cfg, name)
    #do_render(cfg, name)
    
    Rhino.RhinoApp.Wait()
    if (sc.escape_test(False)): raise Exception('Esc key caught post-render in do_capture()')     
    
    # delete_light_by_id(light_id)

def set_camera(cfg, crds):
    pos = Rhino.Geometry.Point3d(*crds)
    tar = Rhino.Geometry.Point3d(0,0,0)
    cfg['view'].ActiveViewport.SetCameraLocations(tar, pos)
    cfg['view'].ActiveViewport.ZoomExtents()

def do_view_capture(cfg, fname):
    #rs.Command("_-ViewCaptureToFile LockAspectRatio=No width={w} height={h} {pth}\{fname}_line.jpg".format(w=cfg['size'],h=cfg['size'],pth=cfg['pth_save'],fname=fname))
    bmp = cfg['view'].CaptureToBitmap(System.Drawing.Size(cfg['size'],cfg['size']))
    bmp.Save( os.path.join(cfg['pth_save_line'], "{}.jpg".format(fname) ) )

####################################################




def setup_display_modes(cfg):
    
    # LINE
    # 
    disp_param_line = {
        'Name':'SCRIPT GENERATED LINE - DELETE THIS',
        'GUID':'076c260a-43c5-4225-a8ec-ca8a3dc2b3b6',
        'TEThickness':'1', # Edge thickness
        'TSiThickness':'3', # Silhouette thickness
        'TEColor':'128,128,128', # Edge color
        'TSiColor':'0,0,0', # Silhouette color
        'CurveColor':'0,0,0', # CurveColor color
        'CurveThickness':2, # Curve thickness 
        'TechnicalMask':6, # 14 shows mesh "edges" (described as "creases"); 6 hides mesh "edges"
        'TechnicalUsageMask': 6,
        
        'FillMode': 2, # 2 or 7
        'ShadeSurface': 'y', # y or n
        'FrontMaterialDiffuse':'256,0,0', 
        'SurfacesShowEdges':'y',
        'SurfacesShowTangentEdges':'y',
        'SurfacesShowTangentSeams':'y',
        'SurfacesEdgeThickness':1,
        
        'ShowMeshWires':'y'
        
    }
    
    pth_ini_line = cfg['pth_save'] + r"\line.ini"
    f = open(pth_ini_line,"w")
    f.write(disp_mode_str.format(**disp_param_line))
    f.close()
    
    guid = Rhino.Display.DisplayModeDescription.ImportFromFile(pth_ini_line)
    dst_display_mode_desc = Rhino.Display.DisplayModeDescription.GetDisplayMode(guid)
    cfg['display_mode_desc_line'] = dst_display_mode_desc
    
    # FILL
    # 
    disp_param_fill = {
        'Name':'SCRIPT GENERATED FILL - DELETE THIS',
        'GUID':'076c260a-43c5-4225-a8ec-ca8a3dc2b3b7',
        'TEThickness':'1', # Edge thickness
        'TSiThickness':'3', # Silhouette thickness
        'TEColor':'128,128,128', # Edge color
        'TSiColor':'0,0,0', # Silhouette color
        'CurveColor':'255,0,0', # CurveColor color
        'CurveThickness':2, # Curve thickness 
        'TechnicalMask':6, # 14 shows mesh "edges" (described as "creases"); 6 hides mesh "edges"
        'TechnicalUsageMask': 6,
        
        'FillMode': 2, # 2 or 7
        'ShadeSurface': 'y', # y or n
        'FrontMaterialDiffuse':'256,0,0', 
        'SurfacesShowEdges':'y',
        'SurfacesShowTangentEdges':'y',
        'SurfacesShowTangentSeams':'y',
        'SurfacesEdgeThickness':1,
        
        'ShowMeshWires':'n'
        
    }
    
    pth_ini_fill = cfg['pth_save'] + r"\fill.ini"
    f = open(pth_ini_fill,"w")
    f.write(disp_mode_str.format(**disp_param_fill))
    f.close()
    
    guid = Rhino.Display.DisplayModeDescription.ImportFromFile(pth_ini_fill)
    dst_display_mode_desc = Rhino.Display.DisplayModeDescription.GetDisplayMode(guid)
    cfg['display_mode_desc_fill'] = dst_display_mode_desc
    
    #os.remove(pth_ini_line)
    #os.remove(pth_ini_fill)
    

def setup_floating_viewport(cfg):
    x,y = 100, 200 # position of floating window relative to the screen (not Rhino)
    cfg['view'] = sc.doc.Views.Add("ksteinfe",Rhino.Display.DefinedViewportProjection.Top,System.Drawing.Rectangle(x,y,cfg['size'],cfg['size']),True)
    cfg['view'].ActiveViewport.DisplayMode = cfg['display_mode_desc_line']
    cfg['view'].Redraw()
    sc.doc.Views.Redraw()

def setup_render_settings(cfg):
    rset = sc.doc.RenderSettings
    cfg['render_settings'] = copy.deepcopy(rset)
    rset.ImageSize = System.Drawing.Size(cfg['size'],cfg['size'])
    rset.TransparentBackground = True
    rset.UseViewportSize = False      
    



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
PipelineId=e1eb7363-87f2-4a2b-a861-256e77835369
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
FillMode={FillMode}
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
[DisplayMode\{GUID}\Shading]
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
[DisplayMode\{GUID}\Shading\Material]
UseBackMaterial=n
FrontIsCustom=n
BackIsCustom=n
[DisplayMode\{GUID}\Shading\Material\Front Material]
FlatShaded=n
OverrideObjectColor=n
OverrideObjectTransparency=y
OverrideObjectReflectivity=y
Diffuse={FrontMaterialDiffuse}
Shine=0
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
    
    
    
    
