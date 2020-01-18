import os, random, functools, shutil, glob, math
import random
from PIL import Image, ImageChops

def load_img(filepath, do_resize=True):
    img = Image.open(filepath).convert('RGBA')
    if do_resize: img = img.resize((256, 256), Image.BICUBIC)
    return img


root = r"C:\Users\ksteinfe\Desktop\TEST\200112_pretty_mushrooms"
pth_test = "m02_0000_00"

a = load_img(os.path.join(root,"line","{}.jpg".format(pth_test)))
b = load_img(os.path.join(root,"rndr","{}.png".format(pth_test)))

def _fill_and_jiggle(line, rndr, fill_color, jig_amt=None, rot_amt=None, scl_amt=None):
    if jig_amt is None: jig_amt = 4
    if rot_amt is None: rot_amt = 2
    if scl_amt is None: scl_amt = 0.1
    jig_amt = int(jig_amt/2)
    dim = rndr.size[0]

    # jiggle rndr img
    dx, dy = random.randint(-jig_amt,jig_amt), random.randint(-jig_amt,jig_amt)
    fill = Image.new('RGBA', rndr.size, (255, 255, 255, 0))
    scl = random.uniform(1-scl_amt,1+scl_amt)
    ndim = int(dim*scl)
    rndr = rndr.resize((ndim,ndim), Image.BICUBIC)
    fill.paste(rndr, (int((dim-ndim)/2), int((dim-ndim)/2)) )

    _,_,_,alpha = fill.split()
    gr = Image.new('RGBA', fill.size, fill_color)
    wh = Image.new('RGBA', fill.size, (255, 255, 255, 255))
    fill = Image.composite(gr,wh,alpha) # an image containing a grey region the shape of the alpha channel of rndr

    # rotate line img
    rot_amt = random.randint(-rot_amt,rot_amt)
    line = line.rotate( rot_amt , Image.BICUBIC, fillcolor='white' )

    # jiggle line img
    dx, dy = random.randint(-jig_amt,jig_amt), random.randint(-jig_amt,jig_amt)
    back = Image.new('RGBA', line.size, (255, 255, 255, 255))
    back.paste(line, (dx,dy))

    return ImageChops.darker(back, fill)

for n in range(10):
    a = _fill_and_jiggle(a,b, (200, 200, 200, 255) )
    a.save(os.path.join(root,"test_{}.png".format(n)))
