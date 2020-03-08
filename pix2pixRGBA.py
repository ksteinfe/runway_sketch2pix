import os, random, functools, shutil, glob, math

from PIL import Image, ImageChops
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms

version_number = 0.0
print('pix2pix r{} has loaded.'.format(version_number))

class ImgUtil():
    whitelist_ext = ['jpg', 'jpeg', 'png'] # referenced in data preparation notebook
    @staticmethod
    def verify_ext(fname):
        return any([fname.lower().endswith(ext) for ext in ImgUtil.whitelist_ext ])

    @staticmethod
    def find_corrupt_images(pth):
        bad_ones = []
        fnames = [f for f in os.listdir(pth) if ImgUtil.verify_ext(f)]
        for n,fname in enumerate( fnames ):
            if len(fnames)>2000 and n%500==0 and n>0: print("...checked {} of {} images.".format(n,len(fnames)))
            try:
                img = ImgUtil.load_img(os.path.join(pth, fname), False, False)
                transforms.ToTensor()(img)
            except:
                bad_ones.append(fname)
        return(bad_ones)

    @staticmethod
    def load_img(filepath, do_resize, do_flatten):
        img = Image.open(filepath)
        if img.mode == "RGBA" and do_flatten:
            img = ImgUtil.img_alpha_to_color(img)

        img = img.convert('RGBA')
        if do_resize: img = img.resize((Pix2Pix256RGBA.IMG_SIZE, Pix2Pix256RGBA.IMG_SIZE), Image.BICUBIC)
        return img

    @staticmethod
    def img_alpha_to_color(image, color=(255, 255, 255)):
        """Alpha composited an RGBA Image with a specified color. Source: http://stackoverflow.com/a/9459208/284318 """
        image.load()  # needed for split()
        background = Image.new('RGB', image.size, color)
        background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        return background

    @staticmethod
    def imgten_to_imgpil(imgten):
        imgnum = imgten.float().numpy()
        imgnum = (np.transpose(imgnum, (1, 2, 0)) + 1) / 2.0 * 255.0
        imgnum = imgnum.clip(0, 255)
        imgnum = imgnum.astype(np.uint8)
        imgpil = Image.fromarray(imgnum)
        return(imgpil)

    @staticmethod
    def ensure_imgpil(img):
        try:
            return ImgUtil.imgten_to_imgpil(img)
        except AttributeError as e:
            return img

    # can take PIL images or tensors, or a bit of both
    @staticmethod
    def plot_imgtens(img_a, img_b, img_c=None, pth_save=None):
        # TODO save plot
        plt.rcParams.update({'font.size': 8})
        imgs = [ImgUtil.ensure_imgpil(img_a), ImgUtil.ensure_imgpil(img_b)]
        w,h = imgs[0].size
        dpi = 100
        title = None
        figsize=(w*2/dpi+1.5,h/dpi+1.0)
        if img_c is not None:
            imgs.append( ImgUtil.ensure_imgpil(img_c) )
            title = ['given', 'target', 'generated']
            figsize=(w*3/dpi+2.0,h/dpi+1.0)

        figure = plt.figure(figsize=figsize, dpi=dpi, facecolor=(0.9,0.9,0.9))
        for i in range(len(imgs)):
            axes = figure.add_subplot(1, len(imgs), i+1)
            if title is not None: axes.set_title(title[i])
            axes.imshow(imgs[i])
            axes.axis('off')

        if pth_save is not None: plt.savefig(pth_save) # save before showing
        plt.show()

class Pix2PixDataset(data.Dataset):
    FILL_COLOR = (224, 224, 224, 255)
    JIG_AMT_DEFAULT = 8 # jiggle amount (at raw scale of images)
    ROT_AMT_DEFAULT = 2 # rotation amount (at raw scale of images)
    SCL_AMT_DEFAULT = 0.1 # scale amount (at raw scale of images)
    JIT_AMT = 60 # jitter amount (at raw scale of images)

    RAW_SZE = 512 # should be twice Pix2Pix256RGBA.IMG_SIZE
    TRN_SZE = 256 # should be same as Pix2Pix256RGBA.IMG_SIZE

    def __init__(self, extraction_rslt, direction="a2b"):
        super(Pix2PixDataset, self).__init__()
        self.direction = direction
        self.is_compound = extraction_rslt['is_compound']
        self.pths = extraction_rslt['pths']

        if extraction_rslt['img_size'] != Pix2PixDataset.RAW_SZE: raise Exception("!!!! Training set is not compatable with this version.\nRaw training image size of {} was expected, but {} was found.".format(Pix2PixDataset.RAW_SZE,extraction_rslt['img_size']))
        if Pix2PixDataset.RAW_SZE != 2*Pix2PixDataset.TRN_SZE: raise Exception("!!!! Raw training image size of {} is not twice as big as {}".format(Pix2PixDataset.RAW_SZE,Pix2PixDataset.TRN_SZE))

    @staticmethod
    def tform():
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))]

        return( transforms.Compose(transform_list) )

    """
    Combines given a0 and a1 (line and fill) images to a single a image.
    Overlays line on top of fill, with a bit of random jiggle, rotation, and (no) scaling
    Only used in cases where fills are given.
    """
    @staticmethod
    def _filled_a_from_composite(img_a0,img_a1, jig_amt=None, rot_amt=None, scl_amt=None):
        if jig_amt is None: jig_amt = Pix2PixDataset.JIG_AMT_DEFAULT
        if rot_amt is None: rot_amt = Pix2PixDataset.ROT_AMT_DEFAULT
        # if scl_amt is None: scl_amt = Pix2PixDataset.SCL_AMT_DEFAULT # scaling not used for now
        jig_amt = int(jig_amt/2)

        img_a0 = ImgUtil.img_alpha_to_color(img_a0).convert('RGBA')
        img_a1 = ImgUtil.img_alpha_to_color(img_a1).convert('RGBA')

        # rotate
        # do not rotate line image?
        #img_a0 = img_a0.rotate( random.randint(-rot_amt,rot_amt) , Image.BICUBIC, fillcolor='white' )
        img_a1 = img_a1.rotate( random.randint(-rot_amt,rot_amt) , Image.BICUBIC, fillcolor='white' )

        # jiggle
        img_a0n = Image.new('RGBA', img_a0.size, (255, 255, 255, 255))
        img_a0n.paste(img_a0, (random.randint(-jig_amt,jig_amt), random.randint(-jig_amt,jig_amt)))
        img_a1n = Image.new('RGBA', img_a1.size, (255, 255, 255, 255))
        img_a1n.paste(img_a1, (random.randint(-jig_amt,jig_amt), random.randint(-jig_amt,jig_amt)))

        return ImageChops.darker(img_a0n, img_a1n)

    """
    Returns an filled line image by extracting the alpha channel of the given b image
    Only used in cases in which fills are desired but are no fill images are present in the given dataset
    """
    @staticmethod
    def _filled_a_from_b_alpha(img_a, img_b, fill_color, jig_amt=None, rot_amt=None, scl_amt=None):
        if jig_amt is None: jig_amt = Pix2PixDataset.JIG_AMT_DEFAULT
        if rot_amt is None: rot_amt = Pix2PixDataset.ROT_AMT_DEFAULT
        if scl_amt is None: scl_amt = Pix2PixDataset.SCL_AMT_DEFAULT
        jig_amt = int(jig_amt/2)
        dim = img_b.size[0]

        # jiggle img_b img
        dx, dy = random.randint(-jig_amt,jig_amt), random.randint(-jig_amt,jig_amt)
        fill = Image.new('RGBA', img_b.size, (255, 255, 255, 0))
        scl = random.uniform(1-scl_amt,1+scl_amt)
        ndim = int(dim*scl)
        img_b = img_b.resize((ndim,ndim), Image.BICUBIC)
        fill.paste(img_b, (int((dim-ndim)/2), int((dim-ndim)/2)) )

        _,_,_,alpha = fill.split()
        gr = Image.new('RGBA', fill.size, fill_color)
        wh = Image.new('RGBA', fill.size, (255, 255, 255, 255))
        fill = Image.composite(gr,wh,alpha) # an image containing a grey region the shape of the alpha channel of img_b

        # rotate img_a
        rot_amt = random.randint(-rot_amt,rot_amt)
        img_a = img_a.rotate( rot_amt , Image.BICUBIC, fillcolor='white' )

        # jiggle img_a
        dx, dy = random.randint(-jig_amt,jig_amt), random.randint(-jig_amt,jig_amt)
        back = Image.new('RGBA', img_a.size, (255, 255, 255, 255))
        back.paste(img_a, (dx,dy))

        return ImageChops.darker(back, fill)

    def __getitem__(self, index): return self._do_get_item(index)

    def _do_get_item(self, index, forced_resize_method=-1):
        def resize(img, sz=Pix2PixDataset.TRN_SZE): return img.resize((sz, sz), Image.BICUBIC)

        img_b = ImgUtil.load_img(self.pths[index]['b'], do_resize=False, do_flatten=False)

        if self.is_compound:
            img_a0 = ImgUtil.load_img(self.pths[index]['a0'], do_resize=False, do_flatten=False)
            img_a1 = ImgUtil.load_img(self.pths[index]['a1'], do_resize=False, do_flatten=False)
            img_a = Pix2PixDataset._filled_a_from_composite(img_a0,img_a1)
        else:
            img_a = ImgUtil.load_img(self.pths[index]['a0'], do_resize=False, do_flatten=True)
            img_a = Pix2PixDataset._filled_a_from_b_alpha(img_a,img_b, Pix2PixDataset.FILL_COLOR )

        # here, images are 'jittered', shifted around at raw scale to mess with the centering
        # at the same time, images are resized to training scale. this may happen via:
        # cropping down without rescaling, thereby cropping out much of the image
        # by scaling from raw to training scale (0.5), thereby keeping all of the image
        # scaling to half training scale (0.25), thereby and expanding the field


        jtr_x = random.randint(-Pix2PixDataset.JIT_AMT, Pix2PixDataset.JIT_AMT)
        jtr_y = random.randint(-Pix2PixDataset.JIT_AMT, Pix2PixDataset.JIT_AMT)
        def jitter(img,pos,clr_back):
            ret = Image.new('RGBA', img.size, clr_back)
            ret.paste(img, pos)
            return ret

        resize_method = np.random.choice([0,1,2],p=[0.1,0.8,0.1]) # random selection with probabililtes
        if forced_resize_method >=0: resize_method = forced_resize_method
        if resize_method==0:
            # jitter then crop down to center without rescaling
            img_a = jitter(img_a,(jtr_x,jtr_y),(255, 255, 255, 255))
            img_b = jitter(img_b,(jtr_x,jtr_y),(255, 255, 255, 0))
            s0,s1 = (Pix2PixDataset.RAW_SZE - Pix2PixDataset.TRN_SZE)/2 , (Pix2PixDataset.RAW_SZE + Pix2PixDataset.TRN_SZE)/2
            img_a = img_a.crop((s0,s0,s1,s1))
            img_b = img_b.crop((s0,s0,s1,s1))
        elif resize_method==1:
            # jitter then scale down by 50% to training size
            img_a = resize( jitter(img_a,(jtr_x,jtr_y),(255, 255, 255, 255)) )
            img_b = resize( jitter(img_b,(jtr_x,jtr_y),(255, 255, 255, 0)) )
        else:
            # scale to 25% (1/2 training size), then paste onto training size image at jittered location
            img_aa = resize( img_a, int(Pix2PixDataset.TRN_SZE/2) )
            img_bb = resize( img_b, int(Pix2PixDataset.TRN_SZE/2) )
            img_a = Image.new('RGBA', (Pix2PixDataset.TRN_SZE,Pix2PixDataset.TRN_SZE), (255, 255, 255, 255))
            img_b = Image.new('RGBA', (Pix2PixDataset.TRN_SZE,Pix2PixDataset.TRN_SZE), (255, 255, 255, 0))
            img_a.paste(img_aa, ( int(Pix2PixDataset.TRN_SZE/4)+jtr_x , int(Pix2PixDataset.TRN_SZE/4)+jtr_y ) )
            img_b.paste(img_bb, ( int(Pix2PixDataset.TRN_SZE/4)+jtr_x , int(Pix2PixDataset.TRN_SZE/4)+jtr_y ) )


        imgten_a = transforms.ToTensor()(img_a)
        imgten_b = transforms.ToTensor()(img_b)
        imgten_a = transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))(imgten_a)
        imgten_b = transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))(imgten_b)

        if self.direction == "a2b": return imgten_a, imgten_b
        return imgten_b, imgten_a

    def __len__(self):
        return len(self.pths)

    @staticmethod
    def verify_extracted_data(pth, fldrs, check_for_corrupt=True):
        ret = {}
        ret['pth_parent'] = pth
        ret['fldrs'] = {'a0':fldrs['line'], 'a1':False, 'b':fldrs['rndr']}

        pth_a0 = os.path.join(pth,fldrs['line'])
        if not os.path.exists(pth_a0): raise Exception("!!!! Extracted data is not valid. The {} folder does not exist at the top level of the extracted ZIP.".format(fldrs['line']))

        pth_b = os.path.join(pth,fldrs['rndr'])
        if not os.path.exists(pth_b): raise Exception("!!!! Extracted data is not valid. The {} folder does not exist at the top level of the extracted ZIP.".format(fldrs['rndr']))

        ret['is_compound'] = False
        pth_a1 = False
        if 'fill' in fldrs and os.path.exists(os.path.join(pth,fldrs['fill'])):
            ret['fldrs']['a1'] = fldrs['fill']
            pth_a1 = os.path.join(pth,fldrs['fill'])
            ret['is_compound'] = True

        if not ret['is_compound']: print("No fill folder defined or none found in extracted data.\nFill data will be generated from alpha channel of render.")

        if check_for_corrupt:
            print("checking directory a0: {}".format(pth_a0))
            ret['corrupt_images'] = ImgUtil.find_corrupt_images(pth_a0)
            if ret['is_compound']:
                print("checking directory a1: {}".format(pth_a1))
                ret['corrupt_images'] = ImgUtil.find_corrupt_images(pth_a1)
            print("checking directory b: {}".format(pth_b))
            ret['corrupt_images'].extend( ImgUtil.find_corrupt_images(pth_b) )


        # Determine the items that are image files and exist in all directories
        def prefix_ext_sets(pth):
            tups = [os.path.splitext(f) for f in os.listdir(pth) if ImgUtil.verify_ext(f)]
            return set([t[0] for t in tups]), set([t[1] for t in tups])

        pfixs_a0, extn_a0 = prefix_ext_sets(pth_a0)
        pfixs_a1, extn_a1 = prefix_ext_sets(pth_a1) if ret['is_compound'] else (False, False)
        pfixs_b, extn_b = prefix_ext_sets(pth_b)

        if len(list(extn_a0)) != 1: raise Exception("!!!! More than one file extension found in a subfolder:\n{}".format(extn_a0))
        if ret['is_compound'] and (len(list(extn_a1)) != 1): raise Exception("!!!! More than one file extension found in a subfolder:\n{}".format(extn_a1))
        if len(list(extn_b)) != 1: raise Exception("!!!! More than one file extension found in a subfolder:\n{}".format(extn_b))

        extn_a0 = list(extn_a0)[0]
        extn_a1 = list(extn_a1)[0] if ret['is_compound'] else False
        extn_b = list(extn_b)[0]
        print("... discovered the following unique extensions: \n\t{}:'{}'\n\t{}:'{}'".format(ret['fldrs']['a0'], extn_a0, ret['fldrs']['b'], extn_b))
        if ret['is_compound']: print("\t{}:'{}'".format(ret['fldrs']['a1'], extn_a1))

        common_prefixes = False
        if ret['is_compound']:
            common_prefixes = list(pfixs_a0 & pfixs_a1 & pfixs_b)
            ret['orphans'] = list(pfixs_a0 - set(common_prefixes)) + list(pfixs_a1 - set(common_prefixes)) + list(pfixs_b - set(common_prefixes))
            ret['orphans'] = list(set(ret['orphans']))
        else:
            common_prefixes = list(pfixs_a0 & pfixs_b)
            ret['orphans'] = list(pfixs_a0 - pfixs_b) + list(pfixs_b - pfixs_a0)

        ret['pths'] = []
        for pfix in common_prefixes:
            d = {'pfix':pfix, 'a0':os.path.join(pth_a0, "{}{}".format(pfix,extn_a0) ), 'b':os.path.join(pth_b, "{}{}".format(pfix,extn_b) )}
            if ret['is_compound']: d['a1'] = os.path.join(pth_a1, "{}{}".format(pfix,extn_a1) )
            ret['pths'].append( d )

        #ret['pths_a0'] = sorted([os.path.join(pth_a0, f) for f in os.listdir(pth_a0) if any([f.startswith(pfix) for pfix in ret['common_prefixes']])])
        #ret['pths_b'] = sorted([os.path.join(pth_b, f) for f in os.listdir(pth_b) if any([f.startswith(pfix) for pfix in ret['common_prefixes']])])
        #ret['pths_a1'] = False
        #if pth_a1: ret['pths_a1'] = sorted([os.path.join(pth_a1, f) for f in os.listdir(pth_a1) if any([f.startswith(pfix) for pfix in ret['common_prefixes']])])

        # check image size from first rendered image
        # TODO: make sure all images are the same size
        ret['img_size'] = ImgUtil.load_img(ret['pths'][0]['b'], do_resize=False, do_flatten=False).size[0]

        return ret

    @staticmethod
    def define_input_pipeline(extraction_rslt, dinfo):
        all_dataset = Pix2PixDataset(extraction_rslt)
        sze = len(all_dataset)
        print("{} images found in the complete dataset".format(sze))

        val_size = max(30, int(0.02 * sze))
        test_size = int(0.18 * sze)
        train_size = sze - val_size - test_size

        print("{} train / {} test / {} validation".format(train_size,test_size,val_size))

        val_dataset, test_dataset, train_dataset =  torch.utils.data.random_split(all_dataset, [val_size, test_size, train_size])

        print("copying {} validation images to checkpoint folder".format(len(val_dataset)))

        def purge_dir(pth):
            if not os.path.exists(pth):
                os.makedirs(pth)
            else:
                shutil.rmtree(pth)
                os.makedirs(pth)

        purge_dir(dinfo.pth_vald)
        pth_a0 = os.path.join(dinfo.pth_vald, extraction_rslt['fldrs']['a0'])
        pth_a1 = os.path.join(dinfo.pth_vald, extraction_rslt['fldrs']['a1']) if extraction_rslt['is_compound'] else False
        pth_b = os.path.join(dinfo.pth_vald, extraction_rslt['fldrs']['b'])
        os.makedirs(pth_a0)
        if extraction_rslt['is_compound']: os.makedirs(pth_a1)
        os.makedirs(pth_b)
        for idx in val_dataset.indices:
            src_a0 = all_dataset.pths[idx]['a0']
            shutil.copyfile(src_a0, os.path.join(pth_a0, os.path.basename(src_a0) ))
            src_b = all_dataset.pths[idx]['b']
            shutil.copyfile(src_b, os.path.join(pth_b, os.path.basename(src_b) ))
            if extraction_rslt['is_compound']:
                src_a1 = all_dataset.pths[idx]['a1']
                shutil.copyfile(src_a1, os.path.join(pth_a1, os.path.basename(src_a1) ))




        return val_dataset, test_dataset, train_dataset, all_dataset

class Pix2Pix256RGBA():
    IMG_SIZE = 256
    DEFAULT_OPT = {
        'batch_size':1, # 'training batch size
        'test_batch_size':1, # 'testing batch size
        'direction':'b2a', # 'a2b or b2a
        'input_nc':4, # 'input image channels
        'output_nc':4, # 'output image channels
        'ngf':64, # 'generator filters in first conv layer
        'ndf':64, # 'discriminator filters in first conv layer
        'epoch_count':1, # 'the starting epoch count
        'niter':100, # '# of iter at starting learning rate
        'niter_decay':100, # '# of iter to linearly decay learning rate to zero
        'niter_per_checkpoint':20, # '# of iter to run before saving a checkpoint # KSTEINFE ADDED
        'lr':0.0002, # 'initial learning rate for adam
        'lr_policy':'lambda', # 'learning rate policy: lambda|step|plateau|cosine
        'lr_decay_iters':50, # 'multiply by a gamma every lr_decay_iters iterations
        'beta1':0.5, # 'beta1 for adam. default=0.5
        'cuda':True, # 'use cuda?
        'threads':4, # 'number of threads for data loader to use
        'seed':123, # 'random seed to use. Default=123
        'lamb':10 # 'weight on L1 term in objective
    }
    def __init__(self, opts=None):
        print("...initializing Pix2Pix256RGBA model.")
        op = Pix2Pix256RGBA.DEFAULT_OPT
        if opts is not None: op.update(opts)

        self.batch_size = op['batch_size']        # 'training batch size
        self.test_batch_size = op['test_batch_size']   # 'testing batch size
        self.direction = op['direction']         # 'a2b or b2a
        self.input_nc = op['input_nc']          # 'input image channels
        self.output_nc = op['output_nc']         # 'output image channels
        self.ngf = op['ngf']               # 'generator filters in first conv layer
        self.ndf = op['ndf']               # 'discriminator filters in first conv layer
        self.epoch_count = op['epoch_count']       # 'the starting epoch count
        self.niter = op['niter']             # '# of iter at starting learning rate
        self.niter_decay = op['niter_decay']       # '# of iter to linearly decay learning rate to zero
        self.niter_per_checkpoint:op['niter_per_checkpoint']  # '# of iter to run before saving a checkpoint # KSTEINFE ADDED
        self.lr = op['lr']                # 'initial learning rate for adam
        self.lr_policy = op['lr_policy']         # 'learning rate policy: lambda|step|plateau|cosine
        self.lr_decay_iters = op['lr_decay_iters']    # 'multiply by a gamma every lr_decay_iters iterations
        self.beta1 = op['beta1']             # 'beta1 for adam. default=0.5
        self.cuda = op['cuda']              # 'use cuda?
        self.threads = op['threads']           # 'number of threads for data loader to use
        self.seed = op['seed']              # 'random seed to use. Default=123
        self.lamb = op['lamb']              # 'weight on L1 term in objective

        cudnn.benchmark = True
        torch.manual_seed(self.seed)
        if self.cuda: torch.cuda.manual_seed(self.seed)

        self.generator, self.discriminator = False, False # these are initalized in static constructors

    @staticmethod
    def construct_training_model(opts=None, pths_pretrained=None):
        print("Constructing a model for training.")
        mdl = Pix2Pix256RGBA(opts)
        mdl.device = torch.device("cuda:0" if mdl.cuda else "cpu")

        mdl.generator = Pix2Pix256RGBA.define_G(mdl.input_nc, mdl.output_nc, mdl.ngf, 'batch', False, 'normal', 0.02, gpu_id=mdl.device)
        mdl.discriminator= Pix2Pix256RGBA.define_D(mdl.input_nc + mdl.output_nc, mdl.ndf, 'basic', gpu_id=mdl.device)

        mdl.criterionGAN = Pix2Pix256RGBA.GANLoss().to(mdl.device)
        mdl.criterionL1 = nn.L1Loss().to(mdl.device)
        mdl.criterionMSE = nn.MSELoss().to(mdl.device)

        # setup optimizer
        mdl.optimizer_g = optim.Adam(mdl.generator.parameters(), lr=mdl.lr, betas=(mdl.beta1, 0.999))
        mdl.optimizer_d = optim.Adam(mdl.discriminator.parameters(), lr=mdl.lr, betas=(mdl.beta1, 0.999))
        mdl.net_g_scheduler = Pix2Pix256RGBA.get_scheduler(mdl.optimizer_g, mdl)
        mdl.net_d_scheduler = Pix2Pix256RGBA.get_scheduler(mdl.optimizer_d, mdl)

        if pths_pretrained is not None:
            Pix2Pix256RGBA._restore_from_pickles(mdl, pths_pretrained)

        print("Training model constructed!")
        return mdl

    @staticmethod
    def construct_inference_model(pth_stat, opts=None):
        print("Constructing a model for inference.")
        mdl = Pix2Pix256RGBA(opts)
        mdl.device = torch.device("cuda:0" if mdl.cuda else "cpu")
        mdl.generator = Pix2Pix256RGBA.define_G(mdl.input_nc, mdl.output_nc, mdl.ngf, 'batch', False, 'normal', 0.02, gpu_id=mdl.device)
        Pix2Pix256RGBA._restore_generator_from_state_dict(mdl, pth_stat)
        return mdl

    @staticmethod
    def restore_from_checkpoint(pth_check, opts=None):

        print("...restoring model from checkpoints at \n\t'{}".format(pth_check))
        pths_pkld = Pix2Pix256RGBA.list_checkpoints(pth_check)[0]
        print("...generator of latest checkpoint found is \n\t'{}'".format(pths_pkld['pth_g']))
        mdl = Pix2Pix256RGBA.construct_training_model(opts,pths_pkld)

        print("Model restored from checkpoint!")
        return mdl, pths_pkld

    @staticmethod
    def _restore_generator_from_state_dict(mdl, pth_stat):
        print("...restoring from state dict at:\n{}".format(pth_stat))
        if mdl.cuda:
            print("...restoring to run on GPU")
            mdl.generator.load_state_dict(torch.load(pth_stat))
        else:
            print("...restoring to run on CPU")
            mdl.generator.load_state_dict(torch.load(pth_stat, map_location=lambda storage, location: storage))
            #return torch.load(pth_g, map_location=lambda storage, location: storage).to(mdl.device)

        #mdl.generator.eval()
        #mdl.discriminator.eval()

    @staticmethod
    def _restore_from_pickles(mdl, pths_pkld):
        pth_g, pth_d = pths_pkld['pth_g'], pths_pkld['pth_d']
        print("...restoring pickles:\nG\t{}\nD\t{}".format(pth_g, pth_d))
        if mdl.cuda:
            print("...restoring to run on GPU")
            mdl.generator = torch.load(pth_g).to(mdl.device)
            mdl.discriminator = torch.load(pth_d).to(mdl.device)
        else:
            print("...restoring to run on CPU")
            mdl.generator = torch.load(pth_g, map_location=lambda storage, location: storage).to(mdl.device)
            mdl.discriminator = torch.load(pth_d, map_location=lambda storage, location: storage).to(mdl.device)

        #mdl.generator.eval()
        #mdl.discriminator.eval()

    def list_checkpoints(pth_check):
        # here's hoping these files are in valid pairs, and that they sort well
        suffix_g = "_generator.pth"
        suffix_d = "_discriminator.pth"
        fnames_g = sorted([f for f in os.listdir(pth_check) if f.endswith(suffix_g)], reverse=True)
        fnames_d = sorted([f for f in os.listdir(pth_check) if f.endswith(suffix_d)], reverse=True)
        return [{'pth_g':os.path.join(pth_check, g), 'pth_d':os.path.join(pth_check, d)} for g,d in zip(fnames_g, fnames_d)]

    def generate(self, imgpil_in):
        if imgpil_in.size != (Pix2Pix256RGBA.IMG_SIZE, Pix2Pix256RGBA.IMG_SIZE):
            print("Resizing image from {} to {}".format(imgpil_in.size, (Pix2Pix256RGBA.IMG_SIZE, Pix2Pix256RGBA.IMG_SIZE) ))
            imgpil_in = imgpil_in.resize( (Pix2Pix256RGBA.IMG_SIZE, Pix2Pix256RGBA.IMG_SIZE) )

        if imgpil_in.mode != "RGBA": imgpil_in = imgpil_in.convert("RGBA")

        tform = Pix2PixDataset.tform()
        imgten_in = tform(imgpil_in)
        imgten_out = self.generator(imgten_in.unsqueeze(0).to(self.device))
        imgpil_out = ImgUtil.imgten_to_imgpil( imgten_out.detach().squeeze(0).cpu() )
        return( imgpil_out )

    def save_generator_for_inference(self, pth):
        torch.save(self.generator.state_dict(), pth)

    def save_checkpoint(self, epoch, pth_check):
        if not os.path.exists(pth_check): os.mkdir(pth_check)
        net_g_model_out_path = os.path.join(pth_check, "{:04}_generator.pth".format(epoch) )
        net_d_model_out_path = os.path.join(pth_check, "{:04}_discriminator.pth".format(epoch) )
        torch.save(self.generator, net_g_model_out_path)
        torch.save(self.discriminator, net_d_model_out_path)
        print("...checkpoint {:04} saved to Results directory.".format(epoch))

    def fit(self, train_ds, test_ds, dinfo, report_iter_callback, report_epoch_callback):
        pth_check, pth_rslts = dinfo.pth_chck , dinfo.pth_prog

        training_data_loader = DataLoader(dataset=train_ds, num_workers=self.threads, batch_size=self.batch_size, shuffle=True)
        testing_data_loader = DataLoader(dataset=test_ds, num_workers=self.threads, batch_size=self.test_batch_size, shuffle=True) #KS shuffle was false

        for epoch in range(self.epoch_count, self.niter + self.niter_decay + 1):
            # train
            #self.generator.train()
            #self.discriminator.train()
            for iteration, batch in enumerate(training_data_loader, 1):
                # forward
                real_a, real_b = batch[0].to(self.device), batch[1].to(self.device)
                fake_b = self.generator(real_a)

                ######################
                # (1) Update D network
                ######################

                self.optimizer_d.zero_grad()

                # train with fake
                fake_ab = torch.cat((real_a, fake_b), 1)
                pred_fake = self.discriminator.forward(fake_ab.detach())
                loss_d_fake = self.criterionGAN(pred_fake, False)

                # train with real
                real_ab = torch.cat((real_a, real_b), 1)
                pred_real = self.discriminator.forward(real_ab)
                loss_d_real = self.criterionGAN(pred_real, True)

                # Combined D loss
                loss_d = (loss_d_fake + loss_d_real) * 0.5
                loss_d.backward()
                self.optimizer_d.step()

                ######################
                # (2) Update G network
                ######################

                self.optimizer_g.zero_grad()

                # First, G(A) should fake the discriminator
                fake_ab = torch.cat((real_a, fake_b), 1)
                pred_fake = self.discriminator.forward(fake_ab)
                loss_g_gan = self.criterionGAN(pred_fake, True)

                # Second, G(A) = B
                loss_g_l1 = self.criterionL1(fake_b, real_b) * self.lamb
                loss_g = loss_g_gan + loss_g_l1
                loss_g.backward()
                self.optimizer_g.step()

                # callback an iteration report
                iter_info = { 'e':epoch, 'i':iteration, 'loss_d':loss_d.item(), 'loss_g':loss_g.item() }
                report_iter_callback(**iter_info)


            lr_g = Pix2Pix256RGBA.update_learning_rate(self.net_g_scheduler, self.optimizer_g)
            lr_d = Pix2Pix256RGBA.update_learning_rate(self.net_d_scheduler, self.optimizer_d)

            # test
            #self.generator.eval()
            #self.discriminator.eval()
            avg_psnr = 0
            for batch in testing_data_loader:
                input, target = batch[0].to(self.device), batch[1].to(self.device)

                prediction = self.generator(input)
                mse = self.criterionMSE(prediction, target)
                psnr = 10 * math.log10(1 / mse.item())
                avg_psnr += psnr

            # plot progress images and callback an epoch report
            iter_info = { 'e':epoch, 'sig_to_noise_ratio':avg_psnr / len(testing_data_loader), 'learning_rate_d':lr_d, 'learning_rate_g':lr_g }
            report_epoch_callback(**iter_info)

            #save checkpoint
            if epoch % self.niter_per_checkpoint == 0: self.save_checkpoint(epoch, pth_check)




    #############################################

    @staticmethod
    def define_G(input_nc, output_nc, ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
        net = None
        norm_layer = Pix2Pix256RGBA.get_norm_layer(norm_type=norm)

        net = Pix2Pix256RGBA.ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)

        return Pix2Pix256RGBA.init_net(net, init_type, init_gain, gpu_id)

    @staticmethod
    def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
        net = None
        norm_layer = Pix2Pix256RGBA.get_norm_layer(norm_type=norm)

        if netD == 'basic':
            net = Pix2Pix256RGBA.NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
        elif netD == 'n_layers':
            net = Pix2Pix256RGBA.NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
        elif netD == 'pixel':
            net = Pix2Pix256RGBA.PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
        else:
            raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)

        return Pix2Pix256RGBA.init_net(net, init_type, init_gain, gpu_id)

    @staticmethod
    def get_norm_layer(norm_type='instance'):
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        elif norm_type == 'switchable':
            norm_layer = SwitchNorm2d
        elif norm_type == 'none':
            norm_layer = None
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
        return norm_layer

    @staticmethod
    def get_scheduler(optimizer, mdl):
        if mdl.lr_policy == 'lambda':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + mdl.epoch_count - mdl.niter) / float(mdl.niter_decay + 1)
                return lr_l
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif mdl.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=mdl.lr_decay_iters, gamma=0.1)
        elif mdl.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        elif mdl.lr_policy == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=mdl.niter, eta_min=0)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', mdl.lr_policy)
        return scheduler

    # update learning rate (called once every epoch)
    @staticmethod
    def update_learning_rate(scheduler, optimizer):
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        return lr
        #print('learning rate = %.7f' % lr)

    @staticmethod
    def init_weights(net, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

        #print('...initialized network with %s' % init_type)
        net.apply(init_func)

    @staticmethod
    def init_net(net, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
        net.to(gpu_id)
        Pix2Pix256RGBA.init_weights(net, init_type, gain=init_gain)
        return net


    # Defines the generator that consists of Resnet blocks between a few
    # downsampling/upsampling operations.
    class ResnetGenerator(nn.Module):
        def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
            assert(n_blocks >= 0)
            super(Pix2Pix256RGBA.ResnetGenerator, self).__init__()
            self.input_nc = input_nc
            self.output_nc = output_nc
            self.ngf = ngf
            if type(norm_layer) == functools.partial:
                use_bias = norm_layer.func == nn.InstanceNorm2d
            else:
                use_bias = norm_layer == nn.InstanceNorm2d

            self.inc = Pix2Pix256RGBA.Inconv(input_nc, ngf, norm_layer, use_bias)
            self.down1 = Pix2Pix256RGBA.Down(ngf, ngf * 2, norm_layer, use_bias)
            self.down2 = Pix2Pix256RGBA.Down(ngf * 2, ngf * 4, norm_layer, use_bias)

            model = []
            for i in range(n_blocks):
                model += [Pix2Pix256RGBA.ResBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
            self.resblocks = nn.Sequential(*model)

            self.up1 = Pix2Pix256RGBA.Up(ngf * 4, ngf * 2, norm_layer, use_bias)
            self.up2 = Pix2Pix256RGBA.Up(ngf * 2, ngf, norm_layer, use_bias)

            self.outc = Pix2Pix256RGBA.Outconv(ngf, output_nc)

        def forward(self, input):
            out = {}
            out['in'] = self.inc(input)
            out['d1'] = self.down1(out['in'])
            out['d2'] = self.down2(out['d1'])
            out['bottle'] = self.resblocks(out['d2'])
            out['u1'] = self.up1(out['bottle'])
            out['u2'] = self.up2(out['u1'])

            return self.outc(out['u2'])

    class Inconv(nn.Module):
        def __init__(self, in_ch, out_ch, norm_layer, use_bias):
            super(Pix2Pix256RGBA.Inconv, self).__init__()
            self.inconv = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0,
                          bias=use_bias),
                norm_layer(out_ch),
                nn.ReLU(True)
            )

        def forward(self, x):
            x = self.inconv(x)
            return x

    class Down(nn.Module):
        def __init__(self, in_ch, out_ch, norm_layer, use_bias):
            super(Pix2Pix256RGBA.Down, self).__init__()
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3,
                          stride=2, padding=1, bias=use_bias),
                norm_layer(out_ch),
                nn.ReLU(True)
            )

        def forward(self, x):
            x = self.down(x)
            return x

    # Define a Resnet block
    class ResBlock(nn.Module):
        def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
            super(Pix2Pix256RGBA.ResBlock, self).__init__()
            self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

        def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
            conv_block = []
            p = 0
            if padding_type == 'reflect':
                conv_block += [nn.ReflectionPad2d(1)]
            elif padding_type == 'replicate':
                conv_block += [nn.ReplicationPad2d(1)]
            elif padding_type == 'zero':
                p = 1
            else:
                raise NotImplementedError('padding [%s] is not implemented' % padding_type)

            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                           norm_layer(dim),
                           nn.ReLU(True)]
            if use_dropout:
                conv_block += [nn.Dropout(0.5)]

            p = 0
            if padding_type == 'reflect':
                conv_block += [nn.ReflectionPad2d(1)]
            elif padding_type == 'replicate':
                conv_block += [nn.ReplicationPad2d(1)]
            elif padding_type == 'zero':
                p = 1
            else:
                raise NotImplementedError('padding [%s] is not implemented' % padding_type)
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                           norm_layer(dim)]

            return nn.Sequential(*conv_block)

        def forward(self, x):
            out = x + self.conv_block(x)
            return nn.ReLU(True)(out)

    class Up(nn.Module):
        def __init__(self, in_ch, out_ch, norm_layer, use_bias):
            super(Pix2Pix256RGBA.Up, self).__init__()
            self.up = nn.Sequential(
                # nn.Upsample(scale_factor=2, mode='nearest'),
                # nn.Conv2d(in_ch, out_ch,
                #           kernel_size=3, stride=1,
                #           padding=1, bias=use_bias),
                nn.ConvTranspose2d(in_ch, out_ch,
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1,
                                   bias=use_bias),
                norm_layer(out_ch),
                nn.ReLU(True)
            )

        def forward(self, x):
            x = self.up(x)
            return x

    class Outconv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super(Pix2Pix256RGBA.Outconv, self).__init__()
            self.outconv = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0),
                nn.Tanh()
            )

        def forward(self, x):
            x = self.outconv(x)
            return x

    # Defines the PatchGAN discriminator with the specified arguments.
    class NLayerDiscriminator(nn.Module):
        def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
            super(Pix2Pix256RGBA.NLayerDiscriminator, self).__init__()
            if type(norm_layer) == functools.partial:
                use_bias = norm_layer.func == nn.InstanceNorm2d
            else:
                use_bias = norm_layer == nn.InstanceNorm2d

            kw = 4
            padw = 1
            sequence = [
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True)
            ]

            nf_mult = 1
            nf_mult_prev = 1
            for n in range(1, n_layers):
                nf_mult_prev = nf_mult
                nf_mult = min(2**n, 8)
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                              kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]

            nf_mult_prev = nf_mult
            nf_mult = min(2**n_layers, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

            sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

            if use_sigmoid:
                sequence += [nn.Sigmoid()]

            self.model = nn.Sequential(*sequence)

        def forward(self, input):
            return self.model(input)

    class PixelDiscriminator(nn.Module):
        def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
            super(Pix2Pix256RGBA.PixelDiscriminator, self).__init__()
            if type(norm_layer) == functools.partial:
                use_bias = norm_layer.func == nn.InstanceNorm2d
            else:
                use_bias = norm_layer == nn.InstanceNorm2d

            self.net = [
                nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
                norm_layer(ndf * 2),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

            if use_sigmoid:
                self.net.append(nn.Sigmoid())

            self.net = nn.Sequential(*self.net)

        def forward(self, input):
            return self.net(input)

    class GANLoss(nn.Module):
        def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
            super(Pix2Pix256RGBA.GANLoss, self).__init__()
            self.register_buffer('real_label', torch.tensor(target_real_label))
            self.register_buffer('fake_label', torch.tensor(target_fake_label))
            if use_lsgan:
                self.loss = nn.MSELoss()
            else:
                self.loss = nn.BCELoss()

        def get_target_tensor(self, input, target_is_real):
            if target_is_real:
                target_tensor = self.real_label
            else:
                target_tensor = self.fake_label
            return target_tensor.expand_as(input)

        def __call__(self, input, target_is_real):
            target_tensor = self.get_target_tensor(input, target_is_real)
            return self.loss(input, target_tensor)
