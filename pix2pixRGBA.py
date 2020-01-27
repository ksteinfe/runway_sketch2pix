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
                img = ImgUtil.load_img(os.path.join(pth, fname), False)
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
        """Alpha composite an RGBA Image with a specified color. Source: http://stackoverflow.com/a/9459208/284318 """
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

    def __init__(self, extraction_rslt, direction="a2b"):
        super(Pix2PixDataset, self).__init__()
        self.direction = direction

        self.pths_a = extraction_rslt['pths_a']
        self.pths_b = extraction_rslt['pths_b']
        for pa, pb in zip(self.pths_a, self.pths_b):
            if os.path.splitext(os.path.basename(pa))[0] != os.path.splitext(os.path.basename(pb))[0]:
                raise Exception("Dataset contains a non-matching pair.\n{}\n{}".format(pa,pb))

    @staticmethod
    def tform():
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))]

        return( transforms.Compose(transform_list) )

    @staticmethod
    def line_to_fill(line, rndr):
        return Pix2PixDataset._fill_and_jiggle(line,rndr, Pix2PixDataset.FILL_COLOR, jig_amt=0, rot_amt=0, scl_amt=0)

    @staticmethod
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

    @staticmethod
    def load_img_a(filepath):
        return ImgUtil.load_img(filepath, do_resize=True, do_flatten=True)

    @staticmethod
    def load_img_b(filepath):
        return ImgUtil.load_img(filepath, do_resize=True, do_flatten=False)

    def __getitem__(self, index):
        a = Pix2PixDataset.load_img_a(self.pths_a[index])
        b = Pix2PixDataset.load_img_b(self.pths_b[index])

        # here, images are 'jittered': resized to 286 and then cropped back to 256
        a = Pix2PixDataset._fill_and_jiggle(a,b, Pix2PixDataset.FILL_COLOR )

        jtr_amt = random.randint(10, 20)
        jtr_sze = 256+jtr_amt

        a = a.resize((jtr_sze, jtr_sze), Image.BICUBIC)
        b = b.resize((jtr_sze, jtr_sze), Image.BICUBIC)
        a = transforms.ToTensor()(a)
        b = transforms.ToTensor()(b)
        w_offset = random.randint(0, max(0, jtr_sze - 256 - 1))
        h_offset = random.randint(0, max(0, jtr_sze - 256 - 1))

        a = a[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
        b = b[:, h_offset:h_offset + 256, w_offset:w_offset + 256]

        a = transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))(a)
        b = transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))(b)

        """
        if random.random() < 0.5:
            idx = [i for i in range(a.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            a = a.index_select(2, idx)
            b = b.index_select(2, idx)
        """

        if self.direction == "a2b": return a, b
        return b, a

    def __len__(self):
        return len(self.pths_a)

    @staticmethod
    def verify_extracted_data(pth, fldrs, check_for_corrupt=True):
        pth_a = os.path.join(pth,fldrs['a'])
        pth_b = os.path.join(pth,fldrs['b'])
        if not os.path.exists(pth_a) or not os.path.exists(pth_b):
            raise Exception("!!!! Extracted data is not valid. Check the names of any subfolders.")

        ret = {}
        ret['fldrs'] = {'a':fldrs['a'], 'b':fldrs['b']}
        if check_for_corrupt:
            print("checking directory a: {}".format(pth_a))
            ret['corrupt_images'] = ImgUtil.find_corrupt_images(pth_a)
            print("checking directory b: {}".format(pth_b))
            ret['corrupt_images'].extend( ImgUtil.find_corrupt_images(pth_b) )

        # Determine the items that are image files and exist in both directories
        a_set = set([os.path.splitext(f)[0] for f in os.listdir(pth_a) if ImgUtil.verify_ext(f)])
        b_set = set([os.path.splitext(f)[0] for f in os.listdir(pth_b) if ImgUtil.verify_ext(f)])
        ret['common_prefixes'] = list(a_set & b_set)
        ret['orphans'] = list(a_set - b_set) + list(b_set - a_set)

        ret['pths_a'] = sorted([os.path.join(pth_a, f) for f in os.listdir(pth_a) if any([f.startswith(pfix) for pfix in ret['common_prefixes']])])
        ret['pths_b'] = sorted([os.path.join(pth_b, f) for f in os.listdir(pth_b) if any([f.startswith(pfix) for pfix in ret['common_prefixes']])])

        return ret

    @staticmethod
    def define_input_pipeline(extraction_rslt, dinfo):
        all_dataset = Pix2PixDataset(extraction_rslt)
        sze = len(all_dataset)
        print("{} images found in the complete dataset".format(sze))

        val_size = int(0.02 * sze)
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
        pth_a = os.path.join(dinfo.pth_vald, extraction_rslt['fldrs']['a'])
        pth_b = os.path.join(dinfo.pth_vald, extraction_rslt['fldrs']['b'])
        os.makedirs(pth_a)
        os.makedirs(pth_b)
        for idx in val_dataset.indices:
            src_a = all_dataset.pths_a[idx]
            shutil.copyfile(src_a, os.path.join(pth_a, os.path.basename(src_a) ))
            src_b = all_dataset.pths_b[idx]
            shutil.copyfile(src_b, os.path.join(pth_b, os.path.basename(src_b) ))

        return val_dataset, test_dataset, train_dataset

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
