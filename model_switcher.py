import json, os, urllib
from pix2pix import Pix2Pix256RGB, ImgUtil
import tensorflow as tf

#p = Path(os.path.dirname(sys.executable))
#LCL_MDL_PTH = os.path.join( os.getcwd() , "saved_models")
#LCL_MDL_PTH = os.path.join( os.path.abspath(r".\\") , "saved_models")
#LCL_MDL_PTH = r".\saved_models"
LCL_MDL_PTH =r'./saved_models'

class ModelSwitcher():
    def __init__(self, mdls_lst):
        print("init")
        if not os.path.exists(LCL_MDL_PTH): os.makedirs(LCL_MDL_PTH)
        self._mdls_info = {m['display_name']:m for m in mdls_lst} # convert list to dict organized by display_name
        self._mdl_name_current = False
        self._mdl_p2p = Pix2Pix256RGB()
        self.switch_to(mdls_lst[0]['display_name'])

    def switch_to(self, mdl_name):
        print("switch_to {}".format(mdl_name))
        if mdl_name not in self._mdls_info: raise Error("Selected model is not available.")
        if self._mdl_name_current == mdl_name:
            print("selected model {} is the same as the current model {}".format(mdl_name, self._mdl_name_current))
            return True

        print("selected model {} is different than current model {}".format(mdl_name, self._mdl_name_current))
        mdl_info_selected = self._mdls_info[mdl_name]
        pth_mdl = '{}/{}'.format(LCL_MDL_PTH, mdl_info_selected['fname'])
        print("looking for {}".format(pth_mdl))
        print("downloaded models are: {}".format(os.listdir(LCL_MDL_PTH)))
        if not os.path.exists(pth_mdl):
            # download model
            print("the file {} related to this model hasn't been downloaded yet.\ndownloading now from {}".format(mdl_info_selected['fname'], mdl_info_selected['url']))
            last_percent_reported = 0
            def download_progress_hook(count, blockSize, totalSize):
                global last_percent_reported
                percent = int(count * blockSize * 100 / totalSize)

                if last_percent_reported != percent:
                    if percent % 5 == 0:
                        sys.stdout.write("%s%%" % percent)
                        sys.stdout.flush()
                    else:
                        sys.stdout.write(".")
                        sys.stdout.flush()
                last_percent_reported = percent

            urllib.request.urlretrieve(mdl_info_selected['url'], pth_mdl, reporthook=download_progress_hook)
            print("{} models have been downloaded locally.".format(len(os.listdir(LCL_MDL_PTH))))

        # restore model
        #self._mdl_p2p = self._mdl_p2p.restore_from_hdf5(pth_mdl)
        self._mdl_p2p.generator = tf.keras.models.load_model(pth_mdl)
        print("model restored from {}".format(pth_mdl))

        # set current model name
        self._mdl_name_current = mdl_name

    def generate_from_PIL_img(self, img_pil):
        img_ten_in = ImgUtil.imgpil_to_imgten(img_pil) # converts a PIL image to a normalized tensor of dimension [1,sz,sz,3]
        img_ten_out = self._mdl_p2p.generator(img_ten_in, training=True)
        return( ImgUtil.imgten_to_imgpil(img_ten_out) ) # converts a normalized tensor of dimension [1,sz,sz,3] to a PIL image

    # Loads available models
    @staticmethod
    def load_model_info():
        info = False
        with open('available_models.json') as f: info = json.load(f)
        if not info: raise FileNotFoundError("could not find JSON of available models.")
        print("available_models.json indicates that there are {} models available for loading.".format(len(info)))
        print("\n\t".join([k["display_name"] for k in info]))
        return info
