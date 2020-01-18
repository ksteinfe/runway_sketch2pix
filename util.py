import IPython as ipy
import os, datetime, time, glob, zipfile, shutil
version_number = 0.1
print('util module r{} has loaded.'.format(version_number))



def printHTML(s):
  ipy.display.display(ipy.display.HTML('<div style="font-size: large; margin-top: 1em; color: chocolate;">{}</div>'.format(s)))


class DirUtil():
    name_progimg_dir = "progress_images"
    name_valdimg_dir = "validation_images"
    name_checkpt_dir = "checkpoints"

    def __init__(self):
        self.name = False
        self.pth_root = False
        self.pth_chck = False
        self.pth_prog = False
        self.pth_vald = False


    @staticmethod
    def verify_training_data_directory(pth_gdrive_root, name_train_dir, name_zip):
        pth_train_src = os.path.join(pth_gdrive_root, name_train_dir)
        if not os.path.exists(pth_train_src): raise Exception("!!! the specified training data source directory does not exist on Google Drive\n{}".format(pth_train_src))

        print("... looking for ZIP files in the specified directory on Google Drive: ".format(pth_train_src))
        if name_zip not in os.listdir(pth_train_src): raise Exception("!!! the zip file '{}' was not found in the given directory.\nHere's what I found there: \n{}".format(name_zip, os.listdir(pth_train_src)))

        pth_zip = os.path.join(pth_train_src, name_zip)
        print("found training data at {}".format(pth_zip))
        return pth_zip

    @staticmethod
    def extract_training_data(pth_zip, pth_lcl):
        print('... extracting files in {}'.format(pth_zip))
        shutil.rmtree(pth_lcl, ignore_errors=True)
        with zipfile.ZipFile(pth_zip, 'r') as zip_obj:
            print("looks like {} files are in this ZIP".format(len(zip_obj.infolist())))
            zip_obj.extractall(pth_lcl)
        return pth_lcl

    @staticmethod
    def _verify_results_root(pth_gdrive_root, name_rslt_dir):
        pth_rslts_root = os.path.join(pth_gdrive_root, name_rslt_dir)
        if not os.path.exists(pth_rslts_root): raise Exception("!!! the specified results directory does not exist on Google Drive\n{}".format(pth_rslts_root))
        return pth_rslts_root

    @staticmethod
    def verify_results_directory(pth_gdrive_root, name_rslt_dir, name_exprmnt):
        pth_rslts_root = DirUtil._verify_results_root(pth_gdrive_root, name_rslt_dir)
        if not pth_rslts_root: return False

        dinfo = DirUtil()
        dinfo.pth_root = os.path.join(pth_rslts_root, name_exprmnt)
        if not os.path.exists(dinfo.pth_root): raise Exception("!!! the specified experiment does not exist in the results directory\n{}".format(dinfo.pth_root))

        dinfo.pth_chck = os.path.join(dinfo.pth_root, DirUtil.name_checkpt_dir)
        if not os.path.exists(dinfo.pth_chck): raise Exception("!!! the experiment directory does not contain a checkpoints folder\n{}".format(dinfo.pth_chck))
        dinfo.pth_prog = os.path.join(dinfo.pth_root, DirUtil.name_progimg_dir)
        if not os.path.exists(dinfo.pth_prog): raise Exception("!!! the experiment directory does not contain a checkpoints folder\n{}".format(dinfo.pth_prog))
        dinfo.pth_vald = os.path.join(dinfo.pth_root, DirUtil.name_valdimg_dir)
        if not os.path.exists(dinfo.pth_vald): raise Exception("!!! the experiment directory does not contain a checkpoints folder\n{}".format(dinfo.pth_vald))

        return dinfo

    @staticmethod
    def initalize_new_results_directory(pth_gdrive_root, name_rslt_dir, name_exprmnt):
        pth_rslts_root = DirUtil._verify_results_root(pth_gdrive_root, name_rslt_dir)
        if not pth_rslts_root: return

        print("...initializing a fresh experiment results directory at {}".format(pth_rslts_root))
        dinfo = DirUtil()

        for char in 'abcdefghijkmnpqrstuvwxyz':
            name = "{}{}_{}".format(datetime.date.today().strftime('%y%m%d'), char, name_exprmnt)
            if name not in os.listdir( pth_rslts_root ): break

        print("... initializing experiment {}".format(name))
        dinfo._name = name

        dinfo.pth_root = os.path.join(pth_rslts_root, dinfo._name)
        print("... setting up a results directory on Google Drive at\n{}".format(dinfo.pth_root))
        if not os.path.exists(dinfo.pth_root): os.makedirs(dinfo.pth_root)
        else:
            raise Exception("!!! a directory already exists for this experiment. go check Google Drive and back things up!")

        dinfo.pth_chck = os.path.join(dinfo.pth_root, DirUtil.name_checkpt_dir)
        if not os.path.exists(dinfo.pth_chck): os.makedirs(dinfo.pth_chck)
        else:
            raise Exception("!!! a checkpoint sub-directory already exists for this experiment. go check Google Drive and back things up!")

        dinfo.pth_prog = os.path.join(dinfo.pth_root, DirUtil.name_progimg_dir)
        if not os.path.exists(dinfo.pth_prog): os.makedirs(dinfo.pth_prog)
        else:
            raise Exception("!!! a progress images sub-directory already exists for this experiment. go check Google Drive and back things up!")

        dinfo.pth_vald = os.path.join(dinfo.pth_root, DirUtil.name_valdimg_dir)
        if not os.path.exists(dinfo.pth_vald): os.makedirs(dinfo.pth_vald)
        else:
            raise Exception("!!! a validation images sub-directory already exists for this experiment. go check Google Drive and back things up!")

        return dinfo

    @staticmethod
    def purge_dir(pth):
        if not os.path.exists(pth):
            os.makedirs(pth)
        else:
            for f in glob.glob(pth+'/*'): os.remove(f)
