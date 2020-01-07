import os
import runway
from runway.data_types import image, category
#from model_switcher import ModelSwitcher
from pix2pix import Pix2Pix256RGB, ImgUtil
import tensorflow as tf


# Setup the Model
########################################
setup_options = {'checkpoint': runway.file(extension='.h5')}
@runway.setup(options=setup_options)
def setup(opts):
    print('[SETUP] Ran with {} options defined.'.format(len(opts)))
    print("[SETUP] Ran tensorflow version {}".format(tf.__version__))

    checkpoint_path = opts['checkpoint']

    if checkpoint_path is not None:
        print("Checkpoint found. Attempting to restore {}".format(checkpoint_path))
        mdl = Pix2Pix256RGB.restore_from_hdf5(checkpoint_path)
        return mdl

    print("!!! No checkpoint path found.")
    try:
        checkpoint_path = os.path.join(os.getcwd(), "testing_model.h5")
        print("Attempting to restore local testing checkpoint {}".format(checkpoint_path))
        mdl = Pix2Pix256RGB.restore_from_hdf5(checkpoint_path)
        return mdl
    except Exception as e:
        print(e)
        print("!!! Cannot restore this model. Only dummy images will be returned.")
        mdl = Pix2Pix256RGB()
        return mdl




# Generate Image
########################################

def generate_from_PIL_img(mdl, img_pil):
    img_ten_in = ImgUtil.imgpil_to_imgten(img_pil) # converts a PIL image to a normalized tensor of dimension [1,sz,sz,3]
    img_ten_out = mdl.generator(img_ten_in, training=True)
    return( ImgUtil.imgten_to_imgpil(img_ten_out) ) # converts a normalized tensor of dimension [1,sz,sz,3] to a PIL image

generate_command_inputs = {
  'image_in': image(width=256, height=256)
}
generate_command_outputs = {
  'image_out': image(width=256, height=256)
}
@runway.command(name='generate',
                inputs=generate_command_inputs,
                outputs=generate_command_outputs,
                description='this thing does a thing')
def generate(model, args):
    print('[GENERATE]\n image_in: "{}"'.format(args['image_in']))
    #model.switch_to(args['model_to_apply'])
    output_image = args['image_in']
    print(model.generator)
    if model.generator: output_image = generate_from_PIL_img(model, args['image_in'])
    return {'image_out': output_image}


# Generate Image
########################################
if __name__ == '__main__':
    runway.run()
    #runway.run(host='0.0.0.0', port=8000)
