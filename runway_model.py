import os
import runway
from runway.data_types import image, category
#from model_switcher import ModelSwitcher
from pix2pixRGBA import Pix2Pix256RGBA, ImgUtil
import torch
#import tensorflow as tf


DUMMY_CHECKPOINT_FNAME = "200118a_pretty_mushroom.pt"
#DUMMY_CHECKPOINT_FNAME = "200118a_plaster.pt"

# Setup the Model
########################################
setup_options = {'checkpoint': runway.file(extension='.pt')}
@runway.setup(options=setup_options)
def setup(opts):
    print('[SETUP] Ran with {} options defined.'.format(len(opts)))
    print("[SETUP] Ran torch version {}".format(torch.__version__))

    checkpoint_path = opts['checkpoint']
    mdl_opts = {'cuda':torch.cuda.is_available()}
    mdl = False

    if checkpoint_path is not None:
        print("Checkpoint found. Attempting to restore {}".format(checkpoint_path))
        mdl = Pix2Pix256RGBA.construct_inference_model(checkpoint_path, mdl_opts)
        return mdl

    print("!!! No checkpoint path found.")

    try:
        checkpoint_path = os.path.join(os.getcwd(), DUMMY_CHECKPOINT_FNAME)
        print("Attempting to restore local testing checkpoint {}".format(checkpoint_path))
        mdl = Pix2Pix256RGBA.construct_inference_model(checkpoint_path, mdl_opts)
        return mdl
    except Exception as e:
        print(e)
        print("!!! Cannot restore this model. Only dummy images will be returned.")
        mdl = Pix2Pix256RGBA()
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
  'image_out': image(width=256, height=256, channels=4)
}
@runway.command(name='generate',
                inputs=generate_command_inputs,
                outputs=generate_command_outputs,
                description='this thing does a thing')
def generate(model, args):
    print('[GENERATE]\n image_in: "{}"'.format(args['image_in']))
    output_image = args['image_in']
    size_in = args['image_in'].size

    if model.generator:
        with torch.no_grad():
            output_image = model.generate(args['image_in'])

    # add registration pixels to corners
    #pixels = output_image.load() # create the pixel map
    clr_reg = (255,0,0,255)
    output_image.putpixel((0, 0), clr_reg)
    output_image.putpixel((255, 255), clr_reg)
    #pixels[254,254] = (255,255,255,255)

    if output_image.size != size_in:
        print("Resizing output image from {} to {}".format(output_image.size, size_in ))
        output_image = output_image.resize( size_in )

    print(output_image.mode)
    return {'image_out': output_image}


# Generate Image
########################################
if __name__ == '__main__':
    runway.run()
    #runway.run(host='0.0.0.0', port=8000)
