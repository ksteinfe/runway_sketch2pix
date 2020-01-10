from __future__ import absolute_import, division, print_function, unicode_literals
import os, datetime, time, glob, zipfile, shutil
from matplotlib import pyplot as plt
import IPython as ipy
from PIL import Image
import tensorflow as tf
import numpy as np

version_number = 0.1
print('pix2pix r{} has loaded.'.format(version_number))


class ImgUtil():

    # converts a PIL image to a normalized tensor of dimension [1,sz,sz,3]
    @staticmethod
    def imgpil_to_imgten(imgpil):
        imgten = tf.keras.preprocessing.image.img_to_array(imgpil)
        imgten = tf.cast(imgten, tf.float32)
        imgten = tf.expand_dims(imgten, 0)
        imgten = ImgUtil.normalize_imgten(imgten)
        return imgten

    # converts a normalized tensor of dimension [1,sz,sz,3] to a PIL image
    @staticmethod
    def imgten_to_imgpil(imgten):
        imgarr = imgten[0].numpy() # TODO: what if we are passed more than one image?
        imgarr = ((imgarr * 0.5) + 0.5)*255
        imgpil = Image.fromarray(np.uint8(imgarr))
        return imgpil

    # loads a paired image file and returns two PIL image_tensor
    @staticmethod
    def load_img_pair_as_pil(pth_img):
        img_full = Image.open(pth_img)
        w,h = img_full.size
        img_left, img_right = img_full.crop((0,0,w/2,h)), img_full.crop((w/2,0,w,h))
        return img_right, img_left # matching expectations set by function below


    @staticmethod
    def load_img_pair_as_ten(image_file):
        #tf.print(image_file)
        image = tf.io.read_file(image_file)
        #image = tf.image.decode_jpeg(image)
        image = tf.image.decode_png(image, channels=3)

        w = tf.shape(image)[1]
        w = w // 2
        img_left = image[:, :w, :]
        img_right = image[:, w:, :]

        img_right = tf.cast(img_right, tf.float32)
        img_left = tf.cast(img_left, tf.float32)

        return img_right, img_left


    @staticmethod
    def resize_imgten(imgten, sz):
        return tf.image.resize(imgten, [sz, sz], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    @staticmethod
    def random_crop_pair(input_image, real_image):
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.image.random_crop(stacked_image, size=[2, 256, 256, 3])
        return cropped_image[0], cropped_image[1]

    # normalizing the images to [-1, 1]
    @staticmethod
    def normalize_imgten(img): return (img / 127.5) - 1

    @staticmethod
    def find_corrupt_images(pth, channels=3):
        bad_ones = []
        for fname in os.listdir(pth):
            try:
                with tf.io.gfile.GFile(os.path.join(pth, fname), 'rb') as fid: image_data = fid.read()
                image_tensor = tf.image.decode_png(image_data,channels=channels,name=None)
            except:
                bad_ones.append(fname)
        return(bad_ones)

    @staticmethod
    def plot_imgtens(img_a, img_b, img_c=None, pth_save=None):
        # TODO: ensure tensors are the same shape
        cnt,w,h = len(img_a), img_a.shape[1], img_a.shape[2]
        dpi = 100
        if img_c is not None:
            plt.figure(figsize=(w/dpi*3*1.5,h/dpi*1.5), dpi=dpi)
            display_list = [img_a[0], img_b[0], img_c[0]]
            title = ['given', 'target', 'generated']
        else:
            plt.figure(figsize=(w/dpi*2*1.5,h/dpi*1.5), dpi=dpi)
            display_list = [img_a[0], img_b[0]]
            title = None

        n=1
        for j in range(cnt):
            for i in range(len(display_list)):
                plt.subplot(cnt, len(display_list), n) # nrows, ncols, index
                if title is not None: plt.title(title[i])
                # getting the pixel values between [0, 1] to plot it.
                plt.imshow(display_list[i] * 0.5 + 0.5)
                plt.axis('off')
                n+=1

        if pth_save is not None: plt.savefig(pth_save) # save before showing
        plt.show()


class Pix2Pix256RGB():
    IMG_SIZE = 256
    def __init__(self):
        print("...initializing Pix2Pix256RGB model.")
        #self.img_size = 256
        self.output_channels = 3
        self.lmbda = 100

        self.generator, self.discriminator = False, False
        self.loss_object, self.generator_optimizer, self.discriminator_optimizer = False, False, False
        self.checkpoint = False

    @staticmethod
    def construct_training_model():
        print("Constructing a model for training.")
        mdl = Pix2Pix256RGB()
        mdl.generator = mdl._construct_generator()
        mdl.discriminator = mdl._construct_discriminator()

        print("...initalizing optimizers.")
        mdl.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        mdl.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        mdl.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

        mdl.checkpoint = mdl._construct_checkpoint_saver()

        print("Training model constructed!")
        return mdl

    @staticmethod
    def construct_inference_model():
        print("Constructing a model for inference.")
        mdl = Pix2Pix256RGB()
        mdl.generator = mdl._construct_generator()
        return mdl

    @staticmethod
    def restore_from_checkpoint(pth_checkpoint):
        mdl = Pix2Pix256RGB.construct_training_model()
        print("...restoring model from checkpoints at '{}".format(pth_checkpoint))
        latest = tf.train.latest_checkpoint(pth_checkpoint)
        print("...restoring checkpoint '{}'".format(os.path.basename(latest)))
        mdl.checkpoint.restore(latest)
        print("Model restored from checkpoint!")
        return mdl

    @staticmethod
    def restore_from_hdf5(pth_h5):
        mdl = Pix2Pix256RGB.construct_inference_model()
        print("...restoring model from hdf5 at '{}'".format(pth_h5))
        mdl.generator = tf.keras.models.load_model(pth_h5)
        print("Model restored from HDF5!")
        return mdl

    @staticmethod
    def define_input_pipeline(pth_train_locl, dinfo, img_ext, buffer_size=400, batch_size=1):

        all_dataset = tf.data.Dataset.list_files(os.path.join(pth_train_locl ,'*.{}'.format(img_ext)))
        sze = len(list(all_dataset))
        print("{} images found in the complete dataset".format(sze))

        train_size = int(0.80 * sze)
        test_size = int(0.18 * sze)
        val_size = int(0.02 * sze)

        print("{} train / {} test / {} validation".format(train_size,test_size,val_size))

        all_dataset = all_dataset.shuffle(buffer_size)
        val_dataset = all_dataset.take(val_size)
        all_dataset = all_dataset.skip(val_size)
        test_dataset = all_dataset.take(test_size)
        train_dataset = all_dataset.skip(test_size)


        @tf.function()
        def random_jitter(input_image, real_image, do_mirror=False):
            input_image = ImgUtil.resize_imgten(input_image, 286) # resizing to 286 x 286 x 3
            real_image = ImgUtil.resize_imgten(real_image, 286) # resizing to 286 x 286 x 3
            input_image, real_image = ImgUtil.random_crop_pair(input_image, real_image) # randomly cropping to 256 x 256 x 3

            # random mirroring
            if do_mirror:
                if tf.random.uniform(()) > 0.5:
                    input_image = tf.image.flip_left_right(input_image)
                    real_image = tf.image.flip_left_right(real_image)

            return input_image, real_image

        def load_image_train(image_file_ten):
            input_image, real_image = ImgUtil.load_img_pair_as_ten(image_file_ten)
            input_image, real_image = random_jitter(input_image, real_image)
            input_image = ImgUtil.normalize_imgten(input_image)
            real_image = ImgUtil.normalize_imgten(real_image)
            return input_image, real_image

        def load_image_test(image_file_ten):
            input_image, real_image = ImgUtil.load_img_pair_as_ten(image_file_ten)
            input_image = ImgUtil.resize_imgten(input_image, Pix2Pix256RGB.IMG_SIZE)
            real_image = ImgUtil.resize_imgten(real_image, Pix2Pix256RGB.IMG_SIZE)
            input_image = ImgUtil.normalize_imgten(input_image)
            real_image = ImgUtil.normalize_imgten(real_image)
            return input_image, real_image

        # validation set
        # first copy images to checkpoint folder
        '''
        DirUtil.purge_dir(dinfo.pth_vald)
        for next_element in val_dataset:
            pth_src = next_element.numpy().decode()
            shutil.copyfile(pth_src, os.path.join(dinfo.pth_vald, os.path.basename(pth_src) ))
        val_dataset = val_dataset.map(load_image_test)
        val_dataset = val_dataset.batch(batch_size)
        '''

        # test set
        test_dataset = test_dataset.map(load_image_test)
        test_dataset = test_dataset.batch(batch_size)

        # train set
        train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.shuffle(buffer_size)
        train_dataset = train_dataset.batch(batch_size)

        print("{} images copied to results folder to be used for validation".format(len(list(val_dataset))))
        print("{} images set aside and have been processed for testing".format(len(list(test_dataset))))
        print("remaining {} images have been processed for training".format(len(list(train_dataset))))

        return val_dataset, test_dataset, train_dataset

    def _downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
          tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                 kernel_initializer=initializer, use_bias=False))
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())
        return result

    def _upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout: result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())
        return result

    def _construct_generator(self):
        print("...initalizing generator")
        inputs = tf.keras.layers.Input(shape=[256,256,3])

        down_stack = [
            self._downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
            self._downsample(128, 4), # (bs, 64, 64, 128)
            self._downsample(256, 4), # (bs, 32, 32, 256)
            self._downsample(512, 4), # (bs, 16, 16, 512)
            self._downsample(512, 4), # (bs, 8, 8, 512)
            self._downsample(512, 4), # (bs, 4, 4, 512)
            self._downsample(512, 4), # (bs, 2, 2, 512)
            self._downsample(512, 4), # (bs, 1, 1, 512)
        ]

        up_stack = [
            self._upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
            self._upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
            self._upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
            self._upsample(512, 4), # (bs, 16, 16, 1024)
            self._upsample(256, 4), # (bs, 32, 32, 512)
            self._upsample(128, 4), # (bs, 64, 64, 256)
            self._upsample(64, 4), # (bs, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(self.output_channels, 4,
                                             strides=2,
                                             padding='same',
                                             kernel_initializer=initializer,
                                             activation='tanh') # (bs, 256, 256, 3)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    def _construct_discriminator(self):
        print("...initalizing discriminator")
        initializer = tf.random_normal_initializer(0., 0.02)
        inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
        tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

        x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

        down1 = self._downsample(64, 4, False)(x) # (bs, 128, 128, 64)
        down2 = self._downsample(128, 4)(down1) # (bs, 64, 64, 128)
        down3 = self._downsample(256, 4)(down2) # (bs, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                    kernel_initializer=initializer,
                                    use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                    kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)

    def _generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output)) # mean absolute error
        total_gen_loss = gan_loss + (self.lmbda * l1_loss)
        return total_gen_loss, gan_loss, l1_loss

    def _discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss

    def _construct_checkpoint_saver(self):
        print("...initalizing checkpoint saver")
        check_savr = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )
        return check_savr

    def fit(self, train_ds, epochs, epochs_per_save, test_ds, dinfo):
        pth_check, pth_rslts = dinfo.pth_chck , dinfo.pth_prog

        checkpoint_prefix = os.path.join(pth_check, "ckpt")
        log_dir = os.path.join(pth_check, "logs")
        print("logs will be saved to\n{}".format(log_dir))
        summary_writer = tf.summary.create_file_writer( os.path.join(log_dir, "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S") ) )

        @tf.function
        def train_step(input_image, target, epoch):
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                gen_output = self.generator(input_image, training=True)

                disc_real_output = self.discriminator([input_image, target], training=True)
                disc_generated_output = self.discriminator([input_image, gen_output], training=True)

                gen_total_loss, gen_gan_loss, gen_l1_loss = self._generator_loss(disc_generated_output, gen_output, target)
                disc_loss = self._discriminator_loss(disc_real_output, disc_generated_output)


            generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
            discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

            with summary_writer.as_default():
                tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
                tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
                tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
                tf.summary.scalar('disc_loss', disc_loss, step=epoch)

        for epoch in range(epochs):
            start = time.time()
            ipy.display.clear_output(wait=True)

            for example_input, example_target in test_ds.take(1):
                # Note: The training=True is intentional here since we want the batch statistics while running the model on the test dataset.
                # If we use training=False, we will get the accumulated statistics learned from the training dataset (which we don't want)
                prediction = self.generator(example_input, training=True)
                ImgUtil.plot_imgtens(example_input, example_target, prediction, os.path.join(pth_rslts, "{:04d}".format(epoch)) )

            # Train
            print("Epoch: ", epoch)
            for n, (input_image, target) in train_ds.enumerate():
                print('.', end='')
                if (n+1) % 100 == 0: print()
                train_step(input_image, target, epoch)
            print()

            # saving (checkpoint) the model every epochs_per_save epochs
            if (epoch + 1) % epochs_per_save == 0: self.checkpoint.save(file_prefix = checkpoint_prefix)
            print('Time taken for epoch {} is {} sec\n'.format(epoch+1, time.time()-start))

        self.checkpoint.save(file_prefix = checkpoint_prefix)
