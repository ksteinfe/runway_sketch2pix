# Sketch2Pix
In summary, the following procedures and tools have been developed to support novice users in: 

1. Data Production
The production of data for training a Pix2Pix model. This step is supported by a collection of scripts written for various CAD software (Rhino and Blender) that extract pairs of training images from 3d models, and includes:
    * The structured documentation of alpha-masked architectural objects as RGBA images.
    * The processing of these photographs to allow students to create a sketched and alpha-masked version for each image appropriate for training.

2. Pix2Pix Model Training
The training and testing of a Pix2Pix model using this data.
This step is supported by a collection of Google Colab notebooks.

3. Sketching
Interfaces for interacting with this trained Pix2Pix model via digital sketching and masking.
This step is is supported by the Sketch2Pix Plugin for Photoshop


If you're new to Sketch2Pix, we recommend that you start with step (3) Sketching by following [these instructions for installing the Sketch2Pix Runway Model, and then the Sketch2Pix Photoshop Plugin](https://github.com/ksteinfe/runway_sketch2pix/blob/master/docs/getting_started_sketching.md). Together, these two pieces will allow you to run inferences on your sketches in Photoshop using one of the many existing "brushes" that have already been developed. 

Once you feel comfortable sketching, you may want to move on to training your own "brushes". This will require quite a bit more work in (1) Data Production and (2) Pix2Pix Model Training. We suggest first getting to know the model training process as described in [our page on training](https://github.com/ksteinfe/runway_sketch2pix/blob/master/docs/getting_started_training.md), before moving on to produce your own training data as described in [our page on creating training data sets](https://github.com/ksteinfe/runway_sketch2pix/blob/master/docs/getting_started_data.md).


