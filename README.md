#tripletpapertexture

This code repository contains code used to generate results in this paper --

L. Lackey, A. Grootveld, and A.G. Klein, “Semi-supervised convolutional triplet neural networks for assessing paper texture similarity,” in Proc. Asilomar Conf. on Signals, Systems, and Computers, Nov. 2020.

To fully reproduce the results, you will need to obtain a copy of the image data used in this project, consisting of 1597 greyscale PNG images in a "train" folder, and 420 greyscale PNG images in a "test" folder.  You may obtain access to the image files by emailing andy.klein@wwu.edu.  

Steps to reproduce the results:

1. First, create a directory with the following contents:
    ./checkpoints        -- empty directory, stores checkpoints during training
    ./data/train         -- contains 1597 training image files
    ./data/test          -- contains 420 test image files
    ./figs               -- empty directory, stores figures plotted during training
    ./saved_model        -- pre-trained model
    create_datafiles.py  -- Python script to generate pre-processed images from raw images, stored as train.npz and test.npz
    triplet.ipynb        -- Main script

2. Run create_datafiles.py to generate train.npz and test.npz.  You may delete the directories "train_tiled" and "test_tiled" which are created as part of this process.

3. Run triplet.ipynb which will produce the results the same trained model used in the paper.  To train your own model instead, set "TRAIN_MODE = True".  

Authors: L. Lackey, A. Grootveld, A.G. Klein
