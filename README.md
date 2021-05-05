#tripletpapertexture

This code repository contains code used to generate results in this paper --

L. Lackey, A. Grootveld, and A.G. Klein, “Semi-supervised convolutional triplet neural networks for assessing paper texture similarity,” in Proc. Asilomar Conf. on Signals, Systems, and Computers, Nov. 2020.

To fully reproduce the results, you will need to obtain a copy of the image data used in this project, consisting of 1597 greyscale PNG images in a "train" folder, and 420 greyscale PNG images in a "test" folder.  You may obtain access to the image files by emailing andy.klein@wwu.edu.  

Steps to reproduce the results:

1. First, clone this repository which should result in a directory with the following contents:<br>
    checkpoints/         -- empty directory, stores checkpoints during training<br>
    data/train/          -- directory containing 1597 training image files<br>
    data/test/           -- directory containing 420 test image files<br>
    figs/                -- empty directory, stores figures plotted during training<br>
    saved_model/         -- pre-trained model<br>
    README.md            -- this README<br>
    create_datafiles.py  -- Python script to generate pre-processed data files from raw images<br>
    triplet.ipynb        -- Main script<br>

2. Run create_datafiles.py to generate train.npz and test.npz.  You may delete the directories "train_tiled" and "test_tiled" which are created as part of this process.

3. Run triplet.ipynb which will produce the results the same trained model used in the paper.  To train your own model instead, set "TRAIN_MODE = True".  

Authors: L. Lackey, A. Grootveld, A.G. Klein
