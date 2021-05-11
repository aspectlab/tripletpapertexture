#!/usr/bin/env python
# coding: utf-8

# # Triplet Network for Paper Texture Similarity

# L. Lackey, A. Grootveld, and A. G. Klein, "Semi-supervised convolutional triplet neural networks for assessing paper texture similarity," in *54th Asilomar Conference on Signals, Systems and Computers*, Pacific Grove, CA, Nov. 2020
# 

# ## Import necessary libraries

# In[ ]:


import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.metrics.pairwise import euclidean_distances
from tabulate import tabulate
from IPython.display import clear_output


# ## Tweakable parameters

# In[ ]:


TRAIN_MODE = False                 # set to True if we need to train, set to False to load pre-trained model
TRAINDATA_FILE = 'data/train.npz'  # file containing training and validation data set
TESTDATA_FILE =  'data/test.npz'   # file containing test data set
MODEL_FILENAME = 'asil20_weights'  # file to load pre-trained model (if TRAIN_MODE=False)
EMB_SIZE = 16                      # num elements in feature vector / embedding output of NN
BATCH_SIZE = 512                   # size of each batch
EPOCHS = 400                       # number of epochs to run
ALPHA = 0.5                        # triplet loss parameter
L2NORM = True                      # indicates whether embeddings are L2 normalized to hypersphere
PATIENCE = 50                      # training stop if this number of epochs go by without improvement
JUPYTER = False                    # true for live loss plotting (.ipynb), false to save plots to files (.py)


# ## Define functions to assist with triplet data generation and triplet loss

# In[ ]:


# function to create a batch of triplets
def create_batch(x, y, batch_size=256):
    x_anchors = np.zeros((batch_size, x.shape[1], x.shape[1],1))
    x_positives = np.zeros((batch_size, x.shape[1], x.shape[1],1))
    x_negatives = np.zeros((batch_size, x.shape[1], x.shape[1],1))
    
    for i in range(0, batch_size):
        # Pick a random anchor
        random_index = random.randint(0, x.shape[0] - 1)  # pick a random index for anchor
        x_anchors[i] = x[random_index]                    # grab anchor image
        
        # grab indices of all images in same class, and those not in same class
        indices_for_pos = np.squeeze(np.where(y == y[random_index]))
        indices_for_neg = np.squeeze(np.where(y != y[random_index]))
        
        # pick random indices for other tile of same class (positive) and one not in same class (negative)
        x_positives[i] = x[indices_for_pos[random.randint(0, len(indices_for_pos) - 1)]]
        x_negatives[i] = x[indices_for_neg[random.randint(0, len(indices_for_neg) - 1)]]
    
    return [x_anchors, x_positives, x_negatives]


# data generator functions
def data_generator(x, y, batch_size=256):
    while True:
        xx = create_batch(x, y, batch_size)
        yy = np.zeros((batch_size, 3*EMB_SIZE))  # since loss function doesn't depend on classes, set to zero
        yield xx, yy


# triplet loss function
def triplet_loss(y_true, y_pred):
    anchor, positive, negative = y_pred[:,:EMB_SIZE], y_pred[:,EMB_SIZE:2*EMB_SIZE], y_pred[:,2*EMB_SIZE:]
    positive_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    return tf.maximum(positive_dist - negative_dist + ALPHA, 0.)


# ## Define functions to assist with metrics computation

# In[ ]:


def compute_image_distances(embs, tiles_per_image):
    # embs contains the features for each tile, tile_per_image contains the # of tiles per image in the embs
    
    # define some intermediate variables
    tile_distances = euclidean_distances(embs)
    num_classes = len(embs)//tiles_per_image
    
    # calculate per-image distances (approach #1) using sum of total per-tile distances
    img_distances1 = tile_distances.reshape(num_classes,tiles_per_image,num_classes,tiles_per_image).sum(axis=(1,3))  
    img_distances1 = img_distances1 + np.diag(np.diag(img_distances1))/(tiles_per_image-1) # correct for fact that block diagonals add up TILES_PER_IMAGE fewer distances
    
    # calculate per-image distances (approach #2) using euclidean difference of centroids
    centroids = np.zeros((num_classes, EMB_SIZE))
    for i in range(num_classes):
        centroids[i]=embs[range(i*tiles_per_image, (i+1)*tiles_per_image),:].mean(axis=0)/2
    img_distances2 = euclidean_distances(centroids)
    
    return [img_distances1, img_distances2, centroids]

def mean_self_similarity_rank(D):
    vals = np.zeros(D.shape[0])
    for i in range(D.shape[0]):
        vals[i] = np.where(np.argsort(D[:,i]) == i)[0].item()
    return np.mean(vals)/(D.shape[0]-1)  # number between 0 (good) and 1 (bad)

def retrieval_metrics(D, grps, epg=10, digits=2):
    pat1=0
    mrr=0
    map=0
    for k in range(grps):  # loop over similarity groups
        for i in range(epg*k,epg*(k+1)):  # loop over each element in similarity group
            dist = np.delete(np.stack([D[i,:], np.arange(0,D.shape[0])],axis=1),i,axis=0)
            T = dist[np.argsort(dist[:,0]),1]
            g = np.nonzero(np.logical_and(T>=epg*k, T<epg*(k+1)))[0]    # get rank of similar images (i.e. those with same label)
            pat1 = pat1+(g[0]==0)/grps/epg;
            mrr = mrr+1/(1+g[0])/grps/epg;
            map = map+np.mean(np.arange(1,epg)/(g+1))/grps/epg;
    
    return [round(pat1*100,digits), round(mrr*100,digits), round(map*100,digits)]


# ## Load training and validation data

# In[ ]:


npzfile = np.load(TRAINDATA_FILE)
x_train = npzfile['x_train']
y_train = npzfile['y_train']
fnames_train = npzfile['fnames_train']  # filenames corresponding to every tile in x_train (not used below, but helpful for debug)
classnames_train = npzfile['classnames_train']  # ordered image names (which subsequently appear in img_distanceX matrices, helpful for debug)
x_valid = npzfile['x_valid']
y_valid = npzfile['y_valid']
tilesize = x_train.shape[1]  # input images presumed to be square, have this number of pixels per side 
tiles_per_image = x_train.shape[0]//classnames_train.shape[0]  # number of tiles per image (assumed adjacent in data set)
valid_tiles_per_image = x_valid.shape[0]//classnames_train.shape[0]  # number of tiles per image in validation set
x_train=x_train[:,:,:,np.newaxis] # add dimension
x_valid=x_valid[:,:,:,np.newaxis]


# ## Build network for multi-GPU rig

# ### Callbacks

# In[ ]:


# Callback to compute each epoch on validation data
class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data, valid_tiles_per_image, embmod):
        super(Metrics, self).__init__()
        self.x, _ = valid_data
        self.valid_tiles_per_image = valid_tiles_per_image
        self.embmod=embmod
    
    def on_epoch_end(self, epoch, logs):
        # compute mean self-similarity score on validation set
        embs_val = self.embmod.predict(self.x)
        [img_distances1, _, _] = compute_image_distances(embs_val, self.valid_tiles_per_image)
        mssr = mean_self_similarity_rank(img_distances1)
        logs['val_mssr'] = mssr
        
        return

# Callback to live plot losses (in Jupyter) or save to file (in vanilla Python)
class Plotter(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.loss = []
        self.val_loss = []
        self.val_mssr = []
    def on_epoch_end(self, epoch, logs=None):
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_mssr.append(logs.get('val_mssr'))
        clear_output(wait=True)
        plt.figure(figsize=(16, 5))
        plt.semilogy(np.arange(1, epoch+2), self.val_mssr,'o-')
        plt.semilogy(np.arange(1, epoch+2), self.val_loss,'o-')
        plt.semilogy(np.arange(1, epoch+2), self.loss,'o-')
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('loss at each epoch')
        plt.legend(['validation mssr', 'validation loss', 'training loss'], loc='upper left')
        if JUPYTER:
            plt.show()
        else:
            plt.savefig('figs/{:04d}.png'.format(epoch+1))
            plt.close()


# ### Model description

# In[ ]:


# create multi-GPU setup
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    # declare neural network architecture (4-layer conv net, with fully connected layer at output)
    input_img = tf.keras.layers.Input(shape=(tilesize, tilesize,1))
    next = tf.keras.layers.Conv2D(16,(5, 5), activation='relu', padding='same')(input_img)
    next = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(next)
    next = tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same')(next)
    next = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(next)
    next = tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same')(next)
    next = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(next)
    next = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(next)
    next = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(next)
    next = tf.keras.layers.Flatten()(next)
    embedding_out_layer = tf.keras.layers.Dense(EMB_SIZE)(next)
    model = tf.keras.Model(inputs=input_img, outputs=embedding_out_layer)
    model.summary()
    
    if L2NORM:
        embedding_model = tf.keras.Model(inputs=model.layers[0].input, outputs=tf.keras.layers.Lambda(lambda temp: tf.keras.backend.l2_normalize(temp,axis=1))(embedding_out_layer))
    else:
        embedding_model = tf.keras.Model(inputs=model.layers[0].input, outputs=embedding_out_layer)
    embedding_model.summary()
    
    # build the Siamese network
    input_anchor = tf.keras.layers.Input(shape=(tilesize,tilesize,1))
    input_positive = tf.keras.layers.Input(shape=(tilesize,tilesize,1))
    input_negative = tf.keras.layers.Input(shape=(tilesize,tilesize,1))
    embedding_anchor = embedding_model(input_anchor)
    embedding_positive = embedding_model(input_positive)
    embedding_negative = embedding_model(input_negative)
    output = tf.keras.layers.concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)
    net = tf.keras.models.Model([input_anchor, input_positive, input_negative], output)
    net.summary()
    net.compile(loss=triplet_loss, optimizer='adam')


# ## Train network, or load pre-trained model

# In[ ]:


if TRAIN_MODE:
    steps_per_epoch = int(x_train.shape[0]/BATCH_SIZE)
    if x_valid.shape[0]==0:  # data set contains no validation data
        validation_steps = None
        validation_data = None
    else:                    # data set DOES contain validation data, set appropr
        validation_steps = int(x_valid.shape[0]/BATCH_SIZE)
        validation_data=data_generator(x_valid,y_valid,BATCH_SIZE)
    history = net.fit(
        data_generator(x_train,y_train,BATCH_SIZE),
        epochs=EPOCHS,
        callbacks=[Metrics((x_valid, y_valid),valid_tiles_per_image,embedding_model), 
                   Plotter(),
                   tf.keras.callbacks.ModelCheckpoint(filepath='checkpoints/weights-{epoch:04d}.hdf5', save_weights_only=True, save_best_only=True, monitor='val_mssr'),
                   tf.keras.callbacks.EarlyStopping(monitor="val_mssr", patience=PATIENCE, mode="min")],
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_data, validation_steps=validation_steps
    )
else:
    embedding_model.load_weights(MODEL_FILENAME+'.hdf5')


# ## Compute training set tile features on final/trained network. Use tile features to compute training set image distances using two methods. Report mean self-similarity rank on training set (lower is better)

# In[ ]:


embs = embedding_model.predict(x_train)
[img_distances1, _, _] = compute_image_distances(embs, tiles_per_image)
print('mean self-similarity rank: '+str("{:.2e}".format(mean_self_similarity_rank(img_distances1))))


# <hr style="height:3px;border:none;background-color:#DC143C" />
# 
# 
# ## Load the test set, report results including comparison with prior approaches.

# In[ ]:


# import test tile data
npzfile_test = np.load(TESTDATA_FILE)
x_test = npzfile_test['x_test']
y_test = npzfile_test['y_test']
fnames_test = npzfile_test['fnames_test']          # filenames corresponding to every tile in x_test (not used below, but helpful for debug)
classnames_test = npzfile_test['classnames_test']  # ordered image names (which subsequently appear in img_distanceX matrices, helpful for debug)
test_tiles_per_image = x_test.shape[0]//classnames_test.shape[0]  # number of tiles per image (assumed adjacent in test set)

# compute image distances on test set
embs_test = embedding_model.predict(x_test)
[_, img_distances2_test, centroids] = compute_image_distances(embs_test, test_tiles_per_image)

# split test set into paper types: silver gelatin, inkjet, wove (recto), wove (verso)
sg=img_distances2_test[120:240, 120:240]
ij=img_distances2_test[0:120, 0:120]
wv_r=img_distances2_test[240:420:2, 240:420:2]
wv_v=img_distances2_test[241:420:2,241:420:2]

# spit out results with some description
print('Results are reported for comparison with these two publications which can be referred to for an explanation of metrics:')
print('')
print('[1] Y. Zhai and D. L. Neuhoff, "Photographic paper classification via local radius index metric," 2015 IEEE International Conference on Image Processing (ICIP), Quebec City, QC, 2015, pp. 1439-1443, doi: 10.1109/ICIP.2015.7351038.')
print('')
print('[2] P. Abry et al., "Wove paper analysis through texture similarities," 2016 50th Asilomar Conference on Signals, Systems and Computers, Pacific Grove, CA, 2016, pp. 144-148, doi: 10.1109/ACSSC.2016.7869012.')
print('')
print('--------------------------------------------------------------------------------------------------')
print('')

results = [retrieval_metrics(sg,9), retrieval_metrics(ij,9), retrieval_metrics(sg[0:30,0:30],3), retrieval_metrics(sg[30:60,30:60],3), retrieval_metrics(wv_r,6), retrieval_metrics(wv_v,6), retrieval_metrics(wv_r+wv_v,6)]
l = [["silver gelatin", "[1, Table 3]", results[0][0], results[0][1], results[0][2]],
     ["silver gelatin classes 123", "[1, Table 4]", results[2][0], results[2][1], results[2][2]],
     ["silver gelatin classes 456", "[1, Table 4]", results[3][0], results[3][1], results[3][2]],
     ["inkjet", "[1, Table 3]", results[1][0], results[1][1], results[1][2]],
     ["wove (recto)", "[2, Table 1]", results[4][0], results[4][1], results[4][2]],
     ["wove (verso)", "[2, Table 1]", results[5][0], results[5][1], results[5][2]],
     ["wove (sum)", "N/A", results[6][0], results[6][1], results[6][2]]]
table = tabulate(l, headers=['Description', 'Reference', 'P@1','MRR','MAP'], tablefmt='orgtbl')
print(table)

print('--------------------------------------------------------------------------------------------------')
print('Sum of all 7x3=21 above metrics:', np.sum(results), 'out of 2100 possible')
print('Same, but normalized by 2100 and subtracted from 1 (so it''s kinda like a loss):', 1-np.sum(results)/2100)

# plot distance matrices reported in [1]
print('Compare these figures with [1, Fig 3]')
plt.imshow(ij, cmap='gray', interpolation='nearest')
plt.title('inkjet')
if JUPYTER:
    plt.show(block=False)
else:
    plt.savefig('figs/inkjet.png')
    plt.close()

plt.imshow(sg, cmap='gray', interpolation='nearest')
plt.title('silver gelatin b&w')
if JUPYTER:
    plt.show(block=False)
else:
    plt.savefig('figs/silver_gel.png')
    plt.close()

# plot distance matrices reported in [2]
print('Compare these figures with [2, Fig 2]')
plt.imshow(wv_r, cmap='gray', interpolation='nearest')
plt.title('wove, recto')
if JUPYTER:
    plt.show(block=False)
else:
    plt.savefig('figs/wove_recto.png')
    plt.close()

plt.imshow(wv_v, cmap='gray', interpolation='nearest')
plt.title('wove, verso')
if JUPYTER:
    plt.show(block=False)
else:
    plt.savefig('figs/wove_verso.png')
    plt.close()


# ## Save test set distances and features to Matlab file

# In[ ]:


# Save these variables to Matlab:
#     D -- distance matrix ( N x N )
#     f -- feature vector ( N x EMB_SIZE )
#     fnames -- cell array string of filenames ( N x 1 )
sio.savemat('output.mat', {'D':img_distances2_test, 'fnames':np.asarray(classnames_test, dtype='object'), 'f':centroids}, oned_as='column')

