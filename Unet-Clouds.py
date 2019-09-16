#!/usr/bin/env python
# coding: utf-8

# Predicting Cloud Masks, Using a U-Net
import pandas as pd
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,CSVLogger
from time import gmtime, strftime
from eidetic_ml.readers import RLEImageMasksGenerator
from eidetic_ml.models import Unet
from eidetic_ml.metrics import dice


# Define the time for output files
date = strftime("%Y%m%d:%H:%M:%S", gmtime())


# Directories where things live
img_dir = "/home/james/GITHUB/Kaggle/data/Clouds/"
models_dir = img_dir + "models/"
train_dir = img_dir + "train/"
labels_file = img_dir + "train.csv"


# Dimensions of the input layer
height=256
width=256
channels=3

# How many images to work on at a time
batch_size = 12


# The labels are stored in a 204 MB csv file. 
labelsdf = pd.read_csv(labels_file)

#Drop rows that have no label.
labelsdf.dropna(inplace=True)


# The format of the labels is a little messy and hard to use programatically.
# The label is the form "filename_cloudtype".  It will be easier to examine the labels if it is reorganized.

new = labelsdf["Image_Label"].str.split("_", n = 1, expand = True) 
df = pd.DataFrame()
df['filename'] = new[0]
df['type'] = new[1]
df['EncodedPixels'] = labelsdf['EncodedPixels']
print(df.head())

# Instantiate the U-Net
shape=(height,width,channels)
unet = Unet(shape)
model = unet.build_model()
print(model.summary())


# Separate the filenames and labels into two numpy arrays
filenames = df['filename'].to_numpy()
rles = df['EncodedPixels'].to_numpy()



# Create an image reader object
train_gen = RLEImageMasksGenerator(width,height,batch_size,filenames,rles,train_dir)


# Setup the model checkpointing.
filepath=models_dir + date + "-weights.hdf5"

checkpoint = ModelCheckpoint(filepath, 
                             monitor="dice_coef",
                             verbose=1, 
                             save_best_only=True, 
                             mode='max')

# Log results of each epoch to a csv file 
csv_logger = CSVLogger(models_dir + date + '-training.log')



reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.005,
                              patience=6, min_lr=0.001)

callbacks_list = [checkpoint,reduce_lr,csv_logger]

model.compile(
            optimizer=Adam(1e-4), 
            loss='binary_crossentropy', 
            metrics=[dice.dice_coef]
            )


# Fit the U-Net, calling the DataGenerator and callbacks.
history = model.fit_generator(train_gen, 
                    steps_per_epoch=int(df.shape[0]/batch_size), 
                    epochs=600,
                    callbacks=callbacks_list,
                    verbose=1,
                    max_queue_size=1000,
                    use_multiprocessing=True,
                    workers=3,
                    shuffle=True
                   )



