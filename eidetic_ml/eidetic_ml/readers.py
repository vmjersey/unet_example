import cv2
import numpy as np
import pandas as pd
import keras
from eidetic_ml.processing import Encoding


class RLEImageMasksGenerator(keras.utils.Sequence):

    """A Mask Generator that uses the Keras Utils Sequence Object.
    Parameters
    ----------
    width : int, required
        A number representing the width of the image.
    height: int, required
        A number representing the height of the image.
    batch_size : int, required
        A number representing the number of masks to 
        return to the training function for each step.
    filenames : Numpy Array, required
        A list of files to be read over the course of one epoch.
    rles : Numpy Array, required.
        A list of things containing the rles.
    directory: str, required.
        A string containing the directory where images exist
    """


    def __init__(self,width,height,batch_size,filenames,rles,image_dir):

        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.filenames = filenames
        self.rles = rles
        self.image_dir = image_dir
        self.on_epoch_end()
    
    def __len__(self):
        return int(self.filenames.shape[0]/self.batch_size)
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(0,self.filenames.shape[0])

    def __getitem__(self,index):
        
        start = index*self.batch_size
        end = (index+1)*self.batch_size

        image_placeholders = np.arange(start,end)
        
        x = np.array([])
        x = np.zeros((len(image_placeholders),self.width,self.height,3))

        y = np.array([])
        y = np.zeros((len(image_placeholders),self.width,self.height,1))


        counter = 0
        for i in image_placeholders:
            
            image = cv2.imread(self.image_dir+self.filenames[i],1)
 
            # Create Mask
            mask = Encoding.decode_rle_mask(self.rles[i],image.shape[0],image.shape[1])
            
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            # Resize Images
            image = cv2.resize(image, (self.height,self.width))
            mask  = cv2.resize(mask,  (self.height,self.width))
            
            mask = np.resize(mask,(self.width,self.height,1))
            
            x[counter] = image/255
            y[counter] = mask/255
            
            counter+=1
            
        return x,y        



