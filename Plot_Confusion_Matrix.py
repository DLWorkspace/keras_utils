import numpy as np
from sklearn.metrics import *
from keras.callbacks import Callback
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
import os.path

from keras.applications.densenet import DenseNet121, DenseNet
from keras.applications.resnet50 import ResNet50

import tensorflow as tf
from keras.applications import Xception
from keras.utils import multi_gpu_model
from keras.callbacks import Callback
import sys, inspect
import matplotlib.pyplot as plt
import numpy as np

import itertools
from sklearn.metrics import *

class Plot_Confusion_Matrix(Callback):
    def __init__(self, validation_generator, output_path):
        self.my_val_benerator = validation_generator
        self.output_path = output_path
    def on_train_begin(self, logs={}):
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        res = self.model.predict_generator(self.my_val_benerator)

        pred= np.argmax(res,axis=1)

        tt = np.array([0]*7)
        tf = np.array([0]*7)
        ft = np.array([0]*7)
        ff = np.array([0]*7)

        for i in range(len(self.my_val_benerator.classes)):
            # print(str(pred[i])+"/"+str(self.my_val_benerator.classes[i]))
            for j in range(7):
                actc = str(j)
                if str(self.my_val_benerator.classes[i])==actc:
                    if str(pred[i]) == actc:
                        tt[j]+=1
                    else:
                        tf[j]+=1
                else:
                    if str(pred[i]) == actc:
                        ft[j]+=1
                    else:
                        ff[j]+=1
        # print (tt)
        # print (tf)
        # print (ft)
        # print (ff)
        # print (tt+tf+ft+ff)
        recall = (tt ) / ( tt + tf)
        precision = (tt) / (tt + ft)
        precision = np.nan_to_num(precision, copy = False)
        f1 = 2 / ( (1/recall) + (1 / precision))
        #print ("Recall === %s" % recall)
        #print ("Precision === %s" % precision)
        #print ("F1 score === %s" % f1)        
        mAP = np.average(precision)
        print ("mAP === %f" % mAP)  
        
        cnf_matrix = confusion_matrix(self.my_val_benerator.classes, pred)


        def plot_confusion_matrix(cm, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Blues,
                                  filename = None):
            """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            plot1 = plt.figure()
            im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
            title = title + "epoch:%d, mPA:%f" % (epoch, mAP)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            if filename is not None:
                plt.savefig(filename)
            plt.close()
        plot_confusion_matrix(cnf_matrix, self.my_val_benerator.classes,normalize=True,filename = os.path.join(self.output_path,"cm_%d_epoch.jpg"%(epoch)))        
        #plot_confusion_matrix(cnf_matrix, classes,normalize=False)
        
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return
