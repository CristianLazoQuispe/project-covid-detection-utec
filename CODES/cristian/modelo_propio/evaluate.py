# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../RESOURCES')

from loss_functions import * 
from metrics import * 

from tensorflow import keras


import os

WIDTH = 256
HEIGHT = 256


PATH_DATASET = '../DATASET/NEW_DATASET/MODELAMIENTO/PA'
EPOCHS = 1000

model_name = "../RESULTS/model_"+str(EPOCHS)+".h5"


batch_size = 64



model = keras.models.load_model(model_name, custom_objects={"custom_f1": custom_f1},compile=False)
model.summary()

import glob
import cv2
import matplotlib.pyplot as plt
#from tqdm.notebook import tqdm
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt     

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import numpy as np


classes_dict = {0:'COVID', 1:'NORMAL', 2: 'NO_COVID'}

import numpy as np

def get_data(path,tipo):
    predictions = []
    y_test = []
    
    for idx,clase in enumerate(['COVID','NORMAL','NO_COVID']):
        filename_list = glob.glob(os.path.join(path,tipo,clase,'*.jpg'))#[:10]
        for filename in tqdm(filename_list):
            #print(filename)
            
            img_rgb1 = plt.imread(filename)
            img_rgb = cv2.resize(img_rgb1,(WIDTH,HEIGHT))
            img_rgb = img_rgb/255.0
            img = np.reshape(img_rgb,(-1,HEIGHT, WIDTH,3))

            prediction = model.predict(img)            
            clase = prediction[0].argmax(axis=-1)
            
            #print(prediction[0],classes_dict[clase],prediction[0][clase]) #)
            #break
           
            predictions.append(clase)
            y_test.append(idx)
        
        #break
    predictions = np.array(predictions)
    y_test = np.array(y_test)
    
    predictions = np.reshape(predictions, (len(predictions),1))
    y_test = np.reshape(y_test, (len(predictions),1))
    #print('predictions',predictions)
    #print('y_test',y_test)
    #roc = roc_auc_score(predictions,y_test, multi_class='ovo')#, average='weighted')
    f1  = f1_score(y_test, predictions, average='micro')
    #print('roc',roc)
    print('f1',f1)

    #labels = ['COVID','NORMAL','NO_COVID']#
    labels = [0, 1,2]
    cm = confusion_matrix(y_test, predictions, labels)
    
    y_pred = predictions
    #importing accuracy_score, precision_score, recall_score, f1_score
    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

    print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
    print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
    print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))

    print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))

    print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
    print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
    print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))

    from sklearn.metrics import classification_report
    print('\nClassification Report\n')
    print(classification_report(y_test, y_pred, target_names=['COVID','NORMAL','NO_COVID']))

    fig_cm = plt.figure(figsize=(6,4))
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap="Blues");  #annot=True to annotate cells, ftm='g' to disable scientific notation
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title(tipo+" Confusion matrix - f1 "+str(np.round(f1,2))); 
    print("f1 "+str(np.round(f1,2)))
    ax.xaxis.set_ticklabels(['covid','normal','no_covid']); ax.yaxis.set_ticklabels(['covid','normal','no_covid']);
    

    fig_cm.savefig("../RESULTS/MODEL_"+str(EPOCHS)+"_f1_"+str(np.round(f1,4))+"_"+tipo+"_.png", dpi=fig_cm.dpi)

    return predictions,y_test,fig_cm,f1



predictions,y_test,fig_cm,f1 = get_data(PATH_DATASET,'TRAIN')

predictions,y_test,fig_cm,f1 = get_data(PATH_DATASET,'VALIDATION')
predictions,y_test,fig_cm,f1 = get_data(PATH_DATASET,'TEST')

