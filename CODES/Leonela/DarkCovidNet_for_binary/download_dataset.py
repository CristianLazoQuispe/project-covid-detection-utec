# -*- coding: utf-8 -*-
"""Download_dataset.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dud1prhYpF1fjmx9XDGC03-thOWN5PLI
"""

import os
import gdown
import zipfile

URL = 'https://drive.google.com/uc?id=1CmrRt4Uyl3lGTgYKMWkouzfGPU_W47Nj'
OUTPUT = '../DATASET/modelamiento.zip'
DATA_PATH = '../DATASET/'


if not os.path.exists(OUTPUT):
    print('download file')
    gdown.download(URL,OUTPUT, quiet=False)
else:
    print('modelamiento.zip already exists')

if not os.path.exists(os.path.join(DATA_PATH,'NEW_DATASET')):
    with zipfile.ZipFile(OUTPUT, 'r') as zip_ref:
        zip_ref.extractall(DATA_PATH)

print('Finished success')