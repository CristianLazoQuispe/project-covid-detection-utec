import os
import gdown
import zipfile

#url = 'https://drive.google.com/file/d/1CmrRt4Uyl3lGTgYKMWkouzfGPU_W47Nj/view?usp=sharing'
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