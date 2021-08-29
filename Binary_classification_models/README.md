## Developing of Deep learning models for binary classification of x-ray-images for Covid-19 detection
### 1. Datasets
We have used a dataset containing X-ray images of patients undergoing two classes Covid-19 and healthy, the last labelled as “normal”, which can be accessed publicly on GitHub repository (Xu et al, 2020), and being accesible trhrough this lynk: 

Because these data were obtained from different sources, the picture resolutions differed from each other. Then, we rescaled the pictures to a same image. Because we used a CNN-based method, it was not affected by the adverse effects of this type of data compression. 

### 2. Convolutional Transfer learning neuronal networks based models
The pre-processed image data was split into training and validation sets (424 images Covid-19 and 425 images normal), and test sets (662 images), from which we have used the training and validation data to train and validate our models. 

Perfomance of all the generated models were  evaluated in a test set of 662 images. The notebooks related to this work are accesible through this [link](https://github.com/ChristianQF/SARSCov2)

According to our results, Xception outperformed the other four deep-learning models, achieving an overall classification accuracy of 100 %. This model is avialable through this [link](https://drive.google.com/file/d/1-6bnyEataVIl4WFxuwQG41GNVenMRBNj/view?usp=sharing)

### 3. Development of a support vector machine model for combining multiple deep learning models for detection and severity of Covid-19
For this purpose we have used a different dataset of 330 images obtaining probability values for their corresponding labels in a binary classification using the different binary classification deep learnig models obtained in a before step. For the same dataset was calculated different 18 features using torchxrayvision library for severity characterization of lungs. These values were grouped for each image. Then, the data was split into training, validation and test sets for the developing of a support vector machine (SVM) classification model which integrate the results of all deep learning models for detection and severity purpose, using the SVM implementations from scikit-learn library. 

The performance of the proposed model was then measured with the test dataset using standard metrics.
The notebooks related to this work are accesible through this [link](https://github.com/ChristianQF/SARSCov2)
