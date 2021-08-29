## Developing of Deep learning models for binary classification of x-ray-images for Covid-19 detection
### 1. Datasets
We have used a dataset containing X-ray images of patients undergoing two classes Covid-19 and healthy, the last labelled as “normal”, which can be accessed publicly on GitHub repository (Xu et al, 2020).
Because these data were obtained from different sources, the picture resolutions differed from each other. Then, we rescaled the pictures to a same image. Because we used a CNN-based method, it was not affected by the adverse effects of this type of data compression. 

![image](https://user-images.githubusercontent.com/79282400/131243302-f11d4f36-dc51-4bac-87a3-7a0229471abc.png)

### 2. Convolutional Transfer learning neuronal networks based models
The pre-processed image data was split into training and validation sets (424 images Covid-19 and 425 images normal), and test sets (662 images), from which we have used the training and validation data to train and validate our models. We have considered five widely used deep convolutional neuronal networks models, namely, Vgg16 (Simonyan & Zisserman,2015), DenseNet121 (Gottapu et al, 2018), InceptionV3 (Szegedy et al, 2015), Resnet50 (He  et al, 2016) and Xception (Chollet, 2017), as our transfer learning models in each case and keeping the rest of architecture equal. 

![image](https://user-images.githubusercontent.com/79282400/131243390-57cce1e3-2f55-43fb-a49f-869ece1ef97c.png)

Perfomance of all the generated models were  evaluated in a test set of 662 images. The notebooks related to this work are accesible through this [link](https://github.com/ChristianQF/SARSCov2)

According to our results, Xception outperformed the other four deep-learning models, achieving an overall classification accuracy of 100 %. This model is avialable through this [link](https://drive.google.com/file/d/1-6bnyEataVIl4WFxuwQG41GNVenMRBNj/view?usp=sharing)

### 3. Development of a support vector machine model for combining multiple deep learning models for detection and severity of Covid-19
For this purpose we have used a different dataset of 330 images ([link](https://drive.google.com/drive/folders/1-ciDsiTncjb0uZTLEJmktJnvFSwMdwjp?usp=sharing)) obtaining probability values for their corresponding labels in a binary classification using the different binary classification deep learnig models obtained in a before step. For the same dataset was calculated different 18 features using torchxrayvision library for severity characterization of lungs. These values were grouped for each image. Then, the data was split into training, validation and test sets for the developing of a support vector machine (SVM) classification model which integrate the results of all deep learning models for detection and severity purpose, using the SVM implementations from scikit-learn library. 

The performance of the proposed model was then measured with the test dataset using standard metrics.
The notebooks related to this work are accesible through this [link](https://github.com/ChristianQF/SARSCov2)

__References__

K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in Proceedings of the 3rd International Conference on Learning Representations (ICLR 2015), San Diego, CA, USA, September 2015.

R.D. Gottapu, and C.H. Dagli, “DenseNet for Anatomical Brain Segmentation”. Procedia Computer. Science, 2018. 140: p. 179-185

C. Szegedy, W. Liu, and Y. Jia et al., “Going deeper with convolutions,” in Proceedings of the 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1–9, Boston, MA, USA, June 2015. 

K. He, X. Zhang, S. Ren, and J. Sun. “Deep Residual Learning for Image Recognition”. IEEE Conference on Computer Vision and Pattern Recognition, 2016. pp. 770-778.

F. Chollet, “Xception: Deep Learning with Depthwise Separable Convolutions”. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 1800-1807.

