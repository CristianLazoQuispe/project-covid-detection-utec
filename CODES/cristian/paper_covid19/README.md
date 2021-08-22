
# A modified deep convolutional neural network for detecting COVID-19 and pneumonia from chest X-ray images based on the concatenation of Xception and ResNet50V2

El COVID-19 se ha convertido en un grave problema sanitario en todo el mundo.  Se ha confirmado que este virus se ha cobrado más de 126.607 vidas hasta la fecha. Desde el inicio de su propagación, muchos investigadores de Inteligencia Artificial desarrollaron sistemas y métodos para predecir el comportamiento del virus o detectar la infección. Una de las posibles formas de determinar la infección del paciente a COVID-19 es a través del análisis de las imágenes de rayos X del tórax. Dado que hay un gran número de pacientes en los hospitales, sería largo y difícil examinar muchas imágenes de rayos X, por lo que puede ser muy útil desarrollar una red de IA que haga este trabajo automáticamente.  En este trabajo, hemos entrenado varias redes convolucionales profundas con las técnicas de entrenamiento introducidas para clasificar las imágenes de rayos X en tres clases: normal, neumonía y COVID-19, basándonos en dos conjuntos de datos de código abierto. Desafortunadamente, la mayoría de los trabajos anteriores sobre este tema no han compartido su conjunto de datos, y tuvimos que tratar con pocos datos sobre casos de COVID-19. Nuestros datos contienen 180 imágenes de rayos X que pertenecen a personas infectadas por COVID-19, por lo que intentamos aplicar métodos para conseguir los mejores resultados posibles. En esta investigación, introducimos algunas técnicas de entrenamiento que ayudan a la red a aprender mejor cuando el conjunto de datos está desequilibrado (tenemos pocos casos de COVID-19), y también proponemos una red neuronal que es una concatenación de las redes Xception y ResNet50V2. Esta red logró la mejor precisión al utilizar múltiples características extraídas por dos redes robustas. En este trabajo, a pesar de otras investigaciones, hemos probado nuestra red en 11302 imágenes para informar de la precisión real que puede alcanzar nuestra red en circunstancias reales. La precisión media de la red propuesta para detectar los casos de COVID-19 es del 99,56%, y la precisión media general para todas las clases es del 91,4%. Codigo del paper disponible en el siguiente link https://github.com/mr7495/covid19




Descargar los archivos 

1- https://github.com/ieee8023/covid-chestxray-dataset

2-https://www.kaggle.com/c/rsna-pneumonia-detection-challenge 


La red neuronal usa una contenacion de dos redes,  ResNet50V2 y Xception 

<p align="center">
	<img src="images/concatenated_net.png" alt="photo not available" width="100%" height="70%">
	<br>
	<em>arquitectura de la red neural</em>
</p>

