# project-covid-detection-utec

Modelo para detectar 3 clases:
- COVID    : persona portadora del SARS-CoV-2
- NORMAL   : persona sin ninguna enfermedad
- NO COVID : persona con alguna enfermedad pulmonar distinta al covid (Pneumonia)

Se uso transfer learning de 3 redes neuronales Xception, DenseNet121 y VGG16. Resultando el mejor modelo una red Xception
con una sensibilidad de 90%.




# Dataset
  - https://drive.google.com/file/d/1CmrRt4Uyl3lGTgYKMWkouzfGPU_W47Nj/view?usp=sharing


# Reproducir el codigo
    
    # install libraries
    
        $ pip install -r requirements.txt

    # Descargar la carpeta del dataset y descomprimirla automaticamente
    
        $ python download.py
    
    # Entrenar el modelo
    
        $ python training.py
    
    # Evaluar el modelo
    
        $ python evaluate.py
    