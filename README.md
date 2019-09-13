# [SIIM-ACR Pneumothorax Segmentation](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation) challenge

#### 9th place solution code

Description of the solution can be found [here](https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation/discussion/108060).


In order to reproduce the solution, you must perform the following steps:

### Train
#### 1) Train classification models
```
./classification/src/train.sh
```

#### 2) Train segmentation models
```
./segmentation/src/train_long.sh
``` 

### Inference
#### 1) Make predictions of classification models
```
./classification/src/perform_only_inference.sh
```
#### 2) Make predictions of first-stage Unet models, combine them with classification predictions and then make second-stage Unet models inference.
#### All this can be done with a single script
```
./segmentation/src/make_only_inference_long.sh
```
