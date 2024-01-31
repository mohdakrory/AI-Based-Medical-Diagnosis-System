# An AI-based application that offers medical diagnosis services such as Covid-19 detection from chest X-ray images, Brain tumor detection and segmentation from MRI images, Heartbeat abnormalities detection from ECG, Chest abnormalities detection and localization from chest X-ray images, and Skin cancer detection from skin images

## Caution

**This project is a prototype as a proof of concept and con't be used in real-world cases**

## Heartbeat abnormalities detection from ECG signal


Differentiate between Supraventricular premature beat, Premature ventricular contraction, Fusion of ventricular beat, and the normal case 

### Dataset description

[The MIT-BIH Arrhythmia Database](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) 

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/b2606578-813a-421c-a48f-ed5ac160bdbe)

N: Normal beat 

S: Supraventricular premature beat

V: Premature ventricular contraction

F: Fusion of ventricular beat

Q: Unclassifiable beat

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/98566aff-4822-4f31-829a-09a097fcf59c)

This Kaggle dataset was preprocessed and published by [1] 

### Proposed model

![model2](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/4d01f247-5cd4-4eeb-bf1f-47c542626840)

### Model performance

<table>
  <tr>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1</th>
    <th>AUC</th>
    <th>G-mean</th>
    <th>Kappa</th>
  </tr>
  <tr>
    <td>97.11%</td>
    <td>97.17%</td>
    <td>97.09%</td>
    <td>97.11%</td>
    <td>0.9960</td>
    <td>94.62%</td>
    <td>90.94%</td>
  </tr>
</table>

<table>
  <tr>
    <td>
      <img src="https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/921f10ed-c06c-4a3b-bde5-a2a54c5e0de3" alt="Image 1">
    </td>
    <td>
      <img src="https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/b0d74e90-6ce9-4126-a879-e4650cf9dfe6" alt="Image 2">
    </td>
  </tr>
</table>

![curves](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/f52c18c4-aa5b-41e5-85e6-22e9f3780f23)

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/9a0b69a4-05bd-4161-a83b-c80ce833863d)

### Results comparison with the state of the art

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/cfafdf02-f634-41ef-981d-91b8fedb7856)

### Hardware system for real-time analysis of ECG signal

#### Overall diagram

![diagram](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/59af1a62-55cd-421a-bc3d-8f7ed82d4705)

#### Flow diagram

![over all flow diagram](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/a464fea2-5454-4a0c-93f5-630483c80a2e)

#### Demo

![016](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/edee2c3f-750a-4ce6-a089-aa69935ad969)

You can find a video demo of this System [here](https://drive.google.com/file/d/1XEJtzw0dzgUyQ-4QIH5B-__w-oMspPG7/view?usp=sharing)

### Demo

![013](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/47500e6a-a202-4078-959f-4774a023af71)

### Kaggle notebook for model training

[Notebook](https://www.kaggle.com/code/mohamedeldakrory8/heartbeat-classification-from-ecg-graduation)

## Covid-19 detection from chest X-ray images 


Differentiate between Covid-19, Pneumonia, and the normal case from chest X-ray images

### Dataset description

[COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset](https://www.kaggle.com/datasets/amanullahasraf/covid19-pneumonia-normal-chest-xray-pa-dataset)

2313 samples for each class (Balanced) - 
6939 samples in total

![covid-dataset-samples](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/9f8e348d-a40b-457f-b023-77c6b8a7ce41)

### Proposed model

![VGG19+AFR](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/54d74ec0-8b0c-42d3-a933-e63de63da8be)

### Model performance

<table>
  <tr>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1</th>
    <th>AUC</th>
    <th>G-mean</th>
    <th>Kappa</th>
  </tr>
  <tr>
    <td>96.10%</td>
    <td>96.10%</td>
    <td>96.10%</td>
    <td>96.10%</td>
    <td>0.9836</td>
    <td>96.07%</td>
    <td>94.15%</td>
  </tr>
</table>

<table>
  <tr>
    <td>
      <img src="https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/65d0d2d2-1c8d-4dd6-aeae-e1b96122da3a" alt="Image 1">
    </td>
    <td>
      <img src="https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/da7c46d9-3d4f-44aa-9f90-fdc5d6229dc3" alt="Image 2">
    </td>
  </tr>
</table>

![curves](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/c1926be0-69b4-40fe-9006-d5e33e764303)

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/0b0674c2-d7dd-44a7-9dd6-660f23a696ac)

### Results comparison with the state of the art

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/c81ec6f8-a2d0-40dd-822e-4eb74635ac20)

### Demo

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/c8ed3e86-220d-4822-841e-a3d764f6244b)

### Kaggle notebook for model training

[Notebook](https://www.kaggle.com/code/mohamedeldakrory8/covid-19-chest-x-ray-graduation)

## Brain tumor detection from MRI images


***Differentiate between Meningioma, Glioma, Pituitary, and the normal case from MRI images***

### Dataset description

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

![brain-tumor-dataset-samples](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/6664e7c6-54e4-4ffd-866d-c4ae71d358b3)

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/4ab77fde-e07c-41d7-8009-cdb8613b9f55)
![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/10551bbf-85cb-4bff-b4ca-8d2810fdc124)

### Proposed model

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/ad1744f8-843e-4002-b7b0-e56a94b6f94a)

### Model performance

<table>
  <tr>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1</th>
    <th>AUC</th>
    <th>G-mean</th>
    <th>Kappa</th>
  </tr>
  <tr>
    <td>99.08%</td>
    <td>99.08%</td>
    <td>99.01%</td>
    <td>99.04%</td>
    <td>0.9998</td>
    <td>98.34%</td>
    <td>97.67%</td>
  </tr>
</table>

<table>
  <tr>
    <td>
      <img src="https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/f8f5e671-3c58-4962-906d-edfe7147c31e">
    </td>
    <td>
      <img src="https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/05d37008-566c-4c18-bf95-bf47b2912f3d">
    </td>
  </tr>
</table>

![curves](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/bae028d9-92a7-4d5e-9dfc-00c003a749fc)

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/a051a61f-c1b2-4ba7-8233-0447cf3061b1)

### Kaggle notebook for model training

[Notebook](https://www.kaggle.com/code/mohamedeldakrory8/brain-tumor-mri-classification-graduation/notebook)

## Brain tumor segmentation from MRI images 


### Dataset description

[Brain MRI segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/b43a3d14-52c4-468c-8cf8-6bb9f70999fa)

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/8af1bf0f-efef-4b8f-8ee5-b6f44566163f)

### Proposed Model

![model](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/d10a3c8f-7bc7-4b86-aee3-b0af326f0b6a)

Please note that this model was trained only on samples with tumors as we have another classification model with 100% recall on the normal class so it is very good at differentiating between the normal case and other cases

### Results comparison with the state-of-the-art

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/9ed1edbb-a4dc-4708-a3b6-850a603a9b93)

### Model Performance

![seg_curves](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/16fcbb7f-3991-4d44-af3b-99450f082d20)
![preds (7)](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/922903f4-a5a0-4a4b-87a8-91edd3d21b9b)
![preds (6)](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/ec676de2-8239-455e-ab9a-34ce43801102)
![preds (5)](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/629cfbec-3137-48df-8d74-f5eef76b8425)
![preds (4)](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/dbeb2576-9a8e-489b-843f-6359c79c53de)
![preds (3)](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/93a7f73b-997f-42e6-ba00-4384ce0817d3)
![preds (2)](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/ebccaf53-b285-4c1f-ab21-06cdd40e5a7f)
![preds (1)](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/f51683fb-db02-4a97-b000-709f27d811d9)
![preds](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/d2307f0a-4d3e-4da4-ac30-22fdd6451af0)

### Demo

![005](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/0d66b8cb-19ea-4770-a14c-738b1e61d30a)

### Kaggle notebook for model training

[Notebook](https://www.kaggle.com/code/karimelsheery/brain-mri-segmentation/notebook)

## Skin cancer detection from skin images 


Differentiate between Malignant and Benign skin cancer from skin images 

### Dataset description

[Melanoma Skin Cancer Dataset of 10000 Images](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images)

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/bc0e04ee-9fae-4768-abdc-7e3f2ebd58d6)

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/1ac8c2fe-652d-4395-b389-de740bf79af0)

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/4f241938-d1ac-4823-a571-cf2e8b762dcf)

### Proposed model

![model](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/d5c92c86-abf0-4081-a95a-be67ddbe2c35)

### Model Performance

<table>
  <tr>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1</th>
    <th>AUC</th>
    <th>G-mean</th>
    <th>Kappa</th>
  </tr>
  <tr>
    <td>92.10%</td>
    <td>93.58%</td>
    <td>90.40%</td>
    <td>92.10%</td>
    <td>0.9667</td>
    <td>92.08%</td>
    <td>84.20%</td>
  </tr>
</table>

<table>
  <tr>
    <td>
      <img src="https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/5cd5d1c3-3695-4bfd-bffa-97a1dba7078a">
    </td>
    <td>
      <img src="https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/79945b0f-564b-46ba-acc1-c3943c11f7cc">
    </td>
  </tr>
</table>

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/48f0cd6c-1e0c-4058-a1bd-11edee0240a0)

![curves](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/1ddea717-fc15-4d4c-ab9d-0885b0a73d85)

### Results comparison with the state of the art

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/2c7aeff0-a372-4ed8-8984-9374b1069553)

### Demo

![017](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/e54e513d-f200-4754-b6c2-1fcb71b2a57e)


### Kaggle notebook for model training

[Notebook](https://www.kaggle.com/code/mohamedeldakrory8/skin-cancer-detection-graduation/notebook)

## chest X-ray abnormalities detection and localization 


Differentiate between 14 chest abnormalities from chest X-ray images

### Dataset description

[VinBigData Chest X-ray Abnormalities Detection](https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection)

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/c4fc7c6e-f7ed-4443-b1c7-06b4d28e5978)

<table>
  <tr>
    <th>Class name</th>
    <th>Class ID</th>
    <th>Class frequency</th>
  </tr>
  <tr>
    <td>Aortic enlargement</td>
    <td>0</td>
    <td>7162</td>
  </tr>
  <tr>
    <td>Atelectasis</td>
    <td>1</td>
    <td>279</td>
  </tr>
  <tr>
    <td>Calcification</td>
    <td>2</td>
    <td>960</td>
  </tr>
  <tr>
    <td>Cardiomegaly</td>
    <td>3</td>
    <td>5427</td>
  </tr>
  <tr>
    <td>Consolidation</td>
    <td>4</td>
    <td>556</td>
  </tr>
  <tr>
    <td>Interstitial lung disease (ILD)</td>
    <td>5</td>
    <td>1000</td>
  </tr>
  <tr>
    <td>Infiltration</td>
    <td>6</td>
    <td>1247</td>
  </tr>
  <tr>
    <td>Lung Opacity</td>
    <td>7</td>
    <td>2483</td>
  </tr>
  <tr>
    <td>Nodule/Mass</td>
    <td>8</td>
    <td>2580</td>
  </tr>
  <tr>
    <td>Other lesion</td>
    <td>9</td>
    <td>2203</td>
  </tr>
  <tr>
    <td>Pleural effusion</td>
    <td>10</td>
    <td>2476</td>
  </tr>
  <tr>
    <td>Pleural thickening</td>
    <td>11</td>
    <td>4842</td>
  </tr>
  <tr>
    <td>Pneumothorax</td>
    <td>12</td>
    <td>226</td>
  </tr>
  <tr>
    <td>Pulmonary fibrosis</td>
    <td>13</td>
    <td>4655</td>
  </tr>
  <tr>
    <td>No finding</td>
    <td>14</td>
    <td>10,000</td>
  </tr>
</table>

### Yolo V5 versions

![yolo v5 versions](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/760d7e3c-95da-4519-9e07-44c974abd7c0)

Yolo v5x was utilized 

### Model performance

Mean average precision score (mAP0.5): 0.312

![training curves](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/ad3ebe65-beb9-4c5f-8da6-ab79e24be4a3)

![test samples](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/ff721315-3ea3-43c2-b009-cb0815ddfeb9)

Predictions (Left) VS Actual (Right) 

### Results comparison with the state of the art

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/c9d62d97-7e3c-4ee7-9445-396f458b78d6)

### Demo

![015](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/a9547feb-776b-40ca-87dc-8f31505e1baa)

### Kaggle notebook for model training

[Notebook](https://www.kaggle.com/code/mohamedeldakrory8/vinbigdata-cxr-ad-yolov5)

### For more details about experiments on each service, method validation, and online testing please check the Documentation/MDS.pdf file

## References


<table>
  <tr>
    <td>
      [1]
    </td>
    <td>
      M. Kachuee, S. Fazeli, and M. Sarrafzadeh, “Ecg heartbeat classification: A deep transferable representation,” in 2018 IEEE International Conference on Healthcare Informatics (ICHI), 2018, pp. 443–444.
    </td>
  </tr>

  <tr>
    <td>
      [2]
    </td>
    <td>
      Pirova, D., Zaberzhinsky, B., & Mashkov, A. (2020). Detecting heart 2 disease symptoms using machine learning methods. Proceedings of the Information Technology and Nanotechnology (ITNT-2020), 2667, 260-263.
    </td>
  </tr>

  <tr>
    <td>
      [3]
    </td>
    <td>
      Walsh, P. (2019). Support Vector Machine Learning for ECG Classification. In CERC (pp. 195-204).
    </td>
  </tr>

  <tr>
    <td>
      [4]
    </td>
    <td>
      W. Wang, S. Liu, H. Xu, and L. Deng, “COVIDX-LwNet: 	A lightweight network ensemble model for the detection of COVID-19 based on chest X-ray images,” Sensors (Basel), vol. 22, no. 21, p. 8578, 	2022.
    </td>
  </tr>

  <tr>
    <td>
      [5]
    </td>
    <td>
      E. Cengil and A. Çınar, “The effect of deep feature	concatenation in the classification problem: An approach on COVID-19 disease detection,” Int. J. 	Imaging Syst. Technol., vol. 32, no. 1, pp. 26–40, 	2022.
    </td>
  </tr>

  <tr>
    <td>
      [6]
    </td>
    <td>
      M. Buda, A. Saha, and M. A. Mazurowski, “Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm,” Comput. Biol. Med., vol. 109, pp. 218–225, 2019.
    </td>
  </tr>

  <tr>
    <td>
      [7]
    </td>
    <td>
      L. Yi, J. Zhang, R. Zhang, J. Shi, G. Wang, and X. Liu, “SU-net: An efficient encoder-decoder model of federated learning for brain tumor segmentation,” in Artificial Neural Networks and Machine Learning 	– ICANN 2020, Cham: Springer International Publishing, 2020, pp. 761–773.
    </td>
  </tr>

  <tr>
    <td>
      [8]
    </td>
    <td>
      A. A, “A deep learning approach to skin cancer detection in dermoscopy images,” J. Biomed. Phys. Eng., vol. 10, no. 6, 	pp. 801–806, 2020.
    </td>
  </tr>

  <tr>
    <td>
      [9]
    </td>
    <td>
      M. Upadhyay, J. Rawat, and S. Maji, “Skin cancer image classification	using deep neural network models,” in Evolution in Computational 	Intelligence, 	Singapore: Springer Nature Singapore, 2022, pp. 	451–460.
    </td>
  </tr>

  <tr>
    <td>
      [10]
    </td>
    <td>
      Talius, E., & Sayana, R. Zero-Shot Object Detection for Chest X-Rays.
    </td>
  </tr>

  <tr>
    <td>
      [11]
    </td>
    <td>
      V. Parikh, J. Shah, C. Bhatt, J. M. Corchado, and D.-N. Le, “Deep learning based automated chest X-ray abnormalities detection,” in Lecture Notes in 	Networks and Systems, Cham: Springer International Publishing, 2023, pp. 1–12.
    </td>
  </tr>
</table>

