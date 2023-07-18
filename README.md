# AI-Based-Medical-Diagnosis-System
An AI-based application that offers medical diagnosis services such as Covid-19 detection from chest X-ray images, Brain tumor detection and segmentation from MRI images, Heartbeat abnormalities detection from ECG, Chest abnormalities detection and localization from chest X-ray images, and Skin cancer detection for skin images

Heartbeat abnormalities detection from ECG signal
--

Differentiate between Supraventricular premature beat, Premature ventricular contraction, Fusion of ventricular beat, and the normal case 

***Dataset description***

[The MIT-BIH Arrhythmia Database](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) 

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/b2606578-813a-421c-a48f-ed5ac160bdbe)

N: Normal beat 

S: Supraventricular premature beat

V: Premature ventricular contraction

F: Fusion of ventricular beat

Q: Unclassifiable beat

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/98566aff-4822-4f31-829a-09a097fcf59c)

This Kaggle dataset was preprocessed and published by [1] 

***Proposed model***

![model2](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/4d01f247-5cd4-4eeb-bf1f-47c542626840)

***Model performance*** 

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

***Results comparison with the state of the art***

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/cb0dc9d5-3965-4ed6-a825-b60d5dbafd07)

***Hardware system for real-time analysis of ECG signal***

![diagram](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/59af1a62-55cd-421a-bc3d-8f7ed82d4705)

***Demo***

![013](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/47500e6a-a202-4078-959f-4774a023af71)

***Kaggle training notebook***

[Notebook](https://www.kaggle.com/code/mohamedeldakrory8/heartbeat-classification-from-ecg-graduation)

Covid-19 detection from chest X-ray images 
--

Differentiate between Covid-19, Pneumonia, and the normal case from chest X-ray images

***Dataset description***

[COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset](https://www.kaggle.com/datasets/amanullahasraf/covid19-pneumonia-normal-chest-xray-pa-dataset)

2313 samples for each class (Balanced) - 
6939 samples in total

![covid-dataset-samples](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/9f8e348d-a40b-457f-b023-77c6b8a7ce41)

***Proposed model***

![VGG19+AFR](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/54d74ec0-8b0c-42d3-a933-e63de63da8be)

***Model performance*** 

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

***Results comparison with the state of the art***

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/c81ec6f8-a2d0-40dd-822e-4eb74635ac20)

***Demo***

![010](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/015fbf42-3319-4a12-8055-ffd48b72e709)

***Kaggle training notebook***

[Notebook](https://www.kaggle.com/code/mohamedeldakrory8/covid-19-chest-x-ray-graduation)

Brain tumor detection from MRI images
--

***Dataset description***

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/4ab77fde-e07c-41d7-8009-cdb8613b9f55)
![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/10551bbf-85cb-4bff-b4ca-8d2810fdc124)

***Proposed model***

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/ad1744f8-843e-4002-b7b0-e56a94b6f94a)

***Model performance***

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

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/a051a61f-c1b2-4ba7-8233-0447cf3061b1)

***Kaggle training notebook***

[Notebook](https://www.kaggle.com/code/mohamedeldakrory8/brain-tumor-mri-classification-graduation/notebook)

Reference
--

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
  
</table>
