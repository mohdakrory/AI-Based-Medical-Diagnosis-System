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

Covid-19 detection from chest x-ray images 


References
--

[1]  M. Kachuee, S. Fazeli, and M. Sarrafzadeh, “Ecg heartbeat classification: A deep transferable representation,” in 2018 IEEE International Conference on               Healthcare Informatics (ICHI), 2018, pp. 443–444.

[2]   Pirova, D., Zaberzhinsky, B., & Mashkov, A. (2020). Detecting heart 2 disease symptoms using machine learning methods. Proceedings of the Information                 Technology and Nanotechnology (ITNT-2020), 2667, 260-263.

[3] Walsh, P. (2019). Support Vector Machine Learning for ECG Classification. In CERC (pp. 195-204).

