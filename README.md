# AI-Based-Medical-Diagnosis-System
An AI-based application that offers medical diagnosis services such as Covid-19 detection from chest X-ray images, Brain tumor detection and segmentation from MRI images, Heartbeat abnormalities detection from ECG, Chest abnormalities detection and localization from chest X-ray images, and Skin cancer detection for skin images

Heartbeat abnormalities detection from ECG signal
--

[The MIT-BIH Arrhythmia Database](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) 

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/ad3b1e4b-dfcd-40a7-a9f2-eb85bf3178a3)

N: Normal beat
S: Supraventricular premature beat
V: Premature ventricular contraction
F: Fusion of ventricular and normal beat
Q: Unclassifiable beat

This Kaggle dataset was preprocessed and published by [1] 

Proposed model 

![model2](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/4d01f247-5cd4-4eeb-bf1f-47c542626840)

Model performance 

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

<table>
  <tr>
    <th>Index</th>
    <th>Class name</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-score</th>
    <th>Support</th>
  </tr>
  <tr>
    <td>0</td>
    <td>N</td>
    <td>0.9946</td>
    <td>0.9738</td>
    <td>0.9841</td>
    <td>18118</td>
  </tr>
  <tr>
    <td>1</td>
    <td>S</td>
    <td>0.6993</td>
    <td>0.8615</td>
    <td>0.772</td>
    <td>556</td>
  </tr>
  <tr>
    <td>2</td>
    <td>V</td>
    <td>0.9347</td>
    <td>0.9593</td>
    <td>0.9468</td>
    <td>1448</td>
  </tr>
  <tr>
    <td>3</td>
    <td>F</td>
    <td>0.4185</td>
    <td>0.9506</td>
    <td>0.5811</td>
    <td>162</td>
  </tr>
  <tr>
    <td>4</td>
    <td>Q</td>
    <td>0.9882</td>
    <td>0.9913</td>
    <td>0.9898</td>
    <td>1608</td>
  </tr>
</table>
<br>
<table>
  <tr>
    <td>Accuracy</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td>0.9711</td>
    <td>21892</td>
  </tr>
  <tr>
    <td>Macro avg</td>
    <td></td>
    <td>0.8071</td>
    <td>0.9473</td>
    <td>0.8548</td>
    <td>21892</td>
  </tr>
  <tr>
    <td>Weighted avg</td>
    <td></td>
    <td>0.9784</td>
    <td>0.9711</td>
    <td>0.9737</td>
    <td>21892</td>
  </tr>
</table>

Results comparison with the state of the art

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/cb0dc9d5-3965-4ed6-a825-b60d5dbafd07)

Hardware system for real-time analysis of ECG signal 

![diagram](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/59af1a62-55cd-421a-bc3d-8f7ed82d4705)



References
--

[1]  M. Kachuee, S. Fazeli, and M. Sarrafzadeh, “Ecg heartbeat classification: A deep transferable representation,” in 2018 IEEE International Conference on               Healthcare Informatics (ICHI), 2018, pp. 443–444.

[2]   Pirova, D., Zaberzhinsky, B., & Mashkov, A. (2020). Detecting heart 2 disease symptoms using machine learning methods. Proceedings of the Information                 Technology and Nanotechnology (ITNT-2020), 2667, 260-263.

[3] Walsh, P. (2019). Support Vector Machine Learning for ECG Classification. In CERC (pp. 195-204).

