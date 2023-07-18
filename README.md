# AI-Based-Medical-Diagnosis-System
An AI-based application that offers medical diagnosis services such as Covid-19 detection from chest X-ray images, Brain tumor detection and segmentation from MRI images, Heartbeat abnormalities detection from ECG, Chest abnormalities detection and localization from chest X-ray images, and Skin cancer detection for skin images

Heartbeat abnormalities detection from ECG signal
--

[The MIT-BIH Arrhythmia Database](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) 
|Architecture|	Temporal| dependency|	Handling data imbalance|	Remarks|	Accuracy|	MacroF1|
|------------|----------------------|------------------------|---------|----------|--------|
Resampling the test data to 800 
[15]	Residual
CNN	-	Oversampling	Low recall on S,F classes	93.4%	
Ours	CNN+AFR	Bi-LSTM	Weighted loss	Good recall 	95.10%	
Test data as it is 
[71]	CNN	-	Oversampling	Low recall on S,F classes	93.47%	
[72]	SVM	-	-	Low recall		82%
Ours	CNN+AFR	Bi-LSTM	Weighted loss	Good recall 	97.11%	85.48%


![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/11daea27-a030-48ab-af3f-6a1a7447b6d8)

Proposed model 

![model2](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/4d01f247-5cd4-4eeb-bf1f-47c542626840)

Results comparison with the state of the art

![image](https://github.com/mohdakrory/AI-Based-Medical-Diagnosis-System/assets/67663339/fa7d3f29-12e8-4fe3-80f6-c8b16f0f46b2)

[1] rgrtgrtg
