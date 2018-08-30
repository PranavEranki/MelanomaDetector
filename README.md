# Melanoma Detector
#### *by Pranav Eranki*  - pranav.eranki@gmail.com

### 1. Project Summary
The purpose of this project is to create a tool that considering the image of a
mole, can calculate the probability that the mole indicates the presence of melanoma within that user.

Skin cancer is a common disease that affect a big amount of
peoples. Some facts about skin cancer:

- Every year there are more new cases of skin cancer than the
combined incidence of cancers of the breast, prostate, lung and colon.
- An estimated 87,110 new cases of invasive melanoma will be diagnosed in the U.S.
in 2017.
- The estimated 5-year survival rate for patients whose melanoma is detected
early is about 98 percent in the U.S. The survival rate falls to 62 percent when
the disease reaches the lymph nodes, and 18 percent when the disease metastasizes
to distant organs.
- Early detection is critical!


### 2. Development process and Data
The idea of this project is to construct a CNN model that can predict the probability
that a specific mole can be malign.


#### 2.1 Data:
To train this model the data to use is a set of images from the International
Skin Imaging Collaboration: Mellanoma Project ISIC https://isic-archive.com.

The benign data was downloaded as a zip file of all the benign cases.

Likewise was done for the malign data.

#### 2.2 Preprocessing:
The following preprocessing tasks are developed for each image:
2. Image resizing: Transform images to 128x128x3
3. Crop images: Automatic or manual Crop
4. Other to define later in order to improve model quality

#### 2.3 CNN Model:
The idea is to develop a CNN model from scratch, and evaluate the performance to set a baseline. 
The following steps to improve the model are:

1. Data augmentation: Rotations, noising, scaling to avoid overfitting
2. Transferred Learning: Using a pre-trained network construct some additional
layer at the end to fine tuning our model. (VGG-16, or other)
3. Full training of VGG-16 + additional layer.


#### 2.4 Model Evaluation:
To evaluate the different models we will use ROC Curves and AUC score. To choose
the correct model we will evaluate the precision and accuracy to set the threshold
level that represent a good tradeoff between TPR and FPR.

### 3. Results presentation
As mentioned before, the idea is to generate a tool to predict the probability of a
malign mole. To do it, I'm planning to provide the following resources:

  **1. Web App:** The web app will have the functionality that if the user uploaded a high
quality image of an specific mole, the result will be a prediction about the
probability that the given mole is malign in terms of percentage.

  **2. Iphone App:** If time allows, and if my friends are willing.
  
  **3. Android App:** Also if time allows.
  
 ### 3. Tools to Use
 - Tensorflow (GPU High performance computing - NVIDIA)
 - keras
 - Python
 - matplotlib
 - scikit-learn
 - AWS (EC2 - S3)
 - IoS swift + core ML (for possible iOS app)
 - Flask or Django for web
 - Android Studio (for possible android model)
 
### 4. Note about documentation
The documentation for this program is fairly self explanatory.

A lot of the code which is written is fairly documented with comments.
 
Please keep in mind that if you wish to understand a lot of these topics through the use of the comments, you would need some ML experience.
 
 ### 5. Next Steps

- Create Ios App
- Create Android App
- Improve model with additional data

### 6. Disclaimer

This tool has been designed only for educational purposes to demonstrate the use of Machine Learning tools in the medical field. 
This tool does not replace advice or evaluation by a medical professional. Nothing in this project should be construed as an attempt to 
offer a medical opinion or practice medicine.

