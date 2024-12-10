Stroke Prediction Using Machine Learning
Overview
This project aims to predict the likelihood of a person experiencing a stroke using machine learning techniques. A neural network classifier is implemented for accurate predictions based on medical and lifestyle data.

Table of Contents
Dataset
Technologies Used
Steps Involved
1. Data Collection
2. Data Preprocessing
3. Data Visualization and EDA
4. Classification
Results
How to Run the Project
Conclusion
Dataset
The dataset contains patient information, including:

Gender
Age
Hypertension
Heart Disease
Marital Status
Work Type
Residence Type
Average Glucose Level
BMI
Smoking Status
Stroke History (Target)
Technologies Used
Programming Language: Python
Data Manipulation: Pandas, NumPy
Data Visualization: Seaborn, Matplotlib
Machine Learning: Scikit-learn, TensorFlow/Keras
Steps Involved
1. Data Collection
The dataset was sourced from Kaggle, containing anonymized patient health records. It includes relevant features that influence stroke prediction.

2. Data Preprocessing
Preprocessing steps include:

Handling missing values (e.g., imputation for BMI).
Encoding categorical variables like gender and work type.
Scaling numerical features such as age and glucose level.
3. Data Visualization and EDA
Exploratory Data Analysis (EDA) was performed to uncover patterns and relationships:

Distribution of Age Groups: Understanding age as a risk factor.
Correlation Heatmap: Highlighting relationships between features and stroke occurrences.
Glucose and BMI Analysis: Analyzing the effects of average glucose levels and BMI.
Sample Visualizations:

Replace path/to/image with the actual path if adding an image.

4. Classification
A neural network classifier using Keras was implemented:

Input Layer: Accepts all features.
Hidden Layers: Fully connected layers with ReLU activation.
Output Layer: Single neuron with sigmoid activation for binary classification.
Loss Function: Binary Crossentropy.
Optimizer: Adam optimizer.
Results
Model Accuracy: 92.47%
Evaluation Metrics:
Confusion Matrix
Precision, Recall, and F1 Score
ROC AUC Score: The model achieves a reliable level of discrimination.
How to Run the Project
Prerequisites
Install Python 3.7 or higher.
Install the required libraries:

Conclusion
This project demonstrates the use of a neural network for stroke prediction. By utilizing medical data and proper preprocessing, the model achieves significant accuracy, potentially aiding in early diagnosis and preventive measures.

Author
Akash