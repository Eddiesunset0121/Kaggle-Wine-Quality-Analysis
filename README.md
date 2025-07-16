# Project 4 : Wine Quality Prediction: Multiple Classifiers Analysis

![wine_quality_2](https://github.com/user-attachments/assets/eeb2feef-f00c-46e1-91b1-d149bd5185a2)

🔷 Project Objective:

This project analyzes the public Red Wine Quality dataset from Kaggle to determine the key chemical properties that influence a wine's perceived quality. The primary goal is to build an accurate classification model that can predict whether a wine is of 'low', 'medium', or 'high' quality based on its physicochemical attributes.

🌟 Key Skills & Tools

- Data Cleaning & Manipulation: Pandas, NumPy

- Data Visualization: Matplotlib, Seaborn

- Machine Learning: Scikit-learn (Train-Test Split, Pipeline, StandardScaler, multiple classifiers including RandomForestClassifier, GridSearchCV)

- Core Competencies: Exploratory Data Analysis (EDA), Feature Engineering (Quantile Binning), Predictive Modeling, Hyperparameter Tuning, Model Evaluation.

🌿 Analysis & Key Findings

- The analysis involved a comprehensive EDA, a baseline comparison of seven different classification models, and hyperparameter tuning to develop a robust and accurate final model.

🌿 Foundational Analysis & Feature Engineering:

- A significant challenge was the class imbalance in the original 'quality' score. This was addressed by categorizing the target variable into three more balanced classes ('low', 'medium', 'high') using quantile binning.

- EDA revealed that the most promising predictors for wine quality are alcohol, volatile acidity, sulphates, and citric acid, all of which showed clear trends when plotted against the quality score. Higher quality wines are strongly associated with higher alcohol content and lower volatile acidity.

🌿 Model Development & Selection:

- A reusable pipeline was created to train and evaluate seven standard classification models on their baseline performance.

- The Random Forest classifier was the clear winner, achieving a baseline F1-score of 0.75, significantly outperforming the next-best model. This model was selected for further optimization.

🌿 Final Model Performance:

- Hyperparameter tuning was performed on the Random Forest model using GridSearchCV.

- The final, tuned model demonstrates strong predictive power, achieving an overall accuracy of 75% and a weighted average F1-score of 0.75 on unseen test data.

- The confusion matrix shows the model is particularly effective at identifying 'low' quality wines, with a high precision of 0.80 and recall of 0.83 for this class.

💡 Business Application

- Enhancing Quality Control: Allowing for the rapid assessment of a wine batch's quality based on lab measurements, helping to flag batches that do not meet desired standards.

- Optimizing Production: Providing winemakers with data-driven insights on which chemical properties (like alcohol and acidity levels) to target for consistent, high-quality wine production.

- Informing Blending Decisions: Assisting in the strategic blending of different wine batches to achieve a final product with a specific quality profile and chemical balance.



## Project Workflow

This project follows a standard data science methodology to identify key chemical predictors of wine quality and to build a predictive model.

1. Exploratory Data Analysis (EDA)

Loaded the dataset and performed an initial inspection of its structure, data types, and summary statistics.

Conducted a correlation analysis to understand the relationships between different chemical properties.

Visualized the distributions of each feature against wine quality using Kernel Density Estimate (KDE) plots to identify the most promising predictors. Key predictors identified were alcohol, volatile acidity, sulphates, and citric acid.

2. Feature Engineering & Data Preparation

Identified a significant class imbalance in the quality target variable.

Transformed the quality score into three balanced categories ('low', 'medium', 'high') using quantile-based binning to create a more suitable target for classification.

Split the data into an 80% training set and a 20% test set to ensure a final, unbiased evaluation of the model.

3. Model Development & Hyperparameter Tuning

Established a performance baseline by training and evaluating seven different classification models (e.g., Logistic Regression, Random Forest, SVC).

Random Forest was selected as the champion model based on its superior F1-score (0.75) in the baseline tests.

Optimized the Random Forest model by performing an exhaustive hyperparameter search using GridSearchCV with 5-fold cross-validation.

4. Final Model Evaluation

Evaluated the final, tuned model on the held-out test set, achieving a weighted average F1-score of 0.76.

Generated a confusion matrix and a multi-class ROC curve to provide a detailed visualization of the model's predictive performance across the different quality categories.

5. Interactive Prediction

Developed a simple, interactive function that allows a user to input wine characteristics and receive a quality prediction from the trained model.
