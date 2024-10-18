# Project I: Diabetes Risk Prediction and Biomarker Clustering Analysis Report
**Author: Dong Liang**

**Highlights:**

The analysis notebooks were prepared to predict metabolite biomarkers for incidental diabetes using various machine/deep learning models, including logistic regression, transformer-based architectures, and deep neural networks. These notebooks cover the entire process from data preprocessing to model training and evaluation, showcasing my expertise in handling complex data science tasks and leveraging state-of-the-art machine/deep learning technologies to generate actionable insights.

---

[Analysis Report(html)](https://github.com/DongL/Data-Visualization/blob/master/Diabetes_risk_prediction.html)

[Transformer](https://github.com/DongL/Data-Visualization/blob/master/transformer/)

The Analysis Report (html file) can be previewed using [Github Html Preview](https://htmlpreview.github.io).
---

**1. Introduction**

Diabetes is a chronic metabolic disorder characterized by elevated blood sugar levels, which can lead to severe health complications if not managed effectively. Identifying the risk factors associated with the development of diabetes and predicting its onset are crucial for early intervention and prevention strategies. This report presents an analysis based on a comprehensive dataset derived from a long-term public health study, aiming to achieve the following objectives:

- Identify clinical features and blood biomarkers associated with incident diabetes.
- Develop machine/deep learning predictive models using blood biomarkers to assess the risk of developing diabetes.
- Identify subsets of individuals who developed incident diabetes based on blood biomarkers.

**2. Purpose**

The purpose of these analysis notebooks is to provide a detailed exploration of metabolite biomarkers and their potential role in predicting incidental diabetes. This work not only aims to advance understanding in the field of metabolomics but also to highlight my expertise in data analysis and machine learning to potential employers and collaborators. By showcasing my skills in data science, I hope to open up opportunities for future projects and collaborations.

**3. Goals of the Project**

The goals of this project are threefold:

1. To identify which clinical features and which blood biomarkers are associated with incident diabetes.
2. To build a machine learning predictive model based on blood biomarkers for assessing incident diabetes risk.
3. To identify subsets of subjects who developed incident diabetes based on blood biomarkers.

**4. Dataset Description**

The analysis utilizes the `diabetes_project_dataset.csv`, sourced from a longitudinal public health study. The dataset comprises information from physical examinations and blood tests of participants, tracking their health outcomes over several years. Key attributes in the dataset include:

- **Demographics**: Age, gender.
- **Clinical Features**: BMI, blood pressure (SBP, DBP), hypertension status, fasting glucose and insulin levels, HbA1c, smoking status, exercise habits, dietary habits.
- **Blood Biomarkers**: Numerous biomarkers prefixed with mtb\_ (e.g., mtb\_2129028). The dataset contains 8,291 records with 60,207 columns, indicating high dimensionality that necessitates meticulous data processing and feature selection.

**5. Data Processing and Exploratory Data Analysis (EDA)**

5.1. **Data Loading and Initial Inspection**

The dataset was efficiently loaded in chunks to manage memory usage. Initial statistical summaries revealed the distribution of various features, highlighting mean values, standard deviations, and the presence of outliers in several blood biomarker measurements.

5.2. **Handling Missing Data**

A significant proportion (30.1%) of the columns had less than 20% missing data. Given that missing values in blood biomarkers often result from concentrations below detection limits, missing values were imputed with zeros during preprocessing. For skewed data with high absolute skewness (>1), log transformation was applied, and missing values were further imputed with small constants (e.g., 1e-10).

5.3. **Distribution Analysis**

- **Blood Biomarkers**: Histograms of randomly selected mtb\_ biomarkers indicated non-normal distributions with right skewness and outliers, justifying log transformations for normalization.
- **Clinical Data**: Distribution analyses for clinical features revealed patterns that informed subsequent modeling decisions.

5.4. **Multicollinearity Assessment**

A correlation matrix for a subset of mtb\_ biomarkers using Spearman correlation revealed high correlations among certain features, suggesting redundancy. This finding underscored the necessity for feature selection to mitigate multicollinearity in predictive models.

5.5. **Participant Demographics**

- **Age Distribution**: Participants' ages ranged from 24 to 74 years, with a mean age of approximately 48 years.
- **Exercise Habits**: Exercise levels varied, categorized into minimal movement, 3-4 hours per week, and intense exercise.
- **Incident Diabetes Distribution**: The target variable (incident\_diabetes) was zero-inflated, leading to the choice of ROC AUC as the primary evaluation metric for models.


**6. Tools and Libraries Used**

The analysis utilizes a variety of Python libraries for data processing and machine learning, including:

- `numpy` and `pandas` for data manipulation
- `scikit-learn` for preprocessing, model training, and evaluation
- `PyTorch` for building transformer-based deep neural networks
- `matplotlib` and `seaborn` for data visualization
- `statsmodels` for statistical analysis
- Additional tools like `UMAP` for clustering analysis

**7. Analysis**

7.1. **Identification of Associated Clinical Features and Blood Biomarkers**

7.1.1. **Clinical Features**

Univariate logistic regression analyses identified several clinical features significantly associated with incident diabetes. The top risk factors included:

- BMI (Body Mass Index)
- Fasting Glucose Levels
- Hypertension
- HbA1c Levels
- Fasting Insulin Levels

Conversely, features such as HDL cholesterol and exercise habits showed protective associations against diabetes development.

7.1.2. **Blood Biomarkers**

Logistic regression models were fitted for each blood biomarker. The analysis revealed that certain biomarkers (mtb\_744362\_log, mtb\_744357\_log, mtb\_767346\_log, etc.) had strong positive associations with incident diabetes, indicating their potential role as risk factors. The significant p-values and positive coefficients suggest these biomarkers increase the odds of developing diabetes.

7.2. **Predictive Modeling Using Blood Biomarkers**

7.2.1. **Model Development**

The predictive modeling task involved building logistic regression classifiers using selected blood biomarkers. The process included:

- **Feature Selection**: Utilized SelectKBest based on ANOVA F-values and Random Forest-based feature selection to identify relevant biomarkers.
- **Data Preprocessing**: Implemented pipelines with standard scaling, polynomial feature engineering, and Principal Component Analysis (PCA) for dimensionality reduction.
- **Model Training**: Employed cross-validation and grid search techniques to optimize hyperparameters, including regularization strength (C) and penalty type (l1, l2).

7.2.2. **Evaluation Metrics**

The models were evaluated using a range of metrics:

- **ROC AUC**: Primary metric due to class imbalance.
- **Accuracy, F1 Score, Recall, Specificity, Precision, Negative Predictive Value**

7.2.3. **Model Performance**

The optimal logistic regression model (lr\_kb\_15000) achieved a Test AUC of 0.8433, indicating good discriminative ability. Other models with fewer biomarkers maintained high accuracy and specificity, though with varying AUC and precision scores. Models based on SelectKBest outperformed those using Random Forest-based feature selection in terms of AUC.

**Key Findings**:

- **Optimal Model Settings**:
  - Classifier C: 0.001
  - Penalty: L2 Regularization
  - Feature Selector k: 15,000
- **Performance Trade-offs**: While reducing the number of biomarkers slightly decreased AUC, it enhanced model interpretability and maintained acceptable performance across other metrics.
- **Model Bias**: High specificity and negative predictive value indicate the model is more adept at identifying non-diabetic individuals than diabetic ones, highlighting potential sensitivity issues.

7.3. **Clustering Analysis of Incident Diabetes Participants**

7.3.1. **Clustering Technique**

Using the top 20 most associated blood biomarkers, participants with incident diabetes were clustered using Uniform Manifold Approximation and Projection (UMAP) for dimensionality reduction, followed by K-Means clustering.

7.3.2. **Cluster Characteristics**

Four distinct clusters emerged, each with unique demographic and clinical profiles:

- **Category I**:

  - Age: Youngest (Mean: 51.36)
  - BMI: Lower
  - Hypertension: Lower
  - Triglycerides & LDL: Higher
  - Junk Food Consumption: Highest
  - Gender: Predominantly Male (68%)

- **Category II**:

  - BMI: Relatively low
  - HbA1c & Fasting Glucose: Lowest
  - Blood Pressure: Lowest
  - Triglycerides: Lowest
  - Junk Food Consumption: Moderate
  - Gender: Balanced (53% Female)

- **Category III**:

  - Age: Highest (Mean: 55.55)
  - HbA1c: Highest
  - Blood Biomarkers: Intermediate values
  - Gender: Predominantly Female (59%)

- **Category IV**:

  - Age: Higher (Mean: 54.19)
  - Diabetes Time: Shortest (7.56 years)
  - BMI: Highest
  - Fasting Glucose & Insulin: Highest
  - Blood Pressure & Triglycerides: Highest
  - HDL: Relatively lower
  - Gender: Predominantly Male (63%)

**Clustering Insights**:

- **Lifestyle Factors**: Differences in junk food consumption and exercise habits contribute to cluster differentiation.
- **Metabolic Indicators**: Variations in BMI, glucose levels, and lipid profiles highlight underlying metabolic disparities.

**8. Summary and Conclusion**

This comprehensive analysis successfully identified key clinical features and blood biomarkers associated with the development of incident diabetes. The predictive modeling efforts yielded a logistic regression model with high discriminative ability (Test AUC = 0.8433) using selected blood biomarkers, demonstrating potential for risk assessment in clinical settings.

**Key Achievements**:

- **Feature Identification**: Highlighted critical clinical and biomarker predictors of diabetes.
- **Predictive Modeling**: Developed and optimized a robust logistic regression model with significant predictive performance.
- **Clustering Analysis**: Unveiled distinct participant profiles based on biomarker data, offering insights into heterogeneous pathways of diabetes development.

**Limitations and Future Work**:

- **Model Sensitivity**: The current models exhibit lower sensitivity in detecting diabetic cases, suggesting the need for improved recall and precision.
- **Computational Constraints**: Limited exploration of hyperparameter spaces and model architectures due to resource restrictions.
- **Advanced Modeling**: Future studies could incorporate deep learning techniques, such as transformer-based neural networks, to enhance model performance and sensitivity.

**Conclusion**:

The project underscores the value of integrating clinical and biomarker data in understanding and predicting diabetes risk. The methodologies applied, from data preprocessing and EDA to sophisticated modeling and clustering techniques, provide a solid foundation for further research and application in public health strategies aimed at diabetes prevention and management.




---

# Project II: Visualization of the Trade History between China and U.S. States

[Report](https://github.com/DongL/Data-Visualization/blob/master/Dviz%20Project%20Report.pdf)

[PPT](https://github.com/DongL/Data-Visualization/blob/master/Dviz_Project_PPT.pdf)

[Jupyter Notebook](https://github.com/DongL/Data-Visualization/blob/master/Dviz_Project_notebook.ipynb)

This project aims to use various visualization techniques and machine learning methods to study the history of trade between China and U.S. states. To achieve this goal, we analyzed the data from different sources, performed manifold learning-based clustering analysis, and created the best visualizations to illustrate our results. We hope that these visualizations and insights gained in our analyses can help us understand the trend of the U.S.-China trade and provide us some clues on the possible impact of the trade war on individual U.S. states.


A manifold learning model based on Uniform Manifold Approximation and Projection (UMAP) was built using a dataset that included a number of export-related economic data (e.g. economic aggregates (GDP), exports to major trading partners and their fractions, exports to China and their fractions), as well as geographical location and political oriention of individual U.S. states. The optimal clustering was achieved with the UMAP hyperparameters of n_neighbor = 3 and min-dist = 0.001, which successfully yielded 4 meaningful structures of U.S. state clusters.

![classification](image/GeographicDistribution.png)

-	Category I: Washington, South Carolina, Oregon, Alabama, Kentucky, Puerto Rico, New Mexico, Alaska, Connecticut, Utah, New Hampshire
-	Category II: Texas, California, New York, Illinois, Georgia, Massachusetts, Pennsylvania, Tennessee, North Carolina, Minnesota, Florida, New Jersey, Virginia, Maryland, Colorado
-	Category III: Louisiana, Nevada, Mississippi, West Virginia, Idaho, Delaware, Arkansas, Maine, Vermont, Rhode Island, Montana, Wyoming, Hawaii, District of Columbia
-	Category IV: Ohio, Michigan, Indiana, Wisconsin, Arizona, Missouri, Kansas, Iowa, Nebraska, Oklahoma, South Dakota, North Dakota


