# Predictive-Modeling-in-Healthcare
Modeled patient care outcomes using Python and decision trees, improving early-risk detection and supporting treatment decisions.
# 🏥 Predictive Modeling in Healthcare Using Decision Trees

This project applies decision tree regression techniques to healthcare data in order to model and predict patient-related sales outcomes across customer segments. The objective was to assist healthcare decision-makers by forecasting spending behavior and enabling early interventions through data-driven insight.

---

## 🎯 Objective

To develop a predictive model using a decision tree regressor that estimates healthcare-related sales based on customer segment features. The ultimate goal is to optimize patient care strategies and business planning in segment-specific healthcare services.

---

## 🛠️ Technologies & Libraries

- Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
- Decision Tree Regressor
- Jupyter Notebook
- Visualizations with `matplotlib`
- Model rendering with `pydotplus` and `export_graphviz`

---

## 📁 Dataset Overview

- Source: Sales data categorized by customer segment
- Features:
  - Age
  - Region
  - Segment (`Corporate`, `Home Office`, `Consumer`)
  - Sales ($)

- Preprocessing:
  - One-hot encoding applied to `Segment`
  - Outliers removed using IQR filtering
  - Data split into training (80%) and testing (20%)

---

## 🔍 Exploratory Data Analysis (EDA)

- Visualized skewness in sales distribution
- Correlation between:
  - Quantity vs. Profit
  - Sales vs. Discount
- Weekly aggregated sales plots
- Exported statistical summaries and cleaned datasets to Excel

📂 Files:
- `EDA.py`: Descriptive statistics  
- `Correlation_analysis_*.py`: Pearson correlation analysis  
- `Skewness of Sales Data.py`: Sales distribution skewness  
- `Weekly Aggregated Sales.py`: Time-based sales trends  

---

## 🌲 Model: Decision Tree Regressor

📂 Code files:
- `Decision_Tree.py`: Complete model training and segment-based predictions  
- `tree_Structure.py`: Visual representation of decision tree using `pydotplus`  
- `visualization.py`: Segment-wise comparison of actual vs. predicted sales  

### 📈 Evaluation Metrics
- **MAE** (Mean Absolute Error): ~Low
- **MSE** (Mean Squared Error): ~Low
- **RMSE** (Root Mean Square Error): Stable
- **R² Score**: High (> 0.9 in most test runs)

---

## 📊 Visualization Highlights

- Predicted vs. Actual Sales by Segment
- Decision tree image generated from model (using `export_graphviz`)
- Skewness plots for model validation
- Weekly aggregated sales bar chart

📷 _Sample Visuals from `visualization.py`_  
(Upload your plot images here, or embed screenshots from Jupyter)

---

## ✅ Key Learnings

- Decision trees performed well with one-hot encoded segment data
- Removal of outliers significantly improved model accuracy
- Data-driven insights can help optimize marketing strategies in healthcare retail

---

## 📬 Author

**Sajith Pemarathna**  
📫 Email: sajiths.pemarathna@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/sajith-pemarathna)

---

## 🚀 Future Enhancements

- Test Random Forest & Gradient Boosting for improved accuracy
- Apply SHAP or LIME for explainable model insights
- Deploy the model as a REST API for real-time prediction
