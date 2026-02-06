# House Price Prediction using Machine Learning

## 📌 Project Overview
This project aims to predict house prices using regression-based machine learning techniques. The project focuses on understanding the regression workflow, feature scaling, model evaluation, and result interpretation using a real-world housing dataset.

---

## 📊 Dataset
- **Dataset Name:** California Housing Dataset
- **Source:** scikit-learn
- **Number of Records:** 20,640
- **Number of Features:** 8
- **Target Variable:** Median house price

The dataset is publicly available, well-structured, and does not contain missing values.

---

## ⚙️ Model Used
- Linear Regression

Linear Regression was chosen as a baseline regression model due to its simplicity and interpretability.

---

## 🔍 Methodology
1. Load and explore the dataset
2. Analyze data distribution
3. Check for missing values
4. Split data into training and testing sets
5. Apply feature scaling using StandardScaler
6. Train a Linear Regression model
7. Evaluate performance using regression metrics
8. Analyze feature coefficients
9. Visualize residual errors

---

## 📈 Evaluation Metrics
- Mean Squared Error (MSE)
- R² Score

---

## 📊 Model Performance
- **Mean Squared Error:** 0.556
- **R² Score:** 0.576

The results indicate moderate predictive performance, which is expected for a linear regression model on this dataset.

---

## 📸 Output Visualizations

### House Price Distribution
![Price Distribution](images/price_distribution.png)

### Residual Plot
![Residual Plot](images/residual_plot.png)

---

## 🧠 Conclusion
The Linear Regression model was able to capture the relationship between housing features and prices reasonably well. Feature coefficients and residual analysis helped interpret model behavior. More advanced models may further improve prediction accuracy.

---

## 🛠️ Tools & Technologies
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

---

## ▶️ How to Run the Project
```bash
python house_price_prediction.py
