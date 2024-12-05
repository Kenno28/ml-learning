# My First Machine Learning Model

This project marks my first interaction with Machine Learning models. Using the Diabetes dataset, I implemented both a **Linear Regression** and a **Logistic Regression** model to explore supervised learning techniques.

## Project Description

In this project, I performed the following tasks:

- **Dataset Exploration**: Loaded and explored the Diabetes dataset.
- **Data Preprocessing**: 
  - Prepared the data by splitting it into training and test sets.
  - Transformed the target variable for Logistic Regression (binarization based on median).
- **Model Implementation**:
  - Built a **Linear Regression** model to predict disease progression as a continuous variable.
  - Developed a **Logistic Regression** model to classify patients based on disease progression (above or below the median).
- **Model Evaluation**:
  - For **Linear Regression**, evaluated performance using:
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
  - For **Logistic Regression**, assessed performance using:
    - Accuracy
    - Precision, Recall, and F1-score
    - Confusion Matrix
- **Visualization**:
  - Visualized prediction errors for Linear Regression using Seaborn and Matplotlib.
  - Generated a confusion matrix heatmap for Logistic Regression.

---
## Dataset

The dataset used in this project is the **Diabetes dataset** from the scikit-learn library. It contains data about patients and their features, with the target variable being the progression of diabetes after one year.

### Features
- **Age**
- **Sex**
- **BMI**
- **Average blood pressure**
- **Six blood serum measurements** (related to diabetes)

### Targets
1. **Disease progression** (Continuous variable for Linear Regression)
2. **Disease progression (Binarized)** (Binary classification for Logistic Regression; 1 for above median, 0 for below)

---

## Requirements

To run this project, you will need the following Python packages:

- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `scikit-learn`

Install them using pip:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn