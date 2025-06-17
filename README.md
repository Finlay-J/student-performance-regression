# Student Performance Regression

This project implements a simple linear regression model from scratch using **gradient descent** to predict students' math scores based on their reading scores.


---

## 📊 Dataset

**Source**: [Students Performance in Exams – Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

The dataset includes scores for math, reading, and writing exams, along with demographic data. This project specifically uses:

- **Reading Score** (as input/feature)
- **Math Score** (as target/label)

---

## 🔧 Technologies Used

- Python 3.x
- Pandas
- Matplotlib
- Scikit-learn (for evaluation metrics)

---

## 🧠 Project Overview

### Goal:
Use a single-variable linear regression model to learn the relationship between reading and math scores.

### Steps:
1. Load and explore the dataset
2. Implement linear regression using gradient descent
3. Update model weights (`m` and `b`) over multiple epochs
4. Evaluate model performance using:
   - **Mean Squared Error (MSE)**
   - **R² Score**
5. Visualize the regression line vs. actual data

---

## 📈 Results

- Final regression equation:  
  **`math_score = m * reading_score + b`**

- Model performance (example):
  - MSE: *e.g., 150.25*
  - R² Score: *e.g., 0.65*
