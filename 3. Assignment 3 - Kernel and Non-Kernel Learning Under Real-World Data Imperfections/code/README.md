# Homework 3: Kernel and Non-Kernel Learning Under Real-World Data Imperfections



---

## Overview

This project investigates how data characteristics, imperfections, and preprocessing decisions affect the performance of **Linear Models**, **Kernel Methods**, and **Distance-Based Methods** across four real-world datasets.

## Datasets

| # | Dataset | Source | Task(s) |
|---|---------|--------|---------|
| 1 | Airbnb Open Data | Inside Airbnb | Price (Regression), Superhost (Classification) |
| 2 | NYC 311 Service Requests | NYC Open Data | Complaint Category (Classification), Resolution Time (Regression) |
| 3 | IBM HR Employee Attrition | UCI/Kaggle | Employee Attrition (Classification) |
| 4 | Online Retail Transactions | UCI ML Repository | Customer Segment (Classification), Future Spending (Regression) |

## Algorithms Implemented (From Scratch)

**Non-Kernel Methods** (NumPy/SciPy only):
- Linear Regression (Ridge, closed-form)
- Logistic Regression (gradient descent, OVR for multi-class)
- K-Nearest Neighbors (classification & regression)
- Decision Tree (Gini impurity / MSE reduction)

**Kernel Methods** (NumPy/SciPy only):
- Kernel SVM (simplified SMO algorithm)
- Kernel Ridge Regression (dual form)
- Kernel KNN (kernel-induced distance)
- Kernel PCA + downstream Logistic Regression

**Kernels Supported**: Linear, Polynomial, RBF (Gaussian)

## Project Structure

```

├── data/                          # Raw datasets
│   ├── Inside Airbnb Dataset/
│   ├── NYC 311 Service Requests/
│   ├── IBM_HR_Analytics_Attrition_Dataset/
│   └── Online Retail.xlsx
├── output/                        # Generated outputs
│   ├── section1/                  # Data engineering plots
│   ├── section2/                  # Data quality investigation
│   ├── section3/                  # Model results & comparisons
│   ├── section4/                  # Kernel investigation
│   └── section5/                  # Discussion & failure analysis
├── src/
│   ├── section1/                  # Data loading & engineering
│   │   ├── airbnb.py
│   │   ├── nyc311.py
│   │   ├── ibm_hr.py
│   │   └── online_retail.py
│   ├── section2/                  # Data quality investigation
│   │   └── data_quality.py
│   ├── section3/                  # Algorithm implementations
│   │   ├── linear_regression.py
│   │   ├── logistic_regression.py
│   │   ├── knn.py
│   │   ├── decision_tree.py
│   │   ├── kernel_svm.py
│   │   ├── kernel_ridge.py
│   │   ├── kernel_knn.py
│   │   └── kpca.py
│   ├── utils/
│   │   ├── kernels.py             # Kernel functions
│   │   ├── metrics.py             # Evaluation metrics
│   │   └── preprocessing.py       # Scaling, splitting, SMOTE
│   └── visualize.py               # ALL visualization code
├── main.py                        # Main orchestrator
├── requirements.txt
└── README.md
```

## Setup & Execution

### Prerequisites
- Python 3.8+

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run
```bash
python main.py
```



### Outputs
All generated plots and results are saved in the `output/` directory, organized by section.



## Assumptions

1. NYC 311 dataset is subsampled to 80K rows for tractability, preserving distribution integrity.
2. Kernel methods use smaller samples (1500) due to O(n²) kernel matrix computation.
3. IBM HR dataset uses SMOTE oversampling to address class imbalance (16% attrition rate).
4. Online Retail data is aggregated from transaction-level to customer-level features (RFM analysis).
5. For multi-class classification, one-vs-rest strategy is used.
6. Isolation Forest outlier detection is approximated via percentile-based method (full implementation would require scikit-learn).
