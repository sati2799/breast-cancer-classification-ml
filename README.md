# Breast Cancer Classification: PCA and Explainable Machine Learning

## Project Overview

This project explores how machine learning can classify breast tumors as malignant or benign, and whether dimensionality reduction can simplify the models without substantially reducing predictive performance.

Using the Breast Cancer Wisconsin Diagnostic dataset, I compared four classification algorithms, evaluated them with stratified cross-validation, and used Principal Component Analysis (PCA) to reduce the feature space. SHAP was then used to examine how the Logistic Regression model makes predictions.

The project focuses on model performance, dimensionality reduction, and interpretability rather than prediction accuracy alone.

## Research Questions

- How well do Logistic Regression, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Random Forest classify malignant and benign tumors?
- How many principal components are needed to retain at least 95% of the variance in the training data?
- Can PCA reduce the number of dimensions while maintaining similar classification performance?
- Which original features contribute most to Logistic Regression predictions, and how can SHAP help explain those predictions?

## Dataset

The project uses the Breast Cancer Wisconsin Diagnostic dataset available through `scikit-learn`.

- **Observations:** 569
- **Predictive features:** 30 numerical measurements of cell nuclei
- **Benign cases:** 357
- **Malignant cases:** 212
- **Missing values:** None found
- **Duplicate feature rows:** None found

The target variable was recoded for this analysis:

- `0` = Benign
- `1` = Malignant

Malignant tumors are treated as the positive class when calculating precision, recall, F1-score, and ROC-AUC.

**Dataset sources:** [Scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

## Methods and Workflow

1. **Exploratory Data Analysis:** Examined class distribution, feature distributions, correlations, and differences between benign and malignant cases.
2. **Data Preparation:** Checked missing values and duplicates, used a stratified 80/20 train-test split, and applied feature scaling within model pipelines.
3. **Model Comparison:** Evaluated Logistic Regression, SVM, KNN, and Random Forest using stratified five-fold cross-validation.
4. **Dimensionality Reduction:** Applied PCA to retain at least 95% of the training-data variance and compared models using original and reduced feature representations.
5. **Final Evaluation:** Compared Logistic Regression with the original features and with PCA on the held-out test set using accuracy, precision, recall, F1-score, confusion matrices, and ROC-AUC.
6. **Explainable AI:** Used SHAP global importance, beeswarm, and waterfall plots to examine feature contributions to Logistic Regression predictions.

Scaling and PCA were included inside the cross-validation pipelines to reduce the risk of data leakage between folds.

## Key Results

### Cross-Validation

The following results are mean scores across five stratified folds on the training set.

| Model               | Features | Recall | ROC-AUC |
| ------------------- | -------- | -----: | ------: |
| Logistic Regression | Original |  0.953 |   0.996 |
| Logistic Regression | PCA      |  0.953 |   0.994 |
| SVM                 | Original |  0.947 |   0.995 |
| SVM                 | PCA      |  0.947 |   0.995 |
| KNN                 | Original |  0.918 |   0.987 |
| KNN                 | PCA      |  0.929 |   0.987 |
| Random Forest       | Original |  0.935 |   0.989 |
| Random Forest       | PCA      |  0.929 |   0.988 |

Logistic Regression and SVM achieved strong results across the main evaluation metrics. PCA produced similar results overall, although its effect varied by model and metric.

### Final Test Set Evaluation

Logistic Regression was selected for the final comparison based on the cross-validation results.

| Metric    | Original Features |   PCA |
| --------- | ----------------: | ----: |
| Accuracy  |             0.965 | 0.974 |
| Precision |             0.975 | 1.000 |
| Recall    |             0.929 | 0.929 |
| F1-score  |             0.951 | 0.963 |
| ROC-AUC   |             0.996 | 0.997 |

On the full training split, PCA reduced the feature representation from 30 dimensions to 10 components while retaining at least 95% of the variance.

Both Logistic Regression versions correctly identified 39 of the 42 malignant cases and missed 3. The original-feature model produced one false positive, while the PCA version produced none on this test set.

These results suggest that PCA can reduce dimensionality while maintaining similar classification performance on this dataset. The differences are small, however, and do not establish that PCA will consistently improve results on new data.

### Model Interpretability

SHAP was used to examine global feature importance and individual predictions from Logistic Regression trained on the original features.

Features such as `worst texture`, `worst symmetry`, and `radius error` ranked among the influential features in the model's predictions. The beeswarm plot showed variation in feature contributions across test observations, while the waterfall plot illustrated the prediction for one selected case.

SHAP explains how a model uses its inputs; it does not establish causal relationships.

## Tools and Technologies

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- SHAP
- Jupyter Notebook
- Git and GitHub

## Repository Structure

```text
breast-cancer-classification-ml/
├── notebooks/
│   └── breast_cancer_classification.ipynb
├── README.md
└── requirements.txt
```

## Installation and Usage

Clone the repository:

```bash
git clone https://github.com/sati2799/breast-cancer-classification-ml.git
cd breast-cancer-classification-ml
```

Create and activate a virtual environment.

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies and Jupyter Notebook:

```bash
python -m pip install -r requirements.txt
python -m pip install notebook
```

Launch the notebook:

```bash
jupyter notebook notebooks/breast_cancer_classification.ipynb
```

The dataset is loaded directly through scikit-learn, so no separate dataset download is required.

## Limitations and Future Work

- The analysis uses a single dataset with 569 observations. Performance may differ on data collected from other populations or clinical settings.
- The test-set results come from one held-out split and require independent external validation before broader conclusions can be drawn.
- Correlated features can affect how individual contributions are distributed in SHAP explanations.
- PCA reduces the dimensionality of the model inputs but does not necessarily reduce the number of original measurements required to calculate those components.

Future work could include external validation, more systematic hyperparameter tuning, and evaluation on additional independent datasets.

**Disclaimer:** This is an educational machine learning project. The models are not intended for clinical diagnosis or medical decision-making.
