# Tabular 3-Class ML Pipeline

This project focuses on training and selecting effective feature preprocessing methods and prediction models for a large tabular dataset with three class labels (`Apple`, `Google`, `Meta`).

The goal is to:

- Clean and preprocess a real-world style dataset  
- Explore different feature transformations (outlier handling, scaling, PCA)  
- Compare several classifiers  
- Use the best model to generate predictions for an unlabeled evaluation set  

---

## Dataset & Task

Each row is one observation with:

- **Target label `y`** ∈ {`Apple`, `Google`, `Meta`}
- **Features** `x1`–`x13`, for example:

| y     | x1       | x2      | x3       | x4      | x5       | x6        | x7          | x8     | x9      | x10     | x11     | x12  | x13      |
|-------|----------|---------|----------|---------|----------|-----------|-------------|--------|---------|---------|---------|------|----------|
| Meta  | 200.4492 | -0.0219 | -96.8066 | -0.9706 | 230.1735 | -122.2511 | Churn       | 2.9139 | 2.57123 | 15.56023| -3.7395 | True | 953.8427 |
| Apple | 198.0707 | -0.2379 | -97.6809 | -0.9829 | 227.8439 | -117.8249 | Acquisition | 1.8819 | -0.7749 | 10.4415 | -2.3821 | True | 941.5128 |
| Meta  | 201.1316 | 0.8704  | -98.4326 | -0.9741 | 230.8579 | -122.7044 | Nominal     | 0.5232 | -0.9993 | 8.7650  | -3.1947 | True | 956.4418 |
| ...   | ...      | ...     | ...      | ...     | ...      | ...       | ...         | ...    | ...     | ...     | ...     | ...  | ...      |

Notable properties:

- `x7` is a **categorical feature** (`Churn`, `Acquisition`, `Nominal`, `Release`, …).
- `x1`–`x6`, `x8`–`x13` are numerical and contain outliers / noisy entries.
- A separate evaluation file contains the same features but **no labels**.

---

## Approach

High-level steps in `main.ipynb`:

1. **Load data**
   - Training data with labels
   - Evaluation data without labels

2. **Preprocessing**
   - Encode the categorical feature `x7` using `LabelEncoder`
   - Remove outliers based on quantiles
   - Check and remove highly correlated features
   - Standardize features with `StandardScaler`
   - Optionally apply **PCA** for dimensionality reduction

3. **Modeling**
   - Train and compare:
     - `RandomForestClassifier`
     - `CatBoostClassifier`
     - `XGBClassifier`
   - Use a train/validation split on the training data to estimate performance

4. **Model selection**
   - Select the model with the best validation performance / stability

5. **Inference**
   - Refit the chosen model on the full training data
   - Predict labels for the evaluation dataset
   - Save predictions to a `.txt` file with **one label per line**, in dataset order

---

## Results

On a held-out validation split of the training data, the best performance was, as expected, achieved by the **CatBoostClassifier**, with an accuracy of approximately **89%**.

The model comparison showed that CatBoost handled the mix of numerical and categorical features particularly well compared to RandomForest and XGBoost in this setting.

---

## Requirements

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```
