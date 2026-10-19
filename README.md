# Kaggle-ML-Projects

Kaggle datasets par bane mere **AI/ML learning projects** ka collection. Har project me poora machine learning workflow follow kiya gaya hai: data samajhna, preprocessing, pipeline banana, model train karna, evaluate karna aur predictions nikalna.

## Projects

| # | Project | Problem type | Model | Folder |
|---|---|---|---|---|
| 1 | Titanic Survival Prediction | Binary classification (survived / not survived) | Random Forest + GridSearchCV | [`Titanic_survival_prediction`](./Titanic_survival_prediction) |
| 2 | Predictive Maintenance | Binary classification (machine failure / no failure) | Random Forest + GridSearchCV | [`Predictive_Maintenance_prediction`](./Predictive_Maintenance_prediction) |

### 1. Titanic Survival Prediction
Passengers ki details (class, sex, age, fare etc.) se predict karna ki wo survive karega ya nahi. Isme missing age ko impute karna, `Sex` ko one-hot encode karna aur unnecessary columns drop karna jaise steps custom transformers me kiye gaye hain.

### 2. Predictive Maintenance
Machine ke sensor data (air/process temperature, rotational speed, torque, tool wear, machine type) se predict karna ki machine fail hogi ya nahi. Data heavily imbalanced hai (sirf ~3.4% failures), isliye `class_weight="balanced"` aur `f1` scoring use hui.

**Test set result:** accuracy 97.4%, ROC-AUC 0.96, failure class ka recall 0.63. Poori detail project ke apne README me hai.

## Common workflow (dono projects me)

1. Data load karna aur `describe()` / correlation heatmap se samajhna
2. `StratifiedShuffleSplit` se train/test split
3. Custom `BaseEstimator` + `TransformerMixin` transformers
4. `sklearn.pipeline.Pipeline`
5. `StandardScaler`
6. `RandomForestClassifier` + `GridSearchCV`
7. Test set par evaluation
8. Poore data par final model train karke `predictions.csv` banana

## Repository structure

```
Kaggle-ML-Projects/
├── Titanic_survival_prediction/
├── Predictive_Maintenance_prediction/
│   ├── Predictive_Maintenance_prediction.ipynb
│   ├── predictive_maintenance.csv
│   ├── predictions.csv
│   └── README.md
└── README.md
```

## Tech stack

- Python 3
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn
- Jupyter Notebook / Google Colab

## Run kaise karein

```bash
git clone https://github.com/PanwarParth1/Kaggle-ML-Projects.git
cd Kaggle-ML-Projects
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook
```

Phir jis project ka notebook chalana ho uska folder kholein, dataset (`.csv`) notebook ke same folder me rakhein aur **Run all** karein. Colab me chalane par pehla cell file upload maangega.

## Learning focus

Is repo ka maqsad sirf accuracy badhana nahi, balki ML ke important concepts practice karna hai:

- Data leakage se bachna (test set par sirf `transform`, `fit` nahi)
- Imbalanced data ko handle karna
- Accuracy ke saath precision, recall, F1 aur ROC-AUC dekhna
- Reusable preprocessing pipelines banana

## Aage ke plans

- Aur Kaggle datasets par projects add karna
- Alag models (Logistic Regression, XGBoost etc.) se comparison

## Author

**Parth Panwar** ([@PanwarParth1](https://github.com/PanwarParth1))

Suggestions ya improvements ke liye issue ya pull request khol sakte hain.
