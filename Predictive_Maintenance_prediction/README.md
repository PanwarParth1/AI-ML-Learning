# Predictive Maintenance - Machine Failure Prediction

Machine ke sensor data se ye predict karna ki machine **fail hogi ya nahi**.
Notebook ka structure `Titanic_survival_prediction.ipynb` ke sequence par based hai.

## Files

| File | Kya hai |
|---|---|
| `predictive_maintenance.csv` | Input data (10,000 rows) |
| `Predictive_Maintenance_prediction.ipynb` | Poora code: data load → training → prediction |
| `predictions.csv` | Test set (2000 rows) ke predictions |
| `README.md` | Ye file |

## Dataset

| Column | Matlab |
|---|---|
| `UDI`, `Product ID` | Row / machine ki ID (training me use nahi hoti) |
| `Type` | Machine quality: L (Low), M (Medium), H (High) |
| `Air temperature [K]` | Hawa ka temperature |
| `Process temperature [K]` | Process ka temperature |
| `Rotational speed [rpm]` | Ghumne ki speed |
| `Torque [Nm]` | Torque |
| `Tool wear [min]` | Tool kitne minute use hua |
| `Target` | **Predict karna hai**: 0 = No Failure, 1 = Failure |
| `Failure Type` | Failure ka type (training se hata diya gaya, neeche dekhein) |

Data imbalanced hai: 9,661 rows "No Failure" aur sirf 339 rows "Failure" (~3.4%).

## Notebook ka sequence

1. CSV load karna
2. `describe()` aur class count
3. Correlation heatmap
4. `StratifiedShuffleSplit` (80% train / 20% test, `Target` + `Type` par stratify)
5. Custom transformers:
   - `NullImputer`: missing numeric values ko median se bharta hai
   - `FeatureEncoder`: `Type` ko one-hot encode karta hai (`Type_H`, `Type_L`, `Type_M`)
   - `FeatureDropper`: `UDI`, `Product ID`, `Type`, `Failure Type` hatata hai
6. `Pipeline`
7. `StandardScaler`
8. `RandomForestClassifier` + `GridSearchCV`
9. Test set par evaluation (accuracy, classification report, confusion matrix, ROC-AUC, feature importance)
10. Poore data par final (production) model
11. `predictions.csv` banana
12. Naye machine ke liye `predict_failure(...)` function

## Important decisions

- **`class_weight="balanced"` aur `scoring="f1"`**: failure bahut kam hain, sirf accuracy dekhne se model sab ko "No Failure" bol ke bhi ~96.6% accuracy le leta.
- **`Failure Type` drop kiya**: ye answer ka hi hissa hai (data leakage), use rakhne se result jhootha achha aata.
- **Test set par sirf `transform`**: pipeline aur scaler sirf train par `fit` hote hain.
- **Alag `test.csv` nahi hai**, isliye predictions original data ke 20% held-out rows par nikale gaye hain.

## Result (test set, 2000 rows)

| Metric | Value |
|---|---|
| Accuracy | 97.4% |
| ROC-AUC | 0.96 |
| Failure precision | 0.61 |
| Failure recall | 0.63 |

Best params (train-only model): `max_depth=10`, `min_samples_split=4`, `n_estimators=100`.

## Run kaise karein

**Requirements:** Python 3, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Predictive_Maintenance_prediction.ipynb
```

1. `predictive_maintenance.csv` ko notebook ke same folder me rakhein (Colab me pehla cell upload maangega).
2. **Run all** karein. GridSearch me lagbhag 1 minute lagta hai.
3. `predictions.csv` usi folder me ban jayegi.

## `predictions.csv` ke columns

| Column | Matlab |
|---|---|
| `UDI`, `Product ID` | Machine ki ID |
| `Actual` | Asli value (0/1) |
| `Predicted` | Model ka prediction (0/1) |
| `Failure_Probability` | Failure hone ki probability (0 se 1) |

## Naye machine ka prediction

Notebook ke last cell me:

```python
predict_failure("L", 300.0, 311.0, 1300, 65.0, 230)
# arguments: Type, air_temp, process_temp, rpm, torque, tool_wear
```

Output me failure probability aur "Machine OK" ya "FAILURE LIKELY" aata hai (threshold 50%).

## Limitations

- Failure class ka recall ~63% hai, yaani kuch failures miss ho jate hain. Zyada failures pakadne ke liye probability threshold 0.5 se kam kar sakte hain (isse false alarms badhenge).
- Data me kuch rows ke `Target` aur `Failure Type` aapas me match nahi karte, ye original dataset ki property hai.
