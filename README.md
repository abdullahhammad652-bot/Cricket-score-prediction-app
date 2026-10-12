# 🏏 T20 Cricket Score Predictor

A machine learning project that predicts the **final score of a T20 innings** from the current match situation, with an interactive Streamlit web app.

**🔗 Live demo:** _add your Streamlit link here_

![T20 Cricket Score Predictor app screenshot](screenshot.png)

---

## 📌 Overview

Given details such as overs played, wickets lost and current run rate, the model estimates how many runs the batting side will finish with. The project covers the full workflow: data cleaning, outlier handling, encoding, model training, residual diagnosis, model improvement and deployment.

## ✨ Features

- 🎯 **Live prediction** with a score gauge, likely range and verdict
- 📊 **Analysis tab** with interactive charts (score vs run rate, overs, wickets, pitch × weather, overs × wickets heatmap)
- 🔥 **Sensitivity cards** showing the effect of +1 run rate, +1 wicket or +1 over
- 🏟️ Team selection with a modern dark UI

## 🧠 Model

| Model | R² (test) | MAE (runs) |
|---|---|---|
| Linear Regression | 0.915 | 15.64 |
| **Linear Regression + degree-2 polynomial features** | **0.997** | **2.52** |

The residual plot of the plain linear model showed a clear U-shaped pattern, which means the relationship between the inputs and the score is non-linear. Adding degree-2 polynomial features (which include interactions such as Overs × Run Rate) removed most of that pattern and cut the average error from about 16 runs to about 2.5 runs.

## 📂 Dataset

`T20_cricket.csv` with 1,500 rows and these columns:

| Column | Description |
|---|---|
| Overs Played | Overs completed (0–20) |
| Wickets Lost | Wickets down (0–10) |
| Run Rate | Current run rate |
| Home/Away | Venue |
| Opponent Strength | Rating from 1 to 10 |
| Pitch Condition | Balanced, Batting or Bowling |
| Weather | Sunny, Cloudy or Overcast |
| Predicted Score | **Target**: final innings score |

## 🔧 Methodology

1. Dropped rows with missing values and checked for duplicates
2. Detected outliers with the **IQR rule** and replaced them with the **median**
3. Dropped the identifier column (Match ID)
4. One-hot encoded categorical columns (`drop_first=True`)
5. 80/20 train/test split (`random_state=42`)
6. Trained a Linear Regression baseline and evaluated with R², MAE and MSE
7. Checked for overfitting (train R² 0.922 vs test R² 0.915)
8. Diagnosed non-linearity with a residual plot
9. Improved the model with polynomial features
10. Saved the model with `joblib` and deployed it with Streamlit

## 🚀 Run locally

```bash
git clone https://github.com/abdullahhammad652-bot/cricket-score-predictor.git
cd cricket-score-predictor
pip install -r requirements.txt
streamlit run main.py
```

## 🗂️ Project structure

```
cricket-score-predictor/
├── main.py            # Streamlit app
├── model.pkl          # Trained polynomial regression model
├── poly.pkl           # Polynomial feature transformer
├── columns.pkl        # Feature column order
├── requirements.txt
├── screenshot.png
└── README.md
```

## 🛠️ Tech stack

Python · pandas · NumPy · scikit-learn · Altair · Streamlit · joblib

## ⚠️ Limitations

- Team names in the app are for display only. The model was **not** trained on team identity.
- Scores above about 200 are rare in the data, so predictions there are slightly less accurate (the residuals spread out for high scores).
- The dataset appears to be simulated rather than real match data, so the app is a learning and demonstration project, not a betting or forecasting tool.

## 🔮 Future improvements

- Train on real ball-by-ball data (for example from Cricsheet)
- Compare with Ridge, Lasso and Random Forest
- Add team and player features
- Add confidence intervals to predictions

## 👤 Author

**Hammad Abdullah**
[GitHub](https://github.com/abdullahhammad652-bot)