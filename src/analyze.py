"""
Exp6: statistical analysis, model development, validation, performance evaluation.
Reads results/sweep_dataset.csv and results/exp1_cooperation_over_time.csv.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style="whitegrid", font_scale=1.0)
PALETTE = {"baseline_no_mechanism": "#D64545", "with_mechanism": "#2E7D5B"}

df = pd.read_csv("results/sweep_dataset.csv")
exp1 = pd.read_csv("results/exp1_cooperation_over_time.csv")

report = {}

# ---------- Descriptive statistics ----------
desc = df["steady_cooperation_rate"].describe()
report["descriptive_stats_cooperation_rate"] = desc.to_dict()
print("Descriptive stats (steady cooperation rate):")
print(desc)

# ---------- EDA plots ----------
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(df["steady_cooperation_rate"], bins=20, color="#2E7D5B", edgecolor="white")
ax.set_xlabel("Steady-state cooperation rate")
ax.set_ylabel("Number of parameter combinations")
ax.set_title("Distribution of steady-state cooperation rate across sweep")
fig.tight_layout()
fig.savefig("results/figures/fig1_coop_rate_histogram.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(df["mechanism_strength_p"], df["steady_cooperation_rate"],
           alpha=0.5, color="#2E7D5B")
ax.set_xlabel("Mechanism strength (tariff, p)")
ax.set_ylabel("Steady-state cooperation rate")
ax.set_title("Cooperation rate vs. mechanism strength")
fig.tight_layout()
fig.savefig("results/figures/fig2_coop_vs_mechanism.png", dpi=150)
plt.close(fig)

corr_cols = ["mechanism_strength_p", "benefit_cost_ratio", "titfortat_share",
             "steady_cooperation_rate", "steady_welfare"]
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(6.5, 5.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
            vmin=-1, vmax=1)
ax.set_title("Correlation between swept parameters and outcomes")
fig.tight_layout()
fig.savefig("results/figures/fig3_correlation_heatmap.png", dpi=150)
plt.close(fig)
report["correlation_matrix"] = corr.round(3).to_dict()

# baseline vs mechanism cooperation-over-time (Experiment 1)
fig, ax = plt.subplots(figsize=(7, 4.5))
for scenario, color in PALETTE.items():
    sub = exp1[exp1["scenario"] == scenario]
    label = "Baseline (no mechanism)" if scenario == "baseline_no_mechanism" else "With tariff mechanism (p=0.6)"
    ax.plot(sub["round"], sub["cooperation_rate"], label=label, color=color, linewidth=2)
ax.set_xlabel("Round")
ax.set_ylabel("Cooperation rate (mean over 20 runs)")
ax.set_title("Cooperation rate over time: baseline vs. mechanism")
ax.legend()
fig.tight_layout()
fig.savefig("results/figures/fig0_baseline_vs_mechanism.png", dpi=150)
plt.close(fig)

baseline_final = exp1[(exp1["scenario"] == "baseline_no_mechanism") & (exp1["round"] >= 100)]["cooperation_rate"].mean()
mechanism_final = exp1[(exp1["scenario"] == "with_mechanism") & (exp1["round"] >= 100)]["cooperation_rate"].mean()
report["baseline_vs_mechanism_steady_state"] = {
    "baseline_no_mechanism": round(float(baseline_final), 4),
    "with_mechanism_p_0.6": round(float(mechanism_final), 4),
}
print(f"\nBaseline steady-state cooperation: {baseline_final:.3f}")
print(f"With mechanism (p=0.6) steady-state cooperation: {mechanism_final:.3f}")

# ---------- Inferential: correlation significance ----------
from scipy import stats
r, pval = stats.pearsonr(df["mechanism_strength_p"], df["steady_cooperation_rate"])
report["pearson_p_vs_cooprate"] = {"r": round(float(r), 4), "p_value": float(pval)}
print(f"\nPearson correlation (mechanism strength vs cooperation rate): r={r:.3f}, p={pval:.2e}")

# ---------- Model development ----------
features = ["mechanism_strength_p", "benefit_cost_ratio", "titfortat_share"]
target = "steady_cooperation_rate"
X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regression": RandomForestRegressor(n_estimators=300, random_state=42, max_depth=6),
}

model_results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(model, X, y, cv=kf, scoring="r2")

    model_results[name] = {
        "test_MAE": round(float(mae), 4),
        "test_RMSE": round(float(rmse), 4),
        "test_R2": round(float(r2), 4),
        "5fold_CV_R2_mean": round(float(cv_r2.mean()), 4),
        "5fold_CV_R2_std": round(float(cv_r2.std()), 4),
    }
    print(f"\n{name}: MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f} "
          f"| 5-fold CV R2={cv_r2.mean():.4f} (+/- {cv_r2.std():.4f})")

report["model_comparison"] = model_results

# best model = highest test R2
best_name = max(model_results, key=lambda k: model_results[k]["test_R2"])
best_model = models[best_name]
report["best_model"] = best_name

# Actual vs predicted plot for best model
preds_best = best_model.predict(X_test)
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, preds_best, alpha=0.6, color="#2E7D5B")
lims = [min(y_test.min(), preds_best.min()), max(y_test.max(), preds_best.max())]
ax.plot(lims, lims, "--", color="#D64545", label="Perfect prediction")
ax.set_xlabel("Actual cooperation rate")
ax.set_ylabel("Predicted cooperation rate")
ax.set_title(f"Actual vs. predicted ({best_name})")
ax.legend()
fig.tight_layout()
fig.savefig("results/figures/fig4_actual_vs_predicted.png", dpi=150)
plt.close(fig)

# Feature importance (Random Forest)
rf = models["Random Forest Regression"]
importances = dict(zip(features, rf.feature_importances_))
report["feature_importance_random_forest"] = {k: round(float(v), 4) for k, v in importances.items()}

fig, ax = plt.subplots(figsize=(7, 4.5))
feat_names = list(importances.keys())
feat_vals = [importances[f] for f in feat_names]
ax.barh(feat_names, feat_vals, color="#2E7D5B")
ax.set_xlabel("Feature importance")
ax.set_title("Drivers of cooperation (Random Forest importance)")
fig.tight_layout()
fig.savefig("results/figures/fig5_feature_importance.png", dpi=150)
plt.close(fig)

with open("results/exp6_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\nAll figures and exp6_report.json saved to results/")
