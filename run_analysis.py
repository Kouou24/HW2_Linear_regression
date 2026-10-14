import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Set style and Chinese font support
if 'seaborn-v0_8-whitegrid' in plt.style.available:
    plt.style.use('seaborn-v0_8-whitegrid')
elif 'seaborn-whitegrid' in plt.style.available:
    plt.style.use('seaborn-whitegrid')
else:
    plt.style.use('default')

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'Segoe UI', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('figures', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=== Step 1: Loading and Exploring Data ===")
df_raw = pd.read_csv('StudentPerformanceFactors.csv')
print(f"Dataset Shape: {df_raw.shape}")

# Missing Value Imputation
# Missing: Teacher_Quality (78), Parental_Education_Level (90), Distance_from_Home (67)
df = df_raw.copy()
for col in ['Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home']:
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)
    print(f"Imputed {col} with mode: '{mode_val}'")

# Correlation Heatmap
num_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(9, 7))
corr_matrix = df[num_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", cbar=True, square=True)
plt.title("數值特徵相關係數矩陣 (Correlation Heatmap)", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig('figures/correlation_heatmap.png', dpi=300)
plt.close()
print("Saved figures/correlation_heatmap.png")

# Data Encoding
# Distinguish Ordinal vs Nominal
# For pure linear regression interpretability, we create dummy variables (drop_first=True)
df_encoded = pd.get_dummies(df, drop_first=True)

# Separate Target and Features
X = df_encoded.drop(columns=['Exam_Score']).astype(float)
y = df_encoded['Exam_Score'].astype(float)

# Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

print("\n=== Step 2: Model 1 - Simple Linear Regression (單純線性回歸) ===")
# We select Attendance as it has the highest correlation with Exam_Score (r = 0.581)
X_train_simple = sm.add_constant(X_train[['Attendance']])
X_test_simple = sm.add_constant(X_test[['Attendance']])

simple_model = sm.OLS(y_train, X_train_simple).fit()
print(simple_model.summary())

# Predict and intervals for test set
y_test_pred_simple = simple_model.predict(X_test_simple)
rmse_simple = np.sqrt(mean_squared_error(y_test, y_test_pred_simple))
mae_simple = mean_absolute_error(y_test, y_test_pred_simple)
r2_simple = r2_score(y_test, y_test_pred_simple)

# Plot Simple Linear Regression with CI and PI
# Generate sorted grid for smooth bands
x_grid = np.linspace(df['Attendance'].min(), df['Attendance'].max(), 300)
X_grid_const = sm.add_constant(pd.DataFrame({'Attendance': x_grid}))
pred_grid = simple_model.get_prediction(X_grid_const)
pred_grid_summary = pred_grid.summary_frame(alpha=0.05)

plt.figure(figsize=(10, 6))
# Sample a subset for clean scatter display
sample_idx = np.random.RandomState(42).choice(len(X_test), size=400, replace=False)
plt.scatter(X_test['Attendance'].iloc[sample_idx], y_test.iloc[sample_idx], alpha=0.35, color='#4A90E2', edgecolors='none', label='測試集觀測值 (Test Sample)')

plt.plot(x_grid, pred_grid_summary['mean'], color='#D0021B', lw=2.5, label='單純線性回歸線 (Fit Line)')
plt.fill_between(x_grid, pred_grid_summary['mean_ci_lower'], pred_grid_summary['mean_ci_upper'], color='#F5A623', alpha=0.35, label='95% 信賴區間 (CI: 期望值區間)')
plt.fill_between(x_grid, pred_grid_summary['obs_ci_lower'], pred_grid_summary['obs_ci_upper'], color='#BD10E0', alpha=0.15, linestyle='--', label='95% 預測區間 (PI: 個別觀測值區間)')

plt.title("單純線性回歸預測圖 (Attendance vs. Exam_Score) 含 95% CI 與 PI", fontsize=14, pad=12)
plt.xlabel("學生出席率 (Attendance %)", fontsize=12)
plt.ylabel("期末考試成績 (Exam_Score)", fontsize=12)
plt.legend(loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig('figures/simple_regression_ci_pi.png', dpi=300)
plt.close()
print("Saved figures/simple_regression_ci_pi.png")

print("\n=== Step 3: Model 2 - Full Multiple Linear Regression (多元線性回歸) ===")
X_train_full_const = sm.add_constant(X_train)
X_test_full_const = sm.add_constant(X_test)
full_model = sm.OLS(y_train, X_train_full_const).fit()

y_test_pred_full = full_model.predict(X_test_full_const)
rmse_full = np.sqrt(mean_squared_error(y_test, y_test_pred_full))
mae_full = mean_absolute_error(y_test, y_test_pred_full)
r2_full = r2_score(y_test, y_test_pred_full)
adj_r2_full = 1 - (1 - r2_full) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

print(f"Full Model - R2: {r2_full:.4f}, Adj R2: {adj_r2_full:.4f}, RMSE: {rmse_full:.4f}, MAE: {mae_full:.4f}")

print("\n=== Step 4: Feature Selection (特徵選擇) ===")
# Method A: Stepwise / Backward Elimination based on p-value (> 0.05)
pvalues = full_model.pvalues[1:] # ignore const
insignificant_features = pvalues[pvalues > 0.05].index.tolist()
print(f"Features removed by Backward Elimination (p > 0.05): {insignificant_features}")
selected_features_pvalue = [col for col in X_train.columns if col not in insignificant_features]

# Method B: Lasso L1 Regularization (Automated Feature Selection)
lasso_cv = LassoCV(cv=5, random_state=42).fit(X_train, y_train)
lasso_coefs = pd.Series(lasso_cv.coef_, index=X_train.columns)
lasso_selected_features = lasso_coefs[lasso_coefs.abs() > 1e-4].index.tolist()
print(f"Lasso selected {len(lasso_selected_features)} features (out of {X_train.shape[1]}). Zeroed out: {list(set(X_train.columns) - set(lasso_selected_features))}")

# Method C: Recursive Feature Elimination (RFE) selecting top 15 features
rfe = RFE(estimator=LinearRegression(), n_features_to_select=15)
rfe.fit(X_train, y_train)
rfe_selected_features = X_train.columns[rfe.support_].tolist()

# Fit Selected Model (Model 3: Reduced OLS using Stepwise/p-value selected features)
X_train_sel_const = sm.add_constant(X_train[selected_features_pvalue])
X_test_sel_const = sm.add_constant(X_test[selected_features_pvalue])
selected_model = sm.OLS(y_train, X_train_sel_const).fit()

y_test_pred_sel = selected_model.predict(X_test_sel_const)
rmse_sel = np.sqrt(mean_squared_error(y_test, y_test_pred_sel))
mae_sel = mean_absolute_error(y_test, y_test_pred_sel)
r2_sel = r2_score(y_test, y_test_pred_sel)
adj_r2_sel = 1 - (1 - r2_sel) * (len(y_test) - 1) / (len(y_test) - len(selected_features_pvalue) - 1)

print(f"Selected Model - R2: {r2_sel:.4f}, Adj R2: {adj_r2_sel:.4f}, RMSE: {rmse_sel:.4f}, MAE: {mae_sel:.4f}")

print("\n=== Step 5: Model 4 - Auto Regression / Automated Regularization & Autoregressive Feature Analysis ===")
# A) ElasticNetCV (Automated regularized regression)
enet_cv = ElasticNetCV(cv=5, random_state=42).fit(X_train, y_train)
y_test_pred_enet = enet_cv.predict(X_test)
rmse_enet = np.sqrt(mean_squared_error(y_test, y_test_pred_enet))
mae_enet = mean_absolute_error(y_test, y_test_pred_enet)
r2_enet = r2_score(y_test, y_test_pred_enet)

# B) Autoregressive Perspective: Model without Previous_Scores vs Model with Previous_Scores
# (Previous_Scores serves as the lag/autoregressive signal of historical academic attainment)
X_train_no_prev = X_train_sel_const.drop(columns=['Previous_Scores'])
X_test_no_prev = X_test_sel_const.drop(columns=['Previous_Scores'])
model_no_prev = sm.OLS(y_train, X_train_no_prev).fit()
y_test_pred_no_prev = model_no_prev.predict(X_test_no_prev)
r2_no_prev = r2_score(y_test, y_test_pred_no_prev)
print(f"Autoregressive Impact: Model with Previous_Scores R2={r2_sel:.4f} vs without Previous_Scores R2={r2_no_prev:.4f}")

# Save the best model and pipeline components
joblib.dump({
    'model': selected_model,
    'features': selected_features_pvalue,
    'full_columns': X.columns.tolist()
}, 'models/best_linear_model.joblib')
print("Saved models/best_linear_model.joblib")

print("\n=== Step 6: Generating Diagnostic & Prediction Plots ===")

# Plot 1: Feature Importance (Standardized Coefficients)
# Compute standardized coefficients for comparison
std_coefs = pd.Series(
    selected_model.params[1:] * (X_train[selected_features_pvalue].std() / y_train.std()),
    index=selected_features_pvalue
).sort_values(ascending=True)

plt.figure(figsize=(10, 8))
colors = ['#D0021B' if c < 0 else '#4A90E2' for c in std_coefs]
bars = plt.barh(std_coefs.index, std_coefs.values, color=colors, height=0.65)
plt.axvline(0, color='black', lw=0.8, linestyle='--')
plt.title("標準化回歸係數 (Standardized Beta Coefficients) - 特徵影響力排序", fontsize=14, pad=12)
plt.xlabel("標準化係數 (Standardized Coefficients, Beta)", fontsize=12)
plt.ylabel("特徵變數 (Features)", fontsize=12)
plt.tight_layout()
plt.savefig('figures/feature_importance.png', dpi=300)
plt.close()
print("Saved figures/feature_importance.png")

# Plot 2: Multiple Regression Prediction vs Actual with 95% Prediction Interval Band
test_preds = selected_model.get_prediction(X_test_sel_const)
test_summary = test_preds.summary_frame(alpha=0.05)
test_summary['actual'] = y_test.values

# Sort by predicted value for ordered curve presentation
test_summary_sorted = test_summary.sort_values(by='mean').reset_index(drop=True)

plt.figure(figsize=(11, 7))
sub_sample = test_summary_sorted.iloc[::3] # downsample smoothly for visual clarity

plt.scatter(range(len(sub_sample)), sub_sample['actual'], color='#4A90E2', alpha=0.5, s=18, label='實際考試成績 (Actual Score)')
plt.plot(range(len(sub_sample)), sub_sample['mean'], color='#D0021B', lw=2.2, label='多元線性回歸預測值 (Predicted)')
plt.fill_between(range(len(sub_sample)), sub_sample['obs_ci_lower'], sub_sample['obs_ci_upper'],
                 color='#BD10E0', alpha=0.18, label='95% 預測區間 (95% Prediction Interval, PI)')
plt.fill_between(range(len(sub_sample)), sub_sample['mean_ci_lower'], sub_sample['mean_ci_upper'],
                 color='#F5A623', alpha=0.35, label='95% 信賴區間 (95% Confidence Interval, CI)')

plt.title("多元回歸預測成果與 95% 預測區間 (Multiple Regression: Actual vs. Predicted with 95% PI & CI)", fontsize=14, pad=12)
plt.xlabel("樣本排序索引 (Sorted Student Instances by Predicted Score)", fontsize=12)
plt.ylabel("考試成績 (Exam_Score)", fontsize=12)
plt.legend(loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig('figures/multiple_regression_prediction_intervals.png', dpi=300)
plt.close()
print("Saved figures/multiple_regression_prediction_intervals.png")

# Plot 3: Residual Diagnostics (Residuals vs Fitted, Q-Q Plot, Distribution, Outlier impact)
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# (a) Residuals vs Fitted
residuals = selected_model.resid
fitted = selected_model.fittedvalues
axes[0, 0].scatter(fitted, residuals, alpha=0.3, color='#4A90E2', s=16)
axes[0, 0].axhline(0, color='#D0021B', linestyle='--', lw=1.5)
# Add lowess trend line
lowess = sm.nonparametric.lowess(residuals, fitted, frac=0.2)
axes[0, 0].plot(lowess[:, 0], lowess[:, 1], color='#F5A623', lw=2, label='局部加權迴歸 (LOWESS)')
axes[0, 0].set_title("(a) 殘差 vs. 擬合值 (Residuals vs Fitted)", fontsize=12)
axes[0, 0].set_xlabel("擬合值 (Fitted Values)")
axes[0, 0].set_ylabel("殘差 (Residuals)")
axes[0, 0].legend()

# (b) Normal Q-Q Plot
sm.qqplot(residuals, line='45', ax=axes[0, 1], color='#4A90E2', alpha=0.3)
axes[0, 1].set_title("(b) 常態 Q-Q 圖 (Normal Q-Q Plot)", fontsize=12)

# (c) Residual Histogram
sns.histplot(residuals, kde=True, ax=axes[1, 0], color='#4A90E2', bins=40)
axes[1, 0].set_title("(c) 殘差分配圖 (Residual Distribution)", fontsize=12)
axes[1, 0].set_xlabel("殘差 (Residuals)")
axes[1, 0].set_ylabel("次數 (Count)")

# (d) Predicted vs Actual Scatter
axes[1, 1].scatter(y_test, y_test_pred_sel, alpha=0.35, color='#4A90E2', s=18)
min_val = min(y_test.min(), y_test_pred_sel.min())
max_val = max(y_test.max(), y_test_pred_sel.max())
axes[1, 1].plot([min_val, max_val], [min_val, max_val], color='#D0021B', linestyle='--', lw=2, label='完美預測線 (y = x)')
axes[1, 1].set_title("(d) 測試集真實值 vs. 預測值 (Actual vs. Predicted)", fontsize=12)
axes[1, 1].set_xlabel("實際成績 (Actual Exam_Score)")
axes[1, 1].set_ylabel("預測成績 (Predicted Exam_Score)")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('figures/residual_diagnostics.png', dpi=300)
plt.close()
print("Saved figures/residual_diagnostics.png")

# Plot 4: Feature Selection Comparison
fs_df = pd.DataFrame({
    'Model': ['單純線性 (Attendance)', '多元全特徵 (27變數)', '特徵選擇精簡 (24變數)', 'ElasticNetCV 自動化'],
    'R2_Score': [r2_simple, r2_full, r2_sel, r2_enet],
    'RMSE': [rmse_simple, rmse_full, rmse_sel, rmse_enet],
    'MAE': [mae_simple, mae_full, mae_sel, mae_enet]
})

fig, ax1 = plt.subplots(figsize=(10, 5))
x_idx = np.arange(len(fs_df))
width = 0.35

rects1 = ax1.bar(x_idx - width/2, fs_df['R2_Score'], width, label='R² 決定係數', color='#4A90E2')
ax1.set_ylabel('R² Score', color='#4A90E2', fontsize=12)
ax1.tick_params(axis='y', labelcolor='#4A90E2')
ax1.set_ylim(0, 1.0)

ax2 = ax1.twinx()
rects2 = ax2.bar(x_idx + width/2, fs_df['RMSE'], width, label='RMSE 均方根誤差', color='#D0021B')
ax2.set_ylabel('RMSE (越低越好)', color='#D0021B', fontsize=12)
ax2.tick_params(axis='y', labelcolor='#D0021B')
ax2.set_ylim(0, 4.0)

ax1.set_xticks(x_idx)
ax1.set_xticklabels(fs_df['Model'], fontsize=11)
plt.title("不同模型與特徵選擇效能比較 (Model Evaluation & Comparison)", fontsize=14, pad=15)
fig.tight_layout()
plt.savefig('figures/model_comparison.png', dpi=300)
plt.close()
print("Saved figures/model_comparison.png")

# Save evaluation metrics json
metrics = {
    'simple_regression': {'r2': r2_simple, 'rmse': rmse_simple, 'mae': mae_simple},
    'full_multiple_regression': {'r2': r2_full, 'adj_r2': adj_r2_full, 'rmse': rmse_full, 'mae': mae_full},
    'selected_features_regression': {'r2': r2_sel, 'adj_r2': adj_r2_sel, 'rmse': rmse_sel, 'mae': mae_sel},
    'elasticnet_cv': {'r2': r2_enet, 'rmse': rmse_enet, 'mae': mae_enet},
    'autoregressive_comparison': {'with_previous_scores_r2': r2_sel, 'without_previous_scores_r2': r2_no_prev},
    'selected_features_count': len(selected_features_pvalue),
    'dropped_features': insignificant_features
}
with open('models/metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=4, ensure_ascii=False)
print("Saved models/metrics.json")
print("Analysis successfully completed!")
