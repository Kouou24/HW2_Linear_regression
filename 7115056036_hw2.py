"""
=============================================================
 線性回歸分析主程式 (雙資料集 CRISP-DM 一體化執行)
 HW2 — Linear Regression with CRISP-DM Methodology
=============================================================

支援雙資料集 (Dual Datasets):
  1. Student Performance Factors (學生考試成績預測, StudentPerformanceFactors.csv)
  2. WHO Life Expectancy Data (各國預期壽命分析, Life_Expectancy_Data.csv)

CRISP-DM 流程：
  Phase 1 - Business Understanding  : 業務理解
  Phase 2 - Data Understanding      : 資料理解 (EDA + 相關係數熱圖)
  Phase 3 - Data Preparation        : 資料準備 (缺失值填補 / One-Hot Encoding / Train-Test Split)
  Phase 4 - Modeling                : 模型建置
      Model 1: Simple Linear Regression     (單純線性回歸)
      Model 2: Full Multiple Linear Regression (多元全特徵 OLS)
      Model 3: Feature-Selected Multiple LR  (p-value 反向逐步淘汰 + Lasso + RFE)
      Model 4: ElasticNetCV (Auto Regularization) + Autoregressive / 結構基線分析
  Phase 5 - Evaluation              : 模型評估 (R^2, Adj-R^2, RMSE, MAE, Residual Diagnostics)
  Phase 6 - Deployment              : 推論介面 (含 95% CI 與 PI)

執行方式:
  python -X utf8 main.py
"""

# ── 標準函式庫 ──────────────────────────────────────────────────────────────
import os
import json
import warnings
warnings.filterwarnings('ignore')

# ── 數值與資料處理 ──────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── 視覺化 ──────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import seaborn as sns

# ── 統計建模 ────────────────────────────────────────────────────────────────
import statsmodels.regression.linear_model as sm_lm
import statsmodels.tools.tools as sm_tools
import statsmodels.nonparametric.api as sm_np
from scipy import stats

# ── 機器學習 ────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── 全域設定：中文字型與繪圖風格 ────────────────────────────────────────────
if 'seaborn-v0_8-whitegrid' in plt.style.available:
    plt.style.use('seaborn-v0_8-whitegrid')
elif 'seaborn-whitegrid' in plt.style.available:
    plt.style.use('seaborn-whitegrid')
else:
    plt.style.use('default')

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Microsoft YaHei', 'SimHei', 'Segoe UI', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

FIGURES_DIR  = 'figures'
MODELS_DIR   = 'models'
RANDOM_STATE = 42
TEST_SIZE    = 0.20

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)


# =============================================================================
# ── PART 1: 資料集一 學生學業成績 (Student Performance Factors) ───────────────
# =============================================================================

def run_student_performance_analysis():
    print("\n" + "#" * 70)
    print("  【資料集一】學生學業成績分析 (Student Performance Factors)")
    print("#" * 70)

    data_path = 'StudentPerformanceFactors.csv'
    if not os.path.exists(data_path):
        print(f"錯誤：找不到檔案 {data_path}")
        return

    # Phase 2 & 3: Data Understanding & Preparation
    df_raw = pd.read_csv(data_path)
    print(f"\n[Phase 2 & 3] 資料概況 Shape: {df_raw.shape}")
    df = df_raw.copy()
    for col in ['Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home']:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)

    # Correlation Heatmap
    num_cols = df.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(9, 7))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".3f", cbar=True, square=True)
    plt.title("學生數值特徵相關係數矩陣 (Student Correlation Heatmap)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/correlation_heatmap.png', dpi=300)
    plt.close()

    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop(columns=['Exam_Score']).astype(float)
    y = df_encoded['Exam_Score'].astype(float)
    full_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Model 1: Simple Linear Regression (Attendance)
    X_tr_s = sm_tools.add_constant(X_train[['Attendance']])
    X_te_s = sm_tools.add_constant(X_test[['Attendance']])
    simple_model = sm_lm.OLS(y_train, X_tr_s).fit()
    y_pred_s = simple_model.predict(X_te_s)
    metrics_s = {
        'r2': float(r2_score(y_test, y_pred_s)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_s))),
        'mae': float(mean_absolute_error(y_test, y_pred_s))
    }

    # Plot Simple LR
    x_grid = np.linspace(df['Attendance'].min(), df['Attendance'].max(), 300)
    X_grid = sm_tools.add_constant(pd.DataFrame({'Attendance': x_grid}))
    pred_grid = simple_model.get_prediction(X_grid).summary_frame(alpha=0.05)
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_test), size=min(400, len(X_test)), replace=False)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_test['Attendance'].iloc[sample_idx], y_test.iloc[sample_idx], alpha=0.35, color='#4A90E2', edgecolors='none', label='測試集觀測值')
    plt.plot(x_grid, pred_grid['mean'], color='#D0021B', lw=2.5, label='單純線性回歸線 (Fit Line)')
    plt.fill_between(x_grid, pred_grid['mean_ci_lower'], pred_grid['mean_ci_upper'], color='#F5A623', alpha=0.35, label='95% 信賴區間 (CI)')
    plt.fill_between(x_grid, pred_grid['obs_ci_lower'], pred_grid['obs_ci_upper'], color='#BD10E0', alpha=0.15, linestyle='--', label='95% 預測區間 (PI)')
    plt.title("單純線性回歸預測圖 (Attendance vs. Exam_Score) 含 95% CI 與 PI", fontsize=14, pad=12)
    plt.xlabel("學生出席率 (Attendance %)", fontsize=12)
    plt.ylabel("期末考試成績 (Exam_Score)", fontsize=12)
    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/simple_regression_ci_pi.png', dpi=300)
    plt.close()

    # Model 2: Full Multiple Linear Regression
    X_tr_f = sm_tools.add_constant(X_train)
    X_te_f = sm_tools.add_constant(X_test)
    full_model = sm_lm.OLS(y_train, X_tr_f).fit()
    y_pred_f = full_model.predict(X_te_f)
    r2_f = r2_score(y_test, y_pred_f)
    adj_r2_f = 1 - (1 - r2_f) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
    metrics_f = {
        'r2': float(r2_f),
        'adj_r2': float(adj_r2_f),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_f))),
        'mae': float(mean_absolute_error(y_test, y_pred_f))
    }

    # Model 3: Feature Selection
    pvalues = full_model.pvalues[1:]
    insig_features = pvalues[pvalues > 0.05].index.tolist()
    sel_features = [c for c in X_train.columns if c not in insig_features]
    X_tr_sel = sm_tools.add_constant(X_train[sel_features])
    X_te_sel = sm_tools.add_constant(X_test[sel_features])
    selected_model = sm_lm.OLS(y_train, X_tr_sel).fit()
    y_pred_sel = selected_model.predict(X_te_sel)
    r2_sel = r2_score(y_test, y_pred_sel)
    adj_r2_sel = 1 - (1 - r2_sel) * (len(y_test) - 1) / (len(y_test) - len(sel_features) - 1)
    metrics_sel = {
        'r2': float(r2_sel),
        'adj_r2': float(adj_r2_sel),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_sel))),
        'mae': float(mean_absolute_error(y_test, y_pred_sel))
    }

    # Model 4: ElasticNetCV & Autoregressive
    enet = ElasticNetCV(cv=5, random_state=RANDOM_STATE).fit(X_train, y_train)
    y_pred_enet = enet.predict(X_test)
    metrics_enet = {
        'r2': float(r2_score(y_test, y_pred_enet)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_enet))),
        'mae': float(mean_absolute_error(y_test, y_pred_enet))
    }
    model_no_prev = sm_lm.OLS(y_train, X_tr_sel.drop(columns=['Previous_Scores'])).fit()
    r2_no_prev = float(r2_score(y_test, model_no_prev.predict(X_te_sel.drop(columns=['Previous_Scores']))))

    # Diagnostic Plots
    std_coefs = pd.Series(
        selected_model.params[1:] * (X_train[sel_features].std() / y_train.std()),
        index=sel_features
    ).sort_values(ascending=True)
    plt.figure(figsize=(10, 8))
    colors = ['#D0021B' if c < 0 else '#4A90E2' for c in std_coefs]
    plt.barh(std_coefs.index, std_coefs.values, color=colors, height=0.65)
    plt.axvline(0, color='black', lw=0.8, linestyle='--')
    plt.title("學生標準化回歸係數 (Beta) - 特徵影響力排序", fontsize=14, pad=12)
    plt.xlabel("標準化係數 (Beta)", fontsize=12)
    plt.ylabel("特徵變數", fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/feature_importance.png', dpi=300)
    plt.close()

    # Multiple regression intervals plot
    test_preds = selected_model.get_prediction(X_te_sel).summary_frame(alpha=0.05)
    test_preds['actual'] = y_test.values
    sorted_df = test_preds.sort_values(by='mean').reset_index(drop=True)
    sub = sorted_df.iloc[::3]

    plt.figure(figsize=(11, 7))
    plt.scatter(range(len(sub)), sub['actual'], color='#4A90E2', alpha=0.5, s=18, label='實際考試成績')
    plt.plot(range(len(sub)), sub['mean'], color='#D0021B', lw=2.2, label='多元線性回歸預測值')
    plt.fill_between(range(len(sub)), sub['obs_ci_lower'], sub['obs_ci_upper'], color='#BD10E0', alpha=0.18, label='95% 預測區間 (PI)')
    plt.fill_between(range(len(sub)), sub['mean_ci_lower'], sub['mean_ci_upper'], color='#F5A623', alpha=0.35, label='95% 信賴區間 (CI)')
    plt.title("學生多元回歸預測成果與 95% 預測區間 (Actual vs. Predicted with 95% PI & CI)", fontsize=14, pad=12)
    plt.xlabel("樣本排序索引", fontsize=12)
    plt.ylabel("考試成績 (Exam_Score)", fontsize=12)
    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/multiple_regression_prediction_intervals.png', dpi=300)
    plt.close()

    # Residual Diagnostics
    residuals = selected_model.resid
    fitted = selected_model.fittedvalues
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes[0, 0].scatter(fitted, residuals, alpha=0.3, color='#4A90E2', s=16)
    axes[0, 0].axhline(0, color='#D0021B', linestyle='--', lw=1.5)
    lowess = sm_np.lowess(residuals, fitted, frac=0.2)
    axes[0, 0].plot(lowess[:, 0], lowess[:, 1], color='#F5A623', lw=2, label='LOWESS')
    axes[0, 0].set_title("(a) 殘差 vs. 擬合值 (Residuals vs Fitted)", fontsize=12)
    axes[0, 0].set_xlabel("擬合值")
    axes[0, 0].set_ylabel("殘差")
    axes[0, 0].legend()

    stats.probplot(residuals, dist='norm', plot=axes[0, 1])
    axes[0, 1].get_lines()[0].set_color('#4A90E2')
    axes[0, 1].get_lines()[0].set_alpha(0.3)
    axes[0, 1].get_lines()[1].set_color('#D0021B')
    axes[0, 1].set_title("(b) 常態 Q-Q 圖 (Normal Q-Q Plot)", fontsize=12)

    sns.histplot(residuals, kde=True, ax=axes[1, 0], color='#4A90E2', bins=40)
    axes[1, 0].set_title("(c) 殘差分配圖 (Residual Distribution)", fontsize=12)
    axes[1, 0].set_xlabel("殘差")
    axes[1, 0].set_ylabel("次數")

    min_v = min(float(y_test.min()), float(y_pred_sel.min()))
    max_v = max(float(y_test.max()), float(y_pred_sel.max()))
    axes[1, 1].scatter(y_test, y_pred_sel, alpha=0.35, color='#4A90E2', s=18)
    axes[1, 1].plot([min_v, max_v], [min_v, max_v], color='#D0021B', linestyle='--', lw=2, label='完美預測線 (y = x)')
    axes[1, 1].set_title("(d) 測試集真實值 vs. 預測值 (Actual vs. Predicted)", fontsize=12)
    axes[1, 1].set_xlabel("實際成績")
    axes[1, 1].set_ylabel("預測成績")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/residual_diagnostics.png', dpi=300)
    plt.close()

    # Model comparison plot
    model_names = ['單純線性 (Attendance)', '多元全特徵 (27變數)', '特徵選擇精簡 (24變數)', 'ElasticNetCV 自動化']
    r2_vals = [metrics_s['r2'], metrics_f['r2'], metrics_sel['r2'], metrics_enet['r2']]
    rmse_vals = [metrics_s['rmse'], metrics_f['rmse'], metrics_sel['rmse'], metrics_enet['rmse']]
    x_idx = np.arange(len(model_names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(x_idx - width/2, r2_vals, width, label='R^2 決定係數', color='#4A90E2')
    ax1.set_ylabel('R^2 Score', color='#4A90E2', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#4A90E2')
    ax1.set_ylim(0, 1.0)
    ax2 = ax1.twinx()
    ax2.bar(x_idx + width/2, rmse_vals, width, label='RMSE 均方根誤差', color='#D0021B')
    ax2.set_ylabel('RMSE (越低越好)', color='#D0021B', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#D0021B')
    ax2.set_ylim(0, 4.0)
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(model_names, fontsize=11)
    plt.title("學生模型不同架構效能比較 (Model Evaluation & Comparison)", fontsize=14, pad=15)
    fig.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/model_comparison.png', dpi=300)
    plt.close()

    # Save metrics JSON
    metrics_json = {
        'simple_regression': metrics_s,
        'full_multiple_regression': metrics_f,
        'selected_features_regression': metrics_sel,
        'elasticnet_cv': metrics_enet,
        'autoregressive_comparison': {'with_previous_scores_r2': metrics_sel['r2'], 'without_previous_scores_r2': r2_no_prev},
        'selected_features_count': len(sel_features),
        'dropped_features': insig_features,
        'params': selected_model.params.to_dict(),
        'residual_scale': float(selected_model.scale**0.5)
    }
    with open(f'{MODELS_DIR}/metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_json, f, indent=4, ensure_ascii=False)

    print("\n【學生資料集完成】成果摘要：")
    print(f"  單純線性回歸 (Attendance)  : R^2 = {metrics_s['r2']:.4f}, RMSE = {metrics_s['rmse']:.4f}")
    print(f"  多元全特徵線性回歸 (27變數)  : R^2 = {metrics_f['r2']:.4f}, RMSE = {metrics_f['rmse']:.4f}")
    print(f"  特徵選擇精簡回歸 (24變數) ★ : R^2 = {metrics_sel['r2']:.4f}, RMSE = {metrics_sel['rmse']:.4f}")


# =============================================================================
# ── PART 2: 資料集二 各國預期壽命 (WHO Life Expectancy Data) ─────────────────
# =============================================================================

def run_life_expectancy_analysis():
    print("\n" + "#" * 70)
    print("  【資料集二】世界衛生組織各國預期壽命分析 (WHO Life Expectancy)")
    print("#" * 70)

    data_path = 'Life_Expectancy_Data.csv'
    if not os.path.exists(data_path):
        print(f"錯誤：找不到檔案 {data_path}")
        return

    df = pd.read_csv(data_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=['Life expectancy']).reset_index(drop=True)
    df['Status_Developed'] = (df['Status'] == 'Developed').astype(int)
    features = [c for c in df.columns if c not in ['Country', 'Year', 'Status', 'Life expectancy']]

    for col in features:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    print(f"\n[Phase 2 & 3] 資料概況 Shape: {df.shape}, 特徵數: {len(features)}")

    # Correlation Heatmap
    plt.figure(figsize=(12, 10))
    corr = df[features + ['Life expectancy']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, square=True)
    plt.title('各國預期壽命數值特徵相關係數熱圖 (Life Expectancy Correlation Heatmap)', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/life_correlation_heatmap.png', dpi=300)
    plt.close()

    X = df[features].astype(float)
    y = df['Life expectancy'].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Model 1: Simple Linear Regression (Schooling)
    X_tr_s = sm_tools.add_constant(X_train[['Schooling']])
    X_te_s = sm_tools.add_constant(X_test[['Schooling']])
    simple_m = sm_lm.OLS(y_train, X_tr_s).fit()
    y_pred_s = simple_m.predict(X_te_s)
    metrics_s = {
        'r2': float(r2_score(y_test, y_pred_s)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_s))),
        'mae': float(mean_absolute_error(y_test, y_pred_s))
    }

    # Plot Simple LR
    x_grid = np.linspace(df['Schooling'].min(), df['Schooling'].max(), 300)
    X_grid_c = sm_tools.add_constant(pd.DataFrame({'Schooling': x_grid}))
    pred_grid = simple_m.get_prediction(X_grid_c).summary_frame(alpha=0.05)
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_test), size=min(400, len(X_test)), replace=False)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_test['Schooling'].iloc[sample_idx], y_test.iloc[sample_idx], alpha=0.35, color='#4A90E2', edgecolors='none', label='測試集觀測值')
    plt.plot(x_grid, pred_grid['mean'], color='#D0021B', lw=2.5, label='單純線性回歸線 (Fit Line)')
    plt.fill_between(x_grid, pred_grid['mean_ci_lower'], pred_grid['mean_ci_upper'], color='#F5A623', alpha=0.35, label='95% 信賴區間 (CI)')
    plt.fill_between(x_grid, pred_grid['obs_ci_lower'], pred_grid['obs_ci_upper'], color='#BD10E0', alpha=0.15, linestyle='--', label='95% 預測區間 (PI)')
    plt.title('單純線性回歸預測圖 (受教育年數 Schooling vs. 預期壽命) 含 95% CI 與 PI', fontsize=14, pad=12)
    plt.xlabel('平均受教育年數 (Schooling, 年)', fontsize=12)
    plt.ylabel('預期壽命 (Life Expectancy, 歲)', fontsize=12)
    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/life_simple_regression_ci_pi.png', dpi=300)
    plt.close()

    # Model 2: Full Multiple Linear Regression
    X_tr_f = sm_tools.add_constant(X_train)
    X_te_f = sm_tools.add_constant(X_test)
    full_m = sm_lm.OLS(y_train, X_tr_f).fit()
    y_pred_f = full_m.predict(X_te_f)
    r2_f = r2_score(y_test, y_pred_f)
    adj_r2_f = 1 - (1 - r2_f) * (len(y_test) - 1) / (len(y_test) - len(features) - 1)
    metrics_f = {
        'r2': float(r2_f),
        'adj_r2': float(adj_r2_f),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_f))),
        'mae': float(mean_absolute_error(y_test, y_pred_f))
    }

    # Model 3: Selected Features Model
    pvals = full_m.pvalues[1:]
    insig_features = pvals[pvals > 0.05].index.tolist()
    sel_features = [f for f in features if f not in insig_features]
    X_tr_sel = sm_tools.add_constant(X_train[sel_features])
    X_te_sel = sm_tools.add_constant(X_test[sel_features])
    sel_m = sm_lm.OLS(y_train, X_tr_sel).fit()
    y_pred_sel = sel_m.predict(X_te_sel)
    r2_sel = r2_score(y_test, y_pred_sel)
    adj_r2_sel = 1 - (1 - r2_sel) * (len(y_test) - 1) / (len(y_test) - len(sel_features) - 1)
    metrics_sel = {
        'r2': float(r2_sel),
        'adj_r2': float(adj_r2_sel),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_sel))),
        'mae': float(mean_absolute_error(y_test, y_pred_sel))
    }

    # Model 4: ElasticNetCV (Standardized)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(X_test)
    enet = ElasticNetCV(cv=5, random_state=RANDOM_STATE).fit(X_tr_scaled, y_train)
    y_pred_enet = enet.predict(X_te_scaled)
    metrics_enet = {
        'r2': float(r2_score(y_test, y_pred_enet)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_enet))),
        'mae': float(mean_absolute_error(y_test, y_pred_enet))
    }

    # Feature Importance Plot
    std_coefs = pd.Series(
        sel_m.params[1:] * (X_train[sel_features].std() / y_train.std()),
        index=sel_features
    ).sort_values(ascending=True)
    plt.figure(figsize=(10, 8))
    colors = ['#D0021B' if c < 0 else '#4A90E2' for c in std_coefs]
    plt.barh(std_coefs.index, std_coefs.values, color=colors, height=0.65)
    plt.axvline(0, color='black', lw=0.8, linestyle='--')
    plt.title('標準化回歸係數 (Standardized Beta) - 預期壽命影響力排序', fontsize=14, pad=12)
    plt.xlabel('標準化係數 (Beta)', fontsize=12)
    plt.ylabel('特徵變數', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/life_feature_importance.png', dpi=300)
    plt.close()

    # Multiple Regression Prediction with CI & PI
    test_preds = sel_m.get_prediction(X_te_sel).summary_frame(alpha=0.05)
    test_preds['actual'] = y_test.values
    sorted_df = test_preds.sort_values(by='mean').reset_index(drop=True)
    sub = sorted_df.iloc[::2]

    plt.figure(figsize=(11, 7))
    plt.scatter(range(len(sub)), sub['actual'], color='#4A90E2', alpha=0.5, s=18, label='實際預期壽命')
    plt.plot(range(len(sub)), sub['mean'], color='#D0021B', lw=2.2, label='多元線性回歸預測值')
    plt.fill_between(range(len(sub)), sub['obs_ci_lower'], sub['obs_ci_upper'], color='#BD10E0', alpha=0.18, label='95% 預測區間 (PI)')
    plt.fill_between(range(len(sub)), sub['mean_ci_lower'], sub['mean_ci_upper'], color='#F5A623', alpha=0.35, label='95% 信賴區間 (CI)')
    plt.title('多元回歸預測成果與 95% 預測區間 (Life Expectancy: Actual vs. Predicted with 95% PI & CI)', fontsize=14, pad=12)
    plt.xlabel('樣本排序索引 (各國樣本)', fontsize=12)
    plt.ylabel('預期壽命 (歲)', fontsize=12)
    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/life_multiple_regression_prediction_intervals.png', dpi=300)
    plt.close()

    # Residual Diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    resid = sel_m.resid
    fitted = sel_m.fittedvalues
    axes[0, 0].scatter(fitted, resid, alpha=0.3, color='#4A90E2', s=16)
    axes[0, 0].axhline(0, color='#D0021B', linestyle='--', lw=1.5)
    lowess = sm_np.lowess(resid, fitted, frac=0.2)
    axes[0, 0].plot(lowess[:, 0], lowess[:, 1], color='#F5A623', lw=2, label='LOWESS')
    axes[0, 0].set_title('(a) 殘差 vs. 擬合值 (Residuals vs Fitted)', fontsize=12)
    axes[0, 0].set_xlabel('擬合值')
    axes[0, 0].set_ylabel('殘差')
    axes[0, 0].legend()

    stats.probplot(resid, dist='norm', plot=axes[0, 1])
    axes[0, 1].get_lines()[0].set_color('#4A90E2')
    axes[0, 1].get_lines()[0].set_alpha(0.3)
    axes[0, 1].get_lines()[1].set_color('#D0021B')
    axes[0, 1].set_title('(b) 常態 Q-Q 圖 (Normal Q-Q Plot)', fontsize=12)

    sns.histplot(resid, kde=True, ax=axes[1, 0], color='#4A90E2', bins=40)
    axes[1, 0].set_title('(c) 殘差分配圖 (Residual Distribution)', fontsize=12)
    axes[1, 0].set_xlabel('殘差')
    axes[1, 0].set_ylabel('次數')

    min_v = min(float(y_test.min()), float(y_pred_sel.min()))
    max_v = max(float(y_test.max()), float(y_pred_sel.max()))
    axes[1, 1].scatter(y_test, y_pred_sel, alpha=0.35, color='#4A90E2', s=18)
    axes[1, 1].plot([min_v, max_v], [min_v, max_v], color='#D0021B', linestyle='--', lw=2, label='完美預測線 (y = x)')
    axes[1, 1].set_title('(d) 測試集真實值 vs. 預測值 (Actual vs. Predicted)', fontsize=12)
    axes[1, 1].set_xlabel('實際預期壽命')
    axes[1, 1].set_ylabel('預測預期壽命')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/life_residual_diagnostics.png', dpi=300)
    plt.close()

    # Model Comparison Plot
    models = ['單純線性 (Schooling)', '多元全特徵 (19變數)', '特徵選擇精簡 (14變數)', 'ElasticNetCV (標準化)']
    r2_list = [metrics_s['r2'], metrics_f['r2'], metrics_sel['r2'], metrics_enet['r2']]
    rmse_list = [metrics_s['rmse'], metrics_f['rmse'], metrics_sel['rmse'], metrics_enet['rmse']]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x_idx = np.arange(len(models))
    width = 0.35
    ax1.bar(x_idx - width/2, r2_list, width, label='R^2 決定係數', color='#4A90E2')
    ax1.set_ylabel('R^2 Score', color='#4A90E2', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#4A90E2')
    ax1.set_ylim(0, 1.0)
    ax2 = ax1.twinx()
    ax2.bar(x_idx + width/2, rmse_list, width, label='RMSE 均方根誤差', color='#D0021B')
    ax2.set_ylabel('RMSE (越低越好)', color='#D0021B', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#D0021B')
    ax2.set_ylim(0, 10.0)
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(models, fontsize=11)
    plt.title('預期壽命不同模型效能比較 (Model Evaluation & Comparison)', fontsize=14, pad=15)
    fig.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/life_model_comparison.png', dpi=300)
    plt.close()

    # Save metrics JSON
    life_metrics = {
        'simple_regression': metrics_s,
        'full_multiple_regression': metrics_f,
        'selected_features_regression': metrics_sel,
        'elasticnet_cv': metrics_enet,
        'selected_features': sel_features,
        'dropped_features': insig_features,
        'params': sel_m.params.to_dict(),
        'residual_scale': float(sel_m.scale**0.5)
    }
    with open(f'{MODELS_DIR}/life_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(life_metrics, f, indent=4, ensure_ascii=False)

    print("\n【預期壽命資料集完成】成果摘要：")
    print(f"  單純線性回歸 (Schooling)     : R^2 = {metrics_s['r2']:.4f}, RMSE = {metrics_s['rmse']:.4f}")
    print(f"  多元全特徵線性回歸 (19變數)  : R^2 = {metrics_f['r2']:.4f}, RMSE = {metrics_f['rmse']:.4f}")
    print(f"  特徵選擇精簡回歸 (14變數) ★ : R^2 = {metrics_sel['r2']:.4f}, RMSE = {metrics_sel['rmse']:.4f}")


# =============================================================================
# ── 主程式進入點 ─────────────────────────────────────────────────────────────
# =============================================================================

def main():
    print("=" * 70)
    print("  回歸分析主程式 (CRISP-DM 双資料集自動化管線)")
    print("=" * 70)

    # 1. 執行學生學業表現回歸分析
    run_student_performance_analysis()

    # 2. 執行世界衛生組織預期壽命回歸分析
    run_life_expectancy_analysis()

    print("\n" + "=" * 70)
    print("  雙資料集分析全數執行完畢！所有圖表已存至 ./figures/，指標存至 ./models/")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()

