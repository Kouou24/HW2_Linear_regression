"""
=============================================================
 StudentPerformanceFactors 線性回歸分析主程式
 HW2 — Linear Regression with CRISP-DM Methodology
=============================================================

CRISP-DM 流程：
  Phase 1 - Business Understanding  : 業務理解
  Phase 2 - Data Understanding      : 資料理解 (EDA + 相關係數熱圖)
  Phase 3 - Data Preparation        : 資料準備 (缺失值填補 / One-Hot Encoding / Train-Test Split)
  Phase 4 - Modeling                : 模型建置
      Model 1: Simple Linear Regression     (單純線性回歸, 特徵: Attendance)
      Model 2: Full Multiple Linear Regression (多元全特徵 OLS)
      Model 3: Feature-Selected Multiple LR  (p-value 反向逐步淘汰 + Lasso + RFE)
      Model 4: ElasticNetCV (Auto Regularization) + Autoregressive Feature Analysis
  Phase 5 - Evaluation              : 模型評估 (R^2, Adj-R^2, RMSE, MAE, Residual Diagnostics)
  Phase 6 - Deployment              : 推論介面 (predict_student, 含 95% CI 與 PI)

執行方式:
  python main.py
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
import statsmodels.api as sm

# ── 機器學習 ────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
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

# ── 常數設定 ─────────────────────────────────────────────────────────────────
DATA_PATH    = 'StudentPerformanceFactors.csv'
FIGURES_DIR  = 'figures'
MODELS_DIR   = 'models'
IMPUTE_COLS  = ['Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home']
TARGET_COL   = 'Exam_Score'
RANDOM_STATE = 42
TEST_SIZE    = 0.20


# =============================================================================
# ── Phase 2 & 3: 資料理解與準備 ─────────────────────────────────────────────
# =============================================================================

def load_and_prepare_data(filepath: str):
    """
    讀取原始 CSV、缺失值填補（眾數）、One-Hot Encoding、Train-Test Split。

    Returns
    -------
    df_raw      : 原始 DataFrame（未經 Encoding，用於後續繪圖）
    X_train, X_test, y_train, y_test : 訓練與測試集
    full_columns : 所有 One-Hot 後的特徵欄位名稱列表
    """
    print("=" * 65)
    print("  Phase 2 & 3 — Data Understanding & Data Preparation")
    print("=" * 65)

    df_raw = pd.read_csv(filepath)
    print(f"[1] 原始資料集維度 (Raw Dataset Shape): {df_raw.shape}")
    print(f"[2] 目標變數 '{TARGET_COL}' 描述統計:\n{df_raw[TARGET_COL].describe().to_string()}")

    # ── 缺失值填補 ────────────────────────────────────────────────────────
    df = df_raw.copy()
    print("\n[3] 缺失值填補 (Mode Imputation):")
    for col in IMPUTE_COLS:
        missing_n = df[col].isnull().sum()
        mode_val  = df[col].mode()[0]
        df[col]   = df[col].fillna(mode_val)
        print(f"    {col}: {missing_n} 筆缺失值 → 以眾數 '{mode_val}' 填補")

    # ── One-Hot Encoding ──────────────────────────────────────────────────
    df_encoded = pd.get_dummies(df, drop_first=True)   # drop_first 避免虛擬變數陷阱
    X = df_encoded.drop(columns=[TARGET_COL]).astype(float)
    y = df_encoded[TARGET_COL].astype(float)
    full_columns = X.columns.tolist()

    print(f"\n[4] One-Hot Encoding 後特徵維度: {X.shape[1]} 個特徵")

    # ── Train / Test Split ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[5] 資料集切分 (80/20): 訓練集 {X_train.shape[0]} 筆 / 測試集 {X_test.shape[0]} 筆")

    # ── 相關係數熱圖 ──────────────────────────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(9, 7))
    corr_matrix = df[num_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", cbar=True, square=True)
    plt.title("數值特徵相關係數矩陣 (Correlation Heatmap)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/correlation_heatmap.png', dpi=300)
    plt.close()
    print(f"\n[6] 數值相關係數矩陣熱圖 → 已儲存: {FIGURES_DIR}/correlation_heatmap.png")

    corr_with_target = corr_matrix[TARGET_COL].drop(TARGET_COL).sort_values(ascending=False)
    print(f"\n[7] 與 '{TARGET_COL}' 的 Pearson 相關係數排序:\n{corr_with_target.to_string()}")

    return df_raw, df, X_train, X_test, y_train, y_test, full_columns


# =============================================================================
# ── Phase 4 Model 1: Simple Linear Regression ────────────────────────────────
# =============================================================================

def fit_simple_regression(X_train, X_test, y_train, y_test, df_raw):
    """
    單純線性回歸：選取相關係數最高之單一特徵 Attendance。
    繪製含 95% CI 與 95% PI 之散佈圖。

    Returns
    -------
    simple_model : 已訓練之 statsmodels OLS 模型
    metrics      : dict (r2, rmse, mae)
    """
    print("\n" + "=" * 65)
    print("  Phase 4 — Model 1: Simple Linear Regression (單純線性回歸)")
    print("  特徵: Attendance (與 Exam_Score 相關係數最高, r = 0.581)")
    print("=" * 65)

    X_tr = sm.add_constant(X_train[['Attendance']])
    X_te = sm.add_constant(X_test[['Attendance']])

    simple_model = sm.OLS(y_train, X_tr).fit()
    print(simple_model.summary())

    # ── 測試集評估指標 ─────────────────────────────────────────────────
    y_pred = simple_model.predict(X_te)
    metrics = {
        'r2'  : r2_score(y_test, y_pred),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'mae' : float(mean_absolute_error(y_test, y_pred))
    }
    print(f"\n測試集評估 — R^2: {metrics['r2']:.4f} | RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f}")

    # ── 繪圖：迴歸線 + 95% CI + 95% PI ────────────────────────────────
    x_grid    = np.linspace(df_raw['Attendance'].min(), df_raw['Attendance'].max(), 300)
    X_grid    = sm.add_constant(pd.DataFrame({'Attendance': x_grid}))
    pred_grid = simple_model.get_prediction(X_grid).summary_frame(alpha=0.05)

    # 隨機取 400 個測試點顯示（視覺整潔）
    rng        = np.random.RandomState(RANDOM_STATE)
    sample_idx = rng.choice(len(X_test), size=min(400, len(X_test)), replace=False)

    plt.figure(figsize=(10, 6))
    plt.scatter(
        X_test['Attendance'].iloc[sample_idx], y_test.iloc[sample_idx],
        alpha=0.35, color='#4A90E2', edgecolors='none',
        label='測試集觀測值 (Test Sample)'
    )
    plt.plot(x_grid, pred_grid['mean'], color='#D0021B', lw=2.5,
             label='單純線性回歸線 (Fit Line)')
    plt.fill_between(x_grid, pred_grid['mean_ci_lower'], pred_grid['mean_ci_upper'],
                     color='#F5A623', alpha=0.35,
                     label='95% 信賴區間 (CI: 期望值區間)')
    plt.fill_between(x_grid, pred_grid['obs_ci_lower'], pred_grid['obs_ci_upper'],
                     color='#BD10E0', alpha=0.15,
                     label='95% 預測區間 (PI: 個別觀測值區間)')
    plt.title("單純線性回歸預測圖 (Attendance vs. Exam_Score) 含 95% CI 與 PI",
              fontsize=14, pad=12)
    plt.xlabel("學生出席率 (Attendance %)", fontsize=12)
    plt.ylabel("期末考試成績 (Exam_Score)", fontsize=12)
    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/simple_regression_ci_pi.png', dpi=300)
    plt.close()
    print(f"預測圖 → 已儲存: {FIGURES_DIR}/simple_regression_ci_pi.png")

    return simple_model, metrics


# =============================================================================
# ── Phase 4 Model 2: Full Multiple Linear Regression ─────────────────────────
# =============================================================================

def fit_full_multiple_regression(X_train, X_test, y_train, y_test):
    """
    多元線性回歸（全特徵 27 個變數）OLS。

    Returns
    -------
    full_model : 已訓練之 statsmodels OLS 模型
    metrics    : dict (r2, adj_r2, rmse, mae)
    """
    print("\n" + "=" * 65)
    print("  Phase 4 — Model 2: Full Multiple Linear Regression (多元全特徵 OLS)")
    print("=" * 65)

    X_tr = sm.add_constant(X_train)
    X_te = sm.add_constant(X_test)

    full_model = sm.OLS(y_train, X_tr).fit()

    y_pred  = full_model.predict(X_te)
    r2      = r2_score(y_test, y_pred)
    adj_r2  = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
    metrics = {
        'r2'    : r2,
        'adj_r2': adj_r2,
        'rmse'  : float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'mae'   : float(mean_absolute_error(y_test, y_pred))
    }
    print(f"測試集評估 — R^2: {metrics['r2']:.4f} | Adj-R^2: {metrics['adj_r2']:.4f} "
          f"| RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f}")

    return full_model, metrics


# =============================================================================
# ── Phase 4 Model 3 & 4: Feature Selection + Auto Regression ─────────────────
# =============================================================================

def fit_selected_and_auto_regression(full_model, X_train, X_test, y_train, y_test):
    """
    特徵選擇精簡模型（p-value 反向淘汰 + Lasso L1 + RFE 三種方法）
    以及 Auto Regression（ElasticNetCV 自動正則化 + Previous_Scores 自回歸項消融實驗）。

    Returns
    -------
    selected_model        : 精簡後 statsmodels OLS 模型
    selected_features     : 精簡後保留的特徵名稱列表
    metrics_sel           : dict (r2, adj_r2, rmse, mae)
    metrics_enet          : dict (r2, rmse, mae)
    r2_no_prev            : float (移除 Previous_Scores 後的 R^2)
    insignificant_features: 被剔除的特徵列表
    """
    print("\n" + "=" * 65)
    print("  Phase 4 — Feature Selection (特徵選擇) — 三種方法")
    print("=" * 65)

    # ── 方法 A: p-value 反向逐步淘汰法 ───────────────────────────────
    pvalues = full_model.pvalues[1:]   # 排除截距項
    insignificant_features = pvalues[pvalues > 0.05].index.tolist()
    selected_features      = [c for c in X_train.columns if c not in insignificant_features]
    print(f"[方法 A] 反向逐步淘汰 (p > 0.05)，剔除特徵 ({len(insignificant_features)} 個):")
    for f in insignificant_features:
        print(f"         ✗ {f}  (p = {pvalues[f]:.4f})")

    # ── 方法 B: Lasso L1 正則化 ───────────────────────────────────────
    lasso_cv = LassoCV(cv=5, random_state=RANDOM_STATE).fit(X_train, y_train)
    lasso_coefs = pd.Series(lasso_cv.coef_, index=X_train.columns)
    lasso_zeroed = lasso_coefs[lasso_coefs.abs() <= 1e-4].index.tolist()
    print(f"\n[方法 B] Lasso L1 (CV=5) 自動收縮至 0 之特徵 ({len(lasso_zeroed)} 個): {lasso_zeroed}")

    # ── 方法 C: Recursive Feature Elimination (RFE, top 15) ──────────
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=15)
    rfe.fit(X_train, y_train)
    rfe_dropped = X_train.columns[~rfe.support_].tolist()
    print(f"\n[方法 C] RFE (n=15) 建議剔除 ({len(rfe_dropped)} 個): {rfe_dropped}")
    print(f"\n→ 最終採用方法 A (p-value) 結果：保留 {len(selected_features)} 個特徵進行精簡模型訓練。")

    # ── 精簡模型訓練 ─────────────────────────────────────────────────
    X_tr_sel = sm.add_constant(X_train[selected_features])
    X_te_sel = sm.add_constant(X_test[selected_features])
    selected_model = sm.OLS(y_train, X_tr_sel).fit()

    y_pred_sel = selected_model.predict(X_te_sel)
    r2_sel     = r2_score(y_test, y_pred_sel)
    adj_r2_sel = 1 - (1 - r2_sel) * (len(y_test) - 1) / (len(y_test) - len(selected_features) - 1)
    metrics_sel = {
        'r2'    : r2_sel,
        'adj_r2': adj_r2_sel,
        'rmse'  : float(np.sqrt(mean_squared_error(y_test, y_pred_sel))),
        'mae'   : float(mean_absolute_error(y_test, y_pred_sel))
    }
    print(f"\n精簡模型測試集評估 — R^2: {metrics_sel['r2']:.4f} | Adj-R^2: {metrics_sel['adj_r2']:.4f} "
          f"| RMSE: {metrics_sel['rmse']:.4f} | MAE: {metrics_sel['mae']:.4f}")

    # ── Phase 4 Model 4A: ElasticNetCV ────────────────────────────────
    print("\n" + "=" * 65)
    print("  Phase 4 — Model 4A: ElasticNetCV (自動化正則化回歸)")
    print("=" * 65)
    enet_cv      = ElasticNetCV(cv=5, random_state=RANDOM_STATE).fit(X_train, y_train)
    y_pred_enet  = enet_cv.predict(X_test)
    metrics_enet = {
        'r2'  : r2_score(y_test, y_pred_enet),
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_enet))),
        'mae' : float(mean_absolute_error(y_test, y_pred_enet))
    }
    print(f"ElasticNetCV 最佳 alpha={enet_cv.alpha_:.4f}, l1_ratio={enet_cv.l1_ratio_:.2f}")
    print(f"測試集評估 — R^2: {metrics_enet['r2']:.4f} | RMSE: {metrics_enet['rmse']:.4f} "
          f"| MAE: {metrics_enet['mae']:.4f}")

    # ── Phase 4 Model 4B: 自回歸項消融實驗 ───────────────────────────
    print("\n" + "=" * 65)
    print("  Phase 4 — Model 4B: Autoregressive Feature Analysis")
    print("  (Previous_Scores 作為歷史自回歸項之消融實驗)")
    print("=" * 65)
    X_tr_no_prev = X_tr_sel.drop(columns=['Previous_Scores'])
    X_te_no_prev = X_te_sel.drop(columns=['Previous_Scores'])
    model_no_prev = sm.OLS(y_train, X_tr_no_prev).fit()
    r2_no_prev    = r2_score(y_test, model_no_prev.predict(X_te_no_prev))
    print(f"含 Previous_Scores (自回歸項) R^2 = {r2_sel:.4f}")
    print(f"不含 Previous_Scores           R^2 = {r2_no_prev:.4f}")
    print(f"→ Previous_Scores 邊際貢獻: Δ R^2 = {r2_sel - r2_no_prev:.4f}")
    print(f"→ 即便移除過往成績，當前學習行為仍解釋 {r2_no_prev:.1%} 的成績變異！")

    # 以 best_model 的相關物件回傳給後續推論使用
    return (selected_model, selected_features, metrics_sel,
            metrics_enet, r2_no_prev, insignificant_features)


# =============================================================================
# ── Phase 5: 評估圖表 ─────────────────────────────────────────────────────────
# =============================================================================

def generate_evaluation_plots(selected_model, selected_features,
                               X_train, X_test, y_train, y_test,
                               metrics_simple, metrics_full, metrics_sel, metrics_enet):
    """
    產生四張評估圖：
      1. 標準化係數（特徵影響力排序）
      2. 多元回歸預測圖（含 95% CI 與 PI）
      3. 殘差四合一診斷圖
      4. 各模型效能比較柱狀圖
    """
    print("\n" + "=" * 65)
    print("  Phase 5 — Evaluation: 生成診斷與評估圖表")
    print("=" * 65)

    X_te_sel = sm.add_constant(X_test[selected_features])

    # ── 圖1: 標準化回歸係數（特徵影響力排序）──────────────────────
    std_coefs = pd.Series(
        selected_model.params[1:] * (X_train[selected_features].std() / y_train.std()),
        index=selected_features
    ).sort_values(ascending=True)

    colors = ['#D0021B' if c < 0 else '#4A90E2' for c in std_coefs]
    plt.figure(figsize=(10, 8))
    plt.barh(std_coefs.index, std_coefs.values, color=colors, height=0.65)
    plt.axvline(0, color='black', lw=0.8, linestyle='--')
    plt.title("標準化回歸係數 (Standardized Beta Coefficients) - 特徵影響力排序",
              fontsize=14, pad=12)
    plt.xlabel("標準化係數 (Standardized Coefficients, Beta)", fontsize=12)
    plt.ylabel("特徵變數 (Features)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/feature_importance.png', dpi=300)
    plt.close()
    print(f"圖1 特徵影響力 → 已儲存: {FIGURES_DIR}/feature_importance.png")

    # ── 圖2: 多元回歸預測圖（含 95% CI 與 PI）───────────────────────
    test_preds  = selected_model.get_prediction(X_te_sel).summary_frame(alpha=0.05)
    test_preds['actual'] = y_test.values
    sorted_df   = test_preds.sort_values(by='mean').reset_index(drop=True)
    sub         = sorted_df.iloc[::3]    # 每 3 筆取一筆，避免過密

    plt.figure(figsize=(11, 7))
    plt.scatter(range(len(sub)), sub['actual'],
                color='#4A90E2', alpha=0.5, s=18, label='實際考試成績 (Actual Score)')
    plt.plot(range(len(sub)), sub['mean'],
             color='#D0021B', lw=2.2, label='多元線性回歸預測值 (Predicted)')
    plt.fill_between(range(len(sub)), sub['obs_ci_lower'], sub['obs_ci_upper'],
                     color='#BD10E0', alpha=0.18, label='95% 預測區間 (PI)')
    plt.fill_between(range(len(sub)), sub['mean_ci_lower'], sub['mean_ci_upper'],
                     color='#F5A623', alpha=0.35, label='95% 信賴區間 (CI)')
    plt.title("多元回歸預測成果與 95% 預測區間 "
              "(Multiple Regression: Actual vs. Predicted with 95% PI & CI)",
              fontsize=14, pad=12)
    plt.xlabel("樣本排序索引 (Sorted Student Instances by Predicted Score)", fontsize=12)
    plt.ylabel("考試成績 (Exam_Score)", fontsize=12)
    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/multiple_regression_prediction_intervals.png', dpi=300)
    plt.close()
    print(f"圖2 多元回歸預測圖 → 已儲存: {FIGURES_DIR}/multiple_regression_prediction_intervals.png")

    # ── 圖3: 殘差四合一診斷圖 ────────────────────────────────────────
    residuals = selected_model.resid
    fitted    = selected_model.fittedvalues
    y_pred_te = selected_model.predict(X_te_sel)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # (a) Residuals vs Fitted
    axes[0, 0].scatter(fitted, residuals, alpha=0.3, color='#4A90E2', s=16)
    axes[0, 0].axhline(0, color='#D0021B', linestyle='--', lw=1.5)
    lowess = sm.nonparametric.lowess(residuals, fitted, frac=0.2)
    axes[0, 0].plot(lowess[:, 0], lowess[:, 1], color='#F5A623', lw=2,
                    label='局部加權迴歸 (LOWESS)')
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

    # (d) Actual vs Predicted (Test Set)
    min_v = min(float(y_test.min()), float(y_pred_te.min()))
    max_v = max(float(y_test.max()), float(y_pred_te.max()))
    axes[1, 1].scatter(y_test, y_pred_te, alpha=0.35, color='#4A90E2', s=18)
    axes[1, 1].plot([min_v, max_v], [min_v, max_v],
                    color='#D0021B', linestyle='--', lw=2, label='完美預測線 (y = x)')
    axes[1, 1].set_title("(d) 測試集真實值 vs. 預測值 (Actual vs. Predicted)", fontsize=12)
    axes[1, 1].set_xlabel("實際成績 (Actual Exam_Score)")
    axes[1, 1].set_ylabel("預測成績 (Predicted Exam_Score)")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/residual_diagnostics.png', dpi=300)
    plt.close()
    print(f"圖3 殘差診斷 → 已儲存: {FIGURES_DIR}/residual_diagnostics.png")

    # ── 圖4: 模型效能比較 ─────────────────────────────────────────────
    model_names = ['單純線性\n(Attendance)', '多元全特徵\n(27變數)',
                   '特徵選擇精簡\n(24變數)', 'ElasticNetCV\n自動化']
    r2_vals   = [metrics_simple['r2'], metrics_full['r2'],
                 metrics_sel['r2'],    metrics_enet['r2']]
    rmse_vals = [metrics_simple['rmse'], metrics_full['rmse'],
                 metrics_sel['rmse'],    metrics_enet['rmse']]

    x_idx = np.arange(len(model_names))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(x_idx - width / 2, r2_vals, width, label='R^2 決定係數', color='#4A90E2')
    ax1.set_ylabel('R^2 Score', color='#4A90E2', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#4A90E2')
    ax1.set_ylim(0, 1.0)

    ax2 = ax1.twinx()
    ax2.bar(x_idx + width / 2, rmse_vals, width, label='RMSE 均方根誤差', color='#D0021B')
    ax2.set_ylabel('RMSE (越低越好)', color='#D0021B', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#D0021B')
    ax2.set_ylim(0, 4.0)

    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(model_names, fontsize=11)
    plt.title("不同模型與特徵選擇效能比較 (Model Evaluation & Comparison)",
              fontsize=14, pad=15)
    fig.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/model_comparison.png', dpi=300)
    plt.close()
    print(f"圖4 模型比較 → 已儲存: {FIGURES_DIR}/model_comparison.png")


# =============================================================================
# ── Phase 6: Deployment — 推論介面 ────────────────────────────────────────────
# =============================================================================

def predict_student(student_data: dict,
                    model, selected_features: list, full_columns: list,
                    alpha: float = 0.05) -> dict:
    """
    Phase 6 — Deployment: 推論函式

    給定一筆學生屬性字典，使用已訓練的最佳模型進行預測，
    並返回點預測值、標準誤、95% 信賴區間 (CI) 與 95% 預測區間 (PI)。

    Parameters
    ----------
    student_data    : 學生特徵字典（原始值，未 Encoding）
    model           : 已訓練之 statsmodels OLS 模型
    selected_features : 精簡模型所使用的特徵名稱列表
    full_columns    : 訓練資料 One-Hot Encoding 後的完整特徵欄位順序
    alpha           : 顯著水準，預設 0.05 (對應 95% CI 與 PI)

    Returns
    -------
    dict: {
        'predicted_score'      : float,      # 預測成績
        'standard_error'       : float,      # 預測標準誤
        'confidence_interval_95': (lower, upper),  # 95% CI (期望值區間)
        'prediction_interval_95': (lower, upper)   # 95% PI (個別學生區間)
    }

    學生屬性字典範例:
    {
        'Hours_Studied': 25, 'Attendance': 90,
        'Parental_Involvement': 'High', 'Access_to_Resources': 'High',
        'Extracurricular_Activities': 'Yes', 'Sleep_Hours': 7,
        'Previous_Scores': 85, 'Motivation_Level': 'High',
        'Internet_Access': 'Yes', 'Tutoring_Sessions': 2,
        'Family_Income': 'High', 'Teacher_Quality': 'High',
        'School_Type': 'Public', 'Peer_Influence': 'Positive',
        'Physical_Activity': 3, 'Learning_Disabilities': 'No',
        'Parental_Education_Level': 'Postgraduate',
        'Distance_from_Home': 'Near', 'Gender': 'Female'
    }
    """
    # One-Hot Encode 並對齊訓練集欄位順序
    raw_df     = pd.DataFrame([student_data])
    encoded_df = pd.get_dummies(raw_df)
    aligned_df = pd.DataFrame(0.0, index=[0], columns=full_columns)
    for col in encoded_df.columns:
        if col in aligned_df.columns:
            aligned_df[col] = float(encoded_df[col].iloc[0])

    X_input       = aligned_df[selected_features].copy()
    X_input_const = sm.add_constant(X_input, has_constant='add')

    pred_res = model.get_prediction(X_input_const)
    sf       = pred_res.summary_frame(alpha=alpha)

    return {
        'predicted_score'       : round(float(sf['mean'].iloc[0]), 2),
        'standard_error'        : round(float(sf['mean_se'].iloc[0]), 3),
        'confidence_interval_95': (round(float(sf['mean_ci_lower'].iloc[0]), 2),
                                   round(float(sf['mean_ci_upper'].iloc[0]), 2)),
        'prediction_interval_95': (round(float(sf['obs_ci_lower'].iloc[0]), 2),
                                   round(float(sf['obs_ci_upper'].iloc[0]), 2))
    }


def demo_deployment(model, selected_features, full_columns):
    """展示兩個典型學生案例的推論結果（Phase 6 部署展示）。"""
    print("\n" + "=" * 65)
    print("  Phase 6 — Deployment: 推論展示 (predict_student)")
    print("=" * 65)

    # ── 案例 A：高學習投入學生 ────────────────────────────────────────
    student_a = {
        'Hours_Studied': 28, 'Attendance': 95,
        'Parental_Involvement': 'High', 'Access_to_Resources': 'High',
        'Extracurricular_Activities': 'Yes', 'Sleep_Hours': 8,
        'Previous_Scores': 90, 'Motivation_Level': 'High',
        'Internet_Access': 'Yes', 'Tutoring_Sessions': 3,
        'Family_Income': 'High', 'Teacher_Quality': 'High',
        'School_Type': 'Public', 'Peer_Influence': 'Positive',
        'Physical_Activity': 4, 'Learning_Disabilities': 'No',
        'Parental_Education_Level': 'Postgraduate',
        'Distance_from_Home': 'Near', 'Gender': 'Female'
    }
    res_a = predict_student(student_a, model, selected_features, full_columns)
    print("\n[範例一：高學習投入學生]")
    print(f"  預測分數            : {res_a['predicted_score']} 分")
    print(f"  95% 信賴區間 (CI)   : {res_a['confidence_interval_95']}  (母體期望值估計範圍)")
    print(f"  95% 預測區間 (PI)   : {res_a['prediction_interval_95']}  (此學生個別得分範圍)")

    # ── 案例 B：高風險/需輔導學生 ─────────────────────────────────────
    student_b = {
        'Hours_Studied': 10, 'Attendance': 65,
        'Parental_Involvement': 'Low', 'Access_to_Resources': 'Low',
        'Extracurricular_Activities': 'No', 'Sleep_Hours': 6,
        'Previous_Scores': 55, 'Motivation_Level': 'Low',
        'Internet_Access': 'No', 'Tutoring_Sessions': 0,
        'Family_Income': 'Low', 'Teacher_Quality': 'Low',
        'School_Type': 'Public', 'Peer_Influence': 'Negative',
        'Physical_Activity': 1, 'Learning_Disabilities': 'Yes',
        'Parental_Education_Level': 'High School',
        'Distance_from_Home': 'Far', 'Gender': 'Male'
    }
    res_b = predict_student(student_b, model, selected_features, full_columns)
    print("\n[範例二：潛在高風險/需輔導學生]")
    print(f"  預測分數            : {res_b['predicted_score']} 分")
    print(f"  95% 信賴區間 (CI)   : {res_b['confidence_interval_95']}  (母體期望值估計範圍)")
    print(f"  95% 預測區間 (PI)   : {res_b['prediction_interval_95']}  (此學生個別得分範圍)")
    print("\n說明:")
    print("  CI (Confidence Interval) = 估計「同類學生群體平均分數」的範圍（較窄）")
    print("  PI (Prediction Interval) = 估計「此單一學生實際得分」的範圍（包含個體隨機誤差，較寬）")


# =============================================================================
# ── 儲存評估指標至 JSON ────────────────────────────────────────────────────────
# =============================================================================

def save_metrics(metrics_simple, metrics_full, metrics_sel, metrics_enet,
                 r2_sel, r2_no_prev, selected_features, insignificant_features):
    """將所有模型評估指標輸出成 models/metrics.json。"""
    metrics_dict = {
        'simple_regression'          : metrics_simple,
        'full_multiple_regression'   : metrics_full,
        'selected_features_regression': metrics_sel,
        'elasticnet_cv'              : metrics_enet,
        'autoregressive_comparison'  : {
            'with_previous_scores_r2'   : r2_sel,
            'without_previous_scores_r2': r2_no_prev
        },
        'selected_features_count'    : len(selected_features),
        'dropped_features'           : insignificant_features
    }
    out_path = f'{MODELS_DIR}/metrics.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=4, ensure_ascii=False)
    print(f"\n評估指標 → 已儲存: {out_path}")


# =============================================================================
# ── 主程式入口 ────────────────────────────────────────────────────────────────
# =============================================================================

def main():
    print("\n" + "=" * 65)
    print("  學生學業成績線性回歸分析主程式")
    print("  HW2 — Linear Regression (CRISP-DM)")
    print("=" * 65)

    # 建立輸出目錄
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR,  exist_ok=True)

    # ── Phase 2 & 3: 資料理解與準備 ─────────────────────────────────
    df_raw, df, X_train, X_test, y_train, y_test, full_columns = \
        load_and_prepare_data(DATA_PATH)

    # ── Phase 4 Model 1: 單純線性回歸 ────────────────────────────────
    simple_model, metrics_simple = \
        fit_simple_regression(X_train, X_test, y_train, y_test, df_raw)

    # ── Phase 4 Model 2: 多元全特徵回歸 ──────────────────────────────
    full_model, metrics_full = \
        fit_full_multiple_regression(X_train, X_test, y_train, y_test)

    # ── Phase 4 Model 3 & 4: 特徵選擇 + Auto Regression ─────────────
    (selected_model, selected_features, metrics_sel,
     metrics_enet, r2_no_prev, insignificant_features) = \
        fit_selected_and_auto_regression(full_model, X_train, X_test, y_train, y_test)

    # ── Phase 5: 生成評估圖表 ────────────────────────────────────────
    generate_evaluation_plots(
        selected_model, selected_features,
        X_train, X_test, y_train, y_test,
        metrics_simple, metrics_full, metrics_sel, metrics_enet
    )

    # ── 儲存評估指標 ─────────────────────────────────────────────────
    save_metrics(metrics_simple, metrics_full, metrics_sel, metrics_enet,
                 metrics_sel['r2'], r2_no_prev,
                 selected_features, insignificant_features)

    # ── Phase 6: 推論展示 ────────────────────────────────────────────
    demo_deployment(selected_model, selected_features, full_columns)

    # ── 最終摘要 ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  ✅  分析完成 — 成果摘要")
    print("=" * 65)
    print(f"  {'模型':<30} {'R^2':>7}  {'Adj-R^2':>8}  {'RMSE':>7}  {'MAE':>7}")
    print(f"  {'-'*60}")
    print(f"  {'單純線性 (Attendance)':<30} {metrics_simple['r2']:>7.4f}  {'—':>8}  "
          f"{metrics_simple['rmse']:>7.4f}  {metrics_simple['mae']:>7.4f}")
    print(f"  {'多元全特徵 OLS (27 vars)':<30} {metrics_full['r2']:>7.4f}  "
          f"{metrics_full['adj_r2']:>8.4f}  {metrics_full['rmse']:>7.4f}  {metrics_full['mae']:>7.4f}")
    print(f"  {'特徵選擇精簡 OLS (24 vars) ★':<30} {metrics_sel['r2']:>7.4f}  "
          f"{metrics_sel['adj_r2']:>8.4f}  {metrics_sel['rmse']:>7.4f}  {metrics_sel['mae']:>7.4f}")
    print(f"  {'ElasticNetCV 自動化':<30} {metrics_enet['r2']:>7.4f}  {'—':>8}  "
          f"{metrics_enet['rmse']:>7.4f}  {metrics_enet['mae']:>7.4f}")
    print(f"\n  輸出圖表: ./{FIGURES_DIR}/")
    print(f"  評估指標: ./{MODELS_DIR}/metrics.json")
    print("=" * 65 + "\n")


if __name__ == '__main__':
    main()
