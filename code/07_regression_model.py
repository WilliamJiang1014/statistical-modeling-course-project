"""
阶段2：回归模型（住院天数预测）
功能：多元线性回归模型构建、评估与诊断
作者：成员B
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 加载数据 ====================
print("=" * 60)
print("阶段2：回归模型开发（住院天数预测）")
print("=" * 60)

print("\n[1] 加载数据...")
with open('data/processed/data_splits.pkl', 'rb') as f:
    data_splits = pickle.load(f)

reg_data = data_splits['regression']
X_train = reg_data['X_train'].copy()
X_test = reg_data['X_test'].copy()
y_train = reg_data['y_train'].copy()
y_test = reg_data['y_test'].copy()
features_base = reg_data['features_base']
features_enhanced = reg_data['features_enhanced']

print(f"训练集: {len(X_train):,} 样本")
print(f"测试集: {len(X_test):,} 样本")

# ==================== 2. 特征编码 ====================
def encode_features(X, encoders=None, fit=True):
    """对分类变量进行编码"""
    X_encoded = X.copy()
    if encoders is None:
        encoders = {}
    
    # 识别分类变量
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        if fit:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
        else:
            if col in encoders:
                # 处理未见过的类别
                X_encoded[col] = X[col].astype(str).apply(
                    lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else 0
                )
    
    return X_encoded, encoders

print("\n[2] 特征编码...")
X_train_encoded, encoders = encode_features(X_train, fit=True)
X_test_encoded, _ = encode_features(X_test, encoders=encoders, fit=False)

# ==================== 3. 基准模型 ====================
print("\n[3] 构建基准模型...")

# 使用基准特征集
X_train_base = X_train_encoded[features_base].copy()
X_test_base = X_test_encoded[features_base].copy()

# 标准化（可选，线性回归通常不需要，但有助于解释）
scaler_base = StandardScaler()
X_train_base_scaled = scaler_base.fit_transform(X_train_base)
X_test_base_scaled = scaler_base.transform(X_test_base)

# 训练模型
model_base = LinearRegression()
model_base.fit(X_train_base_scaled, y_train)

# 预测
y_train_pred_base = model_base.predict(X_train_base_scaled)
y_test_pred_base = model_base.predict(X_test_base_scaled)

# 评估
train_r2_base = r2_score(y_train, y_train_pred_base)
test_r2_base = r2_score(y_test, y_test_pred_base)
train_rmse_base = np.sqrt(mean_squared_error(y_train, y_train_pred_base))
test_rmse_base = np.sqrt(mean_squared_error(y_test, y_test_pred_base))
train_mae_base = mean_absolute_error(y_train, y_train_pred_base)
test_mae_base = mean_absolute_error(y_test, y_test_pred_base)

print("\n基准模型评估结果:")
print(f"训练集 - R²: {train_r2_base:.4f}, RMSE: {train_rmse_base:.4f}, MAE: {train_mae_base:.4f}")
print(f"测试集 - R²: {test_r2_base:.4f}, RMSE: {test_rmse_base:.4f}, MAE: {test_mae_base:.4f}")

# ==================== 4. 改进模型 ====================
print("\n[4] 构建改进模型...")

X_train_enhanced = X_train_encoded[features_enhanced].copy()
X_test_enhanced = X_test_encoded[features_enhanced].copy()

scaler_enhanced = StandardScaler()
X_train_enhanced_scaled = scaler_enhanced.fit_transform(X_train_enhanced)
X_test_enhanced_scaled = scaler_enhanced.transform(X_test_enhanced)

model_enhanced = LinearRegression()
model_enhanced.fit(X_train_enhanced_scaled, y_train)

y_train_pred_enhanced = model_enhanced.predict(X_train_enhanced_scaled)
y_test_pred_enhanced = model_enhanced.predict(X_test_enhanced_scaled)

train_r2_enhanced = r2_score(y_train, y_train_pred_enhanced)
test_r2_enhanced = r2_score(y_test, y_test_pred_enhanced)
train_rmse_enhanced = np.sqrt(mean_squared_error(y_train, y_train_pred_enhanced))
test_rmse_enhanced = np.sqrt(mean_squared_error(y_test, y_test_pred_enhanced))
train_mae_enhanced = mean_absolute_error(y_train, y_train_pred_enhanced)
test_mae_enhanced = mean_absolute_error(y_test, y_test_pred_enhanced)

print("\n改进模型评估结果:")
print(f"训练集 - R²: {train_r2_enhanced:.4f}, RMSE: {train_rmse_enhanced:.4f}, MAE: {train_mae_enhanced:.4f}")
print(f"测试集 - R²: {test_r2_enhanced:.4f}, RMSE: {test_rmse_enhanced:.4f}, MAE: {test_mae_enhanced:.4f}")

# ==================== 5. 模型诊断 ====================
print("\n[5] 模型诊断...")

# 使用改进模型进行诊断
residuals = y_test - y_test_pred_enhanced

# 5.1 残差正态性检验
shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000])  # Shapiro-Wilk检验（样本量大时用前5000个）
print(f"\n残差正态性检验 (Shapiro-Wilk):")
print(f"  统计量: {shapiro_stat:.4f}, p值: {shapiro_p:.4f}")

# 5.2 多重共线性检查（VIF）
print(f"\n多重共线性检查 (VIF):")
vif_data = pd.DataFrame()
vif_data["Variable"] = features_enhanced
vif_data["VIF"] = [variance_inflation_factor(X_train_enhanced_scaled, i) 
                   for i in range(X_train_enhanced_scaled.shape[1])]
vif_data = vif_data.sort_values('VIF', ascending=False)
print(vif_data.head(10))
print(f"\nVIF > 10 的变量数: {(vif_data['VIF'] > 10).sum()}")

# ==================== 6. 结果可视化 ====================
print("\n[6] 生成可视化图表...")

os.makedirs('docs/figures', exist_ok=True)

# 6.1 模型系数条形图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 基准模型系数
coef_base = pd.DataFrame({
    'Feature': features_base,
    'Coefficient': model_base.coef_
})
coef_base = coef_base.sort_values('Coefficient', key=abs, ascending=False).head(15)

axes[0].barh(range(len(coef_base)), coef_base['Coefficient'])
axes[0].set_yticks(range(len(coef_base)))
axes[0].set_yticklabels(coef_base['Feature'])
axes[0].set_xlabel('回归系数')
axes[0].set_title('基准模型 - 主要变量系数')
axes[0].axvline(x=0, color='black', linestyle='--', linewidth=0.8)

# 改进模型系数
coef_enhanced = pd.DataFrame({
    'Feature': features_enhanced,
    'Coefficient': model_enhanced.coef_
})
coef_enhanced = coef_enhanced.sort_values('Coefficient', key=abs, ascending=False).head(15)

axes[1].barh(range(len(coef_enhanced)), coef_enhanced['Coefficient'])
axes[1].set_yticks(range(len(coef_enhanced)))
axes[1].set_yticklabels(coef_enhanced['Feature'])
axes[1].set_xlabel('回归系数')
axes[1].set_title('改进模型 - 主要变量系数')
axes[1].axvline(x=0, color='black', linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.savefig('docs/figures/07_regression_coefficients.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: docs/figures/07_regression_coefficients.png")

# 6.2 残差诊断图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 残差vs预测值
axes[0, 0].scatter(y_test_pred_enhanced, residuals, alpha=0.5, s=10)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('预测值')
axes[0, 0].set_ylabel('残差')
axes[0, 0].set_title('残差 vs 预测值')

# Q-Q图
stats.probplot(residuals, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('残差Q-Q图（正态性检验）')

# 残差直方图
axes[1, 0].hist(residuals, bins=50, edgecolor='black')
axes[1, 0].set_xlabel('残差')
axes[1, 0].set_ylabel('频数')
axes[1, 0].set_title('残差分布')

# 预测值vs实际值
axes[1, 1].scatter(y_test, y_test_pred_enhanced, alpha=0.5, s=10)
axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 1].set_xlabel('实际值')
axes[1, 1].set_ylabel('预测值')
axes[1, 1].set_title(f'预测值 vs 实际值 (R²={test_r2_enhanced:.3f})')

plt.tight_layout()
plt.savefig('docs/figures/08_regression_diagnostics.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: docs/figures/08_regression_diagnostics.png")

# ==================== 7. 保存模型 ====================
print("\n[7] 保存模型...")

os.makedirs('models', exist_ok=True)

model_info = {
    'model_base': model_base,
    'model_enhanced': model_enhanced,
    'scaler_base': scaler_base,
    'scaler_enhanced': scaler_enhanced,
    'encoders': encoders,
    'features_base': features_base,
    'features_enhanced': features_enhanced,
    'metrics_base': {
        'train_r2': train_r2_base,
        'test_r2': test_r2_base,
        'train_rmse': train_rmse_base,
        'test_rmse': test_rmse_base,
        'train_mae': train_mae_base,
        'test_mae': test_mae_base
    },
    'metrics_enhanced': {
        'train_r2': train_r2_enhanced,
        'test_r2': test_r2_enhanced,
        'train_rmse': train_rmse_enhanced,
        'test_rmse': test_rmse_enhanced,
        'train_mae': train_mae_enhanced,
        'test_mae': test_mae_enhanced
    },
    'vif_data': vif_data
}

with open('models/regression_model.pkl', 'wb') as f:
    pickle.dump(model_info, f)
print("✓ 已保存: models/regression_model.pkl")

# ==================== 8. 输出结果摘要 ====================
print("\n" + "=" * 60)
print("回归模型结果摘要")
print("=" * 60)
print(f"\n基准模型 (特征数: {len(features_base)}):")
print(f"  测试集 R²: {test_r2_base:.4f}")
print(f"  测试集 RMSE: {test_rmse_base:.4f}天")
print(f"  测试集 MAE: {test_mae_base:.4f}天")

print(f"\n改进模型 (特征数: {len(features_enhanced)}):")
print(f"  测试集 R²: {test_r2_enhanced:.4f}")
print(f"  测试集 RMSE: {test_rmse_enhanced:.4f}天")
print(f"  测试集 MAE: {test_mae_enhanced:.4f}天")

print(f"\n模型改进:")
print(f"  R²提升: {test_r2_enhanced - test_r2_base:.4f}")
print(f"  RMSE降低: {test_rmse_base - test_rmse_enhanced:.4f}天")

print("\n" + "=" * 60)
print("阶段2完成！回归模型已构建并评估。")
print("=" * 60)

