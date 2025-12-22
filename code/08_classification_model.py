"""
阶段3：分类模型（30天再入院预测）
功能：Logistic回归模型构建、评估与解读
作者：成员B
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, 
                             confusion_matrix, classification_report)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import os

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 加载数据 ====================
print("=" * 60)
print("阶段3：分类模型开发（30天再入院预测）")
print("=" * 60)

print("\n[1] 加载数据...")
with open('data/processed/data_splits.pkl', 'rb') as f:
    data_splits = pickle.load(f)

clf_data = data_splits['classification']
X_train = clf_data['X_train'].copy()
X_test = clf_data['X_test'].copy()
y_train = clf_data['y_train'].copy()
y_test = clf_data['y_test'].copy()
features_base = clf_data['features_base']
features_enhanced = clf_data['features_enhanced']

print(f"训练集: {len(X_train):,} 样本")
print(f"测试集: {len(X_test):,} 样本")
print(f"训练集再入院率: {y_train.mean():.2%}")
print(f"测试集再入院率: {y_test.mean():.2%}")

# ==================== 2. 特征编码 ====================
def encode_features(X, encoders=None, fit=True):
    """对分类变量进行编码"""
    X_encoded = X.copy()
    if encoders is None:
        encoders = {}
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        if fit:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
        else:
            if col in encoders:
                X_encoded[col] = X[col].astype(str).apply(
                    lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else 0
                )
    
    return X_encoded, encoders

print("\n[2] 特征编码...")
X_train_encoded, encoders = encode_features(X_train, fit=True)
X_test_encoded, _ = encode_features(X_test, encoders=encoders, fit=False)

# ==================== 3. 处理类别不平衡 ====================
print("\n[3] 计算类别权重...")
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))
print(f"类别权重: {class_weight_dict}")

# ==================== 4. 基准模型 ====================
print("\n[4] 构建基准模型...")

X_train_base = X_train_encoded[features_base].copy()
X_test_base = X_test_encoded[features_base].copy()

scaler_base = StandardScaler()
X_train_base_scaled = scaler_base.fit_transform(X_train_base)
X_test_base_scaled = scaler_base.transform(X_test_base)

model_base = LogisticRegression(class_weight=class_weight_dict, max_iter=1000, random_state=42)
model_base.fit(X_train_base_scaled, y_train)

y_train_pred_base = model_base.predict(X_train_base_scaled)
y_test_pred_base = model_base.predict(X_test_base_scaled)
y_train_proba_base = model_base.predict_proba(X_train_base_scaled)[:, 1]
y_test_proba_base = model_base.predict_proba(X_test_base_scaled)[:, 1]

# 评估
train_acc_base = accuracy_score(y_train, y_train_pred_base)
test_acc_base = accuracy_score(y_test, y_test_pred_base)
train_prec_base = precision_score(y_train, y_train_pred_base)
test_prec_base = precision_score(y_test, y_test_pred_base)
train_rec_base = recall_score(y_train, y_train_pred_base)
test_rec_base = recall_score(y_test, y_test_pred_base)
train_f1_base = f1_score(y_train, y_train_pred_base)
test_f1_base = f1_score(y_test, y_test_pred_base)
train_auc_base = roc_auc_score(y_train, y_train_proba_base)
test_auc_base = roc_auc_score(y_test, y_test_proba_base)

print("\n基准模型评估结果:")
print(f"训练集 - 准确率: {train_acc_base:.4f}, 精确率: {train_prec_base:.4f}, 召回率: {train_rec_base:.4f}, F1: {train_f1_base:.4f}, AUC: {train_auc_base:.4f}")
print(f"测试集 - 准确率: {test_acc_base:.4f}, 精确率: {test_prec_base:.4f}, 召回率: {test_rec_base:.4f}, F1: {test_f1_base:.4f}, AUC: {test_auc_base:.4f}")

# ==================== 5. 改进模型 ====================
print("\n[5] 构建改进模型...")

X_train_enhanced = X_train_encoded[features_enhanced].copy()
X_test_enhanced = X_test_encoded[features_enhanced].copy()

scaler_enhanced = StandardScaler()
X_train_enhanced_scaled = scaler_enhanced.fit_transform(X_train_enhanced)
X_test_enhanced_scaled = scaler_enhanced.transform(X_test_enhanced)

model_enhanced = LogisticRegression(class_weight=class_weight_dict, max_iter=1000, random_state=42)
model_enhanced.fit(X_train_enhanced_scaled, y_train)

y_train_pred_enhanced = model_enhanced.predict(X_train_enhanced_scaled)
y_test_pred_enhanced = model_enhanced.predict(X_test_enhanced_scaled)
y_train_proba_enhanced = model_enhanced.predict_proba(X_train_enhanced_scaled)[:, 1]
y_test_proba_enhanced = model_enhanced.predict_proba(X_test_enhanced_scaled)[:, 1]

train_acc_enhanced = accuracy_score(y_train, y_train_pred_enhanced)
test_acc_enhanced = accuracy_score(y_test, y_test_pred_enhanced)
train_prec_enhanced = precision_score(y_train, y_train_pred_enhanced)
test_prec_enhanced = precision_score(y_test, y_test_pred_enhanced)
train_rec_enhanced = recall_score(y_train, y_train_pred_enhanced)
test_rec_enhanced = recall_score(y_test, y_test_pred_enhanced)
train_f1_enhanced = f1_score(y_train, y_train_pred_enhanced)
test_f1_enhanced = f1_score(y_test, y_test_pred_enhanced)
train_auc_enhanced = roc_auc_score(y_train, y_train_proba_enhanced)
test_auc_enhanced = roc_auc_score(y_test, y_test_proba_enhanced)

print("\n改进模型评估结果:")
print(f"训练集 - 准确率: {train_acc_enhanced:.4f}, 精确率: {train_prec_enhanced:.4f}, 召回率: {train_rec_enhanced:.4f}, F1: {train_f1_enhanced:.4f}, AUC: {train_auc_enhanced:.4f}")
print(f"测试集 - 准确率: {test_acc_enhanced:.4f}, 精确率: {test_prec_enhanced:.4f}, 召回率: {test_rec_enhanced:.4f}, F1: {test_f1_enhanced:.4f}, AUC: {test_auc_enhanced:.4f}")

# ==================== 6. OR值计算 ====================
print("\n[6] 计算OR值（优势比）...")

# OR = exp(系数)
or_base = pd.DataFrame({
    'Feature': features_base,
    'Coefficient': model_base.coef_[0],
    'OR': np.exp(model_base.coef_[0])
})
or_base = or_base.sort_values('OR', key=abs, ascending=False)

or_enhanced = pd.DataFrame({
    'Feature': features_enhanced,
    'Coefficient': model_enhanced.coef_[0],
    'OR': np.exp(model_enhanced.coef_[0])
})
or_enhanced = or_enhanced.sort_values('OR', key=abs, ascending=False)

print("\n基准模型 - 主要变量OR值:")
print(or_base.head(10))

print("\n改进模型 - 主要变量OR值:")
print(or_enhanced.head(10))

# ==================== 7. 结果可视化 ====================
print("\n[7] 生成可视化图表...")

os.makedirs('docs/figures', exist_ok=True)

# 7.1 OR值条形图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

or_base_top = or_base.head(15)
axes[0].barh(range(len(or_base_top)), or_base_top['OR'])
axes[0].set_yticks(range(len(or_base_top)))
axes[0].set_yticklabels(or_base_top['Feature'])
axes[0].set_xlabel('OR值')
axes[0].set_title('基准模型 - 主要变量OR值')
axes[0].axvline(x=1, color='r', linestyle='--', linewidth=0.8)

or_enhanced_top = or_enhanced.head(15)
axes[1].barh(range(len(or_enhanced_top)), or_enhanced_top['OR'])
axes[1].set_yticks(range(len(or_enhanced_top)))
axes[1].set_yticklabels(or_enhanced_top['Feature'])
axes[1].set_xlabel('OR值')
axes[1].set_title('改进模型 - 主要变量OR值')
axes[1].axvline(x=1, color='r', linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.savefig('docs/figures/09_classification_or_values.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: docs/figures/09_classification_or_values.png")

# 7.2 混淆矩阵
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_base = confusion_matrix(y_test, y_test_pred_base)
sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_xlabel('预测值')
axes[0].set_ylabel('实际值')
axes[0].set_title('基准模型 - 混淆矩阵')

cm_enhanced = confusion_matrix(y_test, y_test_pred_enhanced)
sns.heatmap(cm_enhanced, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_xlabel('预测值')
axes[1].set_ylabel('实际值')
axes[1].set_title('改进模型 - 混淆矩阵')

plt.tight_layout()
plt.savefig('docs/figures/10_classification_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: docs/figures/10_classification_confusion_matrix.png")

# 7.3 ROC曲线
fig, ax = plt.subplots(figsize=(8, 6))

fpr_base, tpr_base, _ = roc_curve(y_test, y_test_proba_base)
fpr_enhanced, tpr_enhanced, _ = roc_curve(y_test, y_test_proba_enhanced)

ax.plot(fpr_base, tpr_base, label=f'基准模型 (AUC={test_auc_base:.3f})', linewidth=2)
ax.plot(fpr_enhanced, tpr_enhanced, label=f'改进模型 (AUC={test_auc_enhanced:.3f})', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', label='随机分类器', linewidth=1)
ax.set_xlabel('假阳性率 (FPR)')
ax.set_ylabel('真阳性率 (TPR)')
ax.set_title('ROC曲线')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('docs/figures/11_classification_roc_curve.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: docs/figures/11_classification_roc_curve.png")

# 7.4 预测概率分布
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(y_test_proba_base[y_test == 0], bins=50, alpha=0.7, label='未再入院', color='blue')
axes[0].hist(y_test_proba_base[y_test == 1], bins=50, alpha=0.7, label='再入院', color='red')
axes[0].set_xlabel('预测概率')
axes[0].set_ylabel('频数')
axes[0].set_title('基准模型 - 预测概率分布')
axes[0].legend()

axes[1].hist(y_test_proba_enhanced[y_test == 0], bins=50, alpha=0.7, label='未再入院', color='blue')
axes[1].hist(y_test_proba_enhanced[y_test == 1], bins=50, alpha=0.7, label='再入院', color='red')
axes[1].set_xlabel('预测概率')
axes[1].set_ylabel('频数')
axes[1].set_title('改进模型 - 预测概率分布')
axes[1].legend()

plt.tight_layout()
plt.savefig('docs/figures/12_classification_probability_distribution.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: docs/figures/12_classification_probability_distribution.png")

# ==================== 8. 保存模型 ====================
print("\n[8] 保存模型...")

os.makedirs('models', exist_ok=True)

model_info = {
    'model_base': model_base,
    'model_enhanced': model_enhanced,
    'scaler_base': scaler_base,
    'scaler_enhanced': scaler_enhanced,
    'encoders': encoders,
    'features_base': features_base,
    'features_enhanced': features_enhanced,
    'class_weight_dict': class_weight_dict,
    'or_base': or_base,
    'or_enhanced': or_enhanced,
    'metrics_base': {
        'train_acc': train_acc_base,
        'test_acc': test_acc_base,
        'train_prec': train_prec_base,
        'test_prec': test_prec_base,
        'train_rec': train_rec_base,
        'test_rec': test_rec_base,
        'train_f1': train_f1_base,
        'test_f1': test_f1_base,
        'train_auc': train_auc_base,
        'test_auc': test_auc_base
    },
    'metrics_enhanced': {
        'train_acc': train_acc_enhanced,
        'test_acc': test_acc_enhanced,
        'train_prec': train_prec_enhanced,
        'test_prec': test_prec_enhanced,
        'train_rec': train_rec_enhanced,
        'test_rec': test_rec_enhanced,
        'train_f1': train_f1_enhanced,
        'test_f1': test_f1_enhanced,
        'train_auc': train_auc_enhanced,
        'test_auc': test_auc_enhanced
    }
}

with open('models/classification_model.pkl', 'wb') as f:
    pickle.dump(model_info, f)
print("✓ 已保存: models/classification_model.pkl")

# ==================== 9. 输出结果摘要 ====================
print("\n" + "=" * 60)
print("分类模型结果摘要")
print("=" * 60)
print(f"\n基准模型 (特征数: {len(features_base)}):")
print(f"  测试集 - 准确率: {test_acc_base:.4f}, 精确率: {test_prec_base:.4f}")
print(f"  测试集 - 召回率: {test_rec_base:.4f}, F1: {test_f1_base:.4f}, AUC: {test_auc_base:.4f}")

print(f"\n改进模型 (特征数: {len(features_enhanced)}):")
print(f"  测试集 - 准确率: {test_acc_enhanced:.4f}, 精确率: {test_prec_enhanced:.4f}")
print(f"  测试集 - 召回率: {test_rec_enhanced:.4f}, F1: {test_f1_enhanced:.4f}, AUC: {test_auc_enhanced:.4f}")

print(f"\n模型改进:")
print(f"  AUC提升: {test_auc_enhanced - test_auc_base:.4f}")
print(f"  F1提升: {test_f1_enhanced - test_f1_base:.4f}")

print("\n" + "=" * 60)
print("阶段3完成！分类模型已构建并评估。")
print("=" * 60)

