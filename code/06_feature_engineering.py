"""
阶段1：数据准备与特征工程
功能：数据加载、特征工程、数据分割
作者：成员B
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 数据加载 ====================
print("=" * 60)
print("阶段1：数据准备与特征工程")
print("=" * 60)

print("\n[1.1] 加载清洗后的数据...")
df = pd.read_csv('data/processed/diabetic_data_cleaned.csv')
print(f"数据形状: {df.shape}")
print(f"数据列数: {len(df.columns)}")

# 查看基本信息
print("\n[1.2] 数据基本信息:")
print(f"- 总记录数: {len(df):,}")
print(f"- 总变量数: {len(df.columns)}")
print(f"- 因变量 - 再入院率: {df['readmitted_30d'].mean():.2%}")
print(f"- 因变量 - 平均住院天数: {df['time_in_hospital'].mean():.2f}天")

# ==================== 2. 特征工程函数 ====================

def categorize_icd9(diag_code):
    """
    将ICD-9诊断编码分组
    参考：https://en.wikipedia.org/wiki/List_of_ICD-9_codes
    """
    if pd.isna(diag_code) or diag_code == 'Unknown':
        return 'Unknown'
    
    # 转换为字符串并提取数字部分
    diag_str = str(diag_code)
    if not diag_str.replace('.', '').isdigit():
        return 'Unknown'
    
    try:
        diag_num = float(diag_str)
    except:
        return 'Unknown'
    
    # ICD-9编码分组
    if 250 <= diag_num < 251:
        return 'Diabetes'
    elif 390 <= diag_num < 460 or 785 <= diag_num < 786:
        return 'Circulatory'
    elif 460 <= diag_num < 520:
        return 'Respiratory'
    elif 520 <= diag_num < 580:
        return 'Digestive'
    elif 580 <= diag_num < 630:
        return 'Genitourinary'
    elif 800 <= diag_num < 1000:
        return 'Injury'
    elif 140 <= diag_num < 240:
        return 'Neoplasms'
    else:
        return 'Other'

def group_discharge_disposition(discharge_id):
    """
    将出院去向分组（30个类别太多，需要分组）
    根据常见分组方式
    """
    # 常见分组：
    # 1: 回家
    # 2-6: 转院/转机构
    # 7-10: 其他医疗机构
    # 11-19: 其他
    # 20-29: 死亡/其他
    # 30: 其他
    
    if discharge_id in [1]:
        return 'Home'
    elif discharge_id in [2, 3, 4, 5, 6]:
        return 'Transfer'
    elif discharge_id in [7, 8, 9, 10]:
        return 'Other_Facility'
    elif discharge_id in [11, 12, 13, 14, 15, 16, 17, 18, 19]:
        return 'Other'
    elif discharge_id in [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]:
        return 'Death_Other'
    else:
        return 'Unknown'

def group_admission_source(source_id):
    """
    将入院来源分组（25个类别，需要分组）
    """
    # 常见分组：
    # 1: 医生转诊
    # 2: 诊所转诊
    # 3: HMO转诊
    # 4-8: 转院
    # 9-17: 急诊/其他
    # 18-25: 其他
    
    if source_id in [1, 2, 3]:
        return 'Referral'
    elif source_id in [4, 5, 6, 7, 8]:
        return 'Transfer'
    elif source_id in [9, 10, 11, 12, 13, 14, 15, 16, 17]:
        return 'Emergency_Other'
    else:
        return 'Other'

# ==================== 3. 特征工程 ====================
print("\n[2] 开始特征工程...")

# 创建数据副本
df_fe = df.copy()

# 3.1 诊断编码分组
print("\n[2.1] 处理诊断编码（ICD-9分组）...")
df_fe['diag_1_group'] = df_fe['diag_1'].apply(categorize_icd9)
df_fe['diag_2_group'] = df_fe['diag_2'].apply(categorize_icd9)
df_fe['diag_3_group'] = df_fe['diag_3'].apply(categorize_icd9)

print(f"主要诊断分组分布:")
print(df_fe['diag_1_group'].value_counts().head(10))

# 3.2 出院去向分组
print("\n[2.2] 处理出院去向分组...")
df_fe['discharge_disposition_group'] = df_fe['discharge_disposition_id'].apply(group_discharge_disposition)
print(f"出院去向分组分布:")
print(df_fe['discharge_disposition_group'].value_counts())

# 3.3 入院来源分组
print("\n[2.3] 处理入院来源分组...")
df_fe['admission_source_group'] = df_fe['admission_source_id'].apply(group_admission_source)
print(f"入院来源分组分布:")
print(df_fe['admission_source_group'].value_counts())

# 3.4 创建交互项（可选，在建模时再决定是否使用）
print("\n[2.4] 创建交互项特征...")
# 年龄与诊断数量的交互（年龄分组需要先转换为数值）
# 这里先创建，后续在建模时再决定是否使用

# 3.5 处理高缺失率变量的指示变量（可选）
print("\n[2.5] 创建高缺失率变量的指示变量...")
df_fe['has_weight'] = (df_fe['weight'].notna()) & (df_fe['weight'] != 'Unknown')
df_fe['has_glucose'] = (df_fe['max_glu_serum'].notna()) & (df_fe['max_glu_serum'] != 'Unknown')
df_fe['has_A1C'] = (df_fe['A1Cresult'].notna()) & (df_fe['A1Cresult'] != 'Unknown')

print(f"有体重数据: {df_fe['has_weight'].sum():,} ({df_fe['has_weight'].mean():.2%})")
print(f"有血糖数据: {df_fe['has_glucose'].sum():,} ({df_fe['has_glucose'].mean():.2%})")
print(f"有A1C数据: {df_fe['has_A1C'].sum():,} ({df_fe['has_A1C'].mean():.2%})")

# ==================== 4. 准备建模变量 ====================
print("\n[3] 准备建模变量...")

# 4.1 回归模型变量（预测住院天数）
regression_features_base = [
    # 人口学
    'age', 'gender', 'race',
    # 既往就医
    'number_outpatient', 'number_emergency', 'number_inpatient',
    # 诊断
    'number_diagnoses',
    # 治疗
    'num_lab_procedures', 'num_procedures', 'num_medications',
    # 入院
    'admission_type_id', 'admission_source_id'
]

regression_features_enhanced = regression_features_base + [
    # 增强特征
    'diag_1_group', 'diag_2_group', 'diag_3_group',
    'discharge_disposition_group', 'admission_source_group',
    'insulin_use', 'diabetesMed', 'medication_changed'
]

# 4.2 分类模型变量（预测30天再入院）
classification_features_base = [
    # 人口学
    'age', 'gender', 'race',
    # 既往就医
    'number_outpatient', 'number_emergency', 'number_inpatient',
    # 诊断
    'number_diagnoses',
    # 治疗
    'num_medications', 'insulin_use', 'diabetesMed',
    # 住院
    'admission_type_id'
    # 注意：time_in_hospital在预测再入院时需谨慎使用
]

classification_features_enhanced = classification_features_base + [
    # 增强特征
    'diag_1_group', 'diag_2_group', 'diag_3_group',
    'discharge_disposition_group', 'admission_source_group',
    'medication_changed',
    'num_lab_procedures', 'num_procedures'
]

print(f"\n回归模型 - 基准特征数: {len(regression_features_base)}")
print(f"回归模型 - 增强特征数: {len(regression_features_enhanced)}")
print(f"分类模型 - 基准特征数: {len(classification_features_base)}")
print(f"分类模型 - 增强特征数: {len(classification_features_enhanced)}")

# ==================== 5. 数据分割 ====================
print("\n[4] 数据分割（训练集/测试集）...")

# 5.1 回归任务数据分割
X_reg = df_fe[regression_features_enhanced].copy()
y_reg = df_fe['time_in_hospital'].copy()

# 5.2 分类任务数据分割
X_clf = df_fe[classification_features_enhanced].copy()
y_clf = df_fe['readmitted_30d'].copy()

# 分割数据（80/20）
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42, stratify=None
)

X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

print(f"\n回归任务:")
print(f"  训练集: {len(X_reg_train):,} 样本")
print(f"  测试集: {len(X_reg_test):,} 样本")
print(f"  训练集平均住院天数: {y_reg_train.mean():.2f}天")
print(f"  测试集平均住院天数: {y_reg_test.mean():.2f}天")

print(f"\n分类任务:")
print(f"  训练集: {len(X_clf_train):,} 样本")
print(f"  测试集: {len(X_clf_test):,} 样本")
print(f"  训练集再入院率: {y_clf_train.mean():.2%}")
print(f"  测试集再入院率: {y_clf_test.mean():.2%}")

# ==================== 6. 保存处理后的数据 ====================
print("\n[5] 保存处理后的数据...")

# 保存特征工程后的完整数据
df_fe.to_csv('data/processed/diabetic_data_featured.csv', index=False)
print("✓ 已保存特征工程后的数据: data/processed/diabetic_data_featured.csv")

# 保存训练集和测试集（用于后续建模）
import pickle

data_splits = {
    'regression': {
        'X_train': X_reg_train,
        'X_test': X_reg_test,
        'y_train': y_reg_train,
        'y_test': y_reg_test,
        'features_base': regression_features_base,
        'features_enhanced': regression_features_enhanced
    },
    'classification': {
        'X_train': X_clf_train,
        'X_test': X_clf_test,
        'y_train': y_clf_train,
        'y_test': y_clf_test,
        'features_base': classification_features_base,
        'features_enhanced': classification_features_enhanced
    }
}

with open('data/processed/data_splits.pkl', 'wb') as f:
    pickle.dump(data_splits, f)
print("✓ 已保存数据分割: data/processed/data_splits.pkl")

print("\n" + "=" * 60)
print("阶段1完成！数据准备与特征工程已完成。")
print("=" * 60)


