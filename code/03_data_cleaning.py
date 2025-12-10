"""
数据清洗
成员A - 第三步：数据清洗流程
包括：删除错误记录、缺失值处理、异常值处理、变量编码
"""

import pandas as pd
import numpy as np
import os

# 设置路径
data_dir = '../data/raw/diabetes_130_us_hospitals_1999_2008'
# 使用抽样后的数据（符合10MB要求）
raw_data_path = os.path.join(data_dir, 'diabetic_data_sampled.csv')
# 如果抽样数据不存在，使用原始数据
if not os.path.exists(raw_data_path):
    raw_data_path = os.path.join(data_dir, 'diabetic_data.csv')
output_dir = '../data/processed'
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("步骤3: 数据清洗")
print("=" * 60)

# 读取原始数据
print("\n1. 读取原始数据...")
df = pd.read_csv(raw_data_path)
print(f"原始数据形状: {df.shape}")

# 2. 删除重复记录
print("\n2. 检查并删除重复记录...")
initial_count = len(df)
df = df.drop_duplicates()
duplicate_count = initial_count - len(df)
print(f"删除重复记录: {duplicate_count} 条")
print(f"剩余记录数: {len(df):,}")

# 3. 处理特殊值（将'?'替换为NaN）
print("\n3. 处理特殊值（将'?'替换为NaN）...")
for col in df.columns:
    if df[col].dtype == 'object':
        if (df[col] == '?').any():
            count = (df[col] == '?').sum()
            df[col] = df[col].replace('?', np.nan)
            print(f"  {col}: 替换了 {count} 个 '?' 为 NaN")

# 4. 处理因变量 - readmitted
print("\n4. 处理因变量 - readmitted...")
print("原始分布:")
print(df['readmitted'].value_counts())

# 创建二分类变量：30天内再入院（1）vs 其他（0）
df['readmitted_30d'] = (df['readmitted'] == '<30').astype(int)
print("\n转换后的二分类变量分布:")
print(df['readmitted_30d'].value_counts())
print(f"30天内再入院率: {df['readmitted_30d'].mean()*100:.2f}%")

# 5. 处理因变量 - time_in_hospital（住院天数）
print("\n5. 检查住院天数异常值...")
print("住院天数统计:")
print(df['time_in_hospital'].describe())

# 检查异常值（如超过某个阈值，可根据实际情况调整）
# 通常住院天数超过100天可能是异常值
outlier_threshold = 100
outlier_count = (df['time_in_hospital'] > outlier_threshold).sum()
print(f"\n住院天数 > {outlier_threshold} 天的记录数: {outlier_count}")
if outlier_count > 0:
    print("建议：可以考虑删除或截断这些异常值")
    # 这里先标记，不直接删除，让用户决定
    df['time_in_hospital_outlier'] = (df['time_in_hospital'] > outlier_threshold).astype(int)

# 6. 处理缺失值
print("\n6. 缺失值统计...")
missing_stats = df.isnull().sum()
missing_stats = missing_stats[missing_stats > 0].sort_values(ascending=False)
print(f"有缺失值的变量数: {len(missing_stats)}")
print("\n缺失值最多的前15个变量:")
for var, count in missing_stats.head(15).items():
    pct = count / len(df) * 100
    print(f"  {var}: {count:,} ({pct:.2f}%)")

# 缺失值处理策略（示例，可根据实际情况调整）
print("\n7. 应用缺失值处理策略...")

# 对于某些变量，如果缺失率很高，可以考虑删除该变量或单独编码为"未知"
# 例如：weight变量缺失率可能很高
if 'weight' in df.columns:
    weight_missing_pct = df['weight'].isnull().sum() / len(df) * 100
    print(f"weight变量缺失率: {weight_missing_pct:.2f}%")
    if weight_missing_pct > 50:
        print("  建议：weight缺失率过高，考虑删除该变量或单独编码为'未知'类别")

# 对于分类变量，将缺失值编码为"Unknown"
categorical_vars = ['race', 'gender', 'payer_code', 'medical_specialty']
for var in categorical_vars:
    if var in df.columns:
        missing_count = df[var].isnull().sum()
        if missing_count > 0:
            df[var] = df[var].fillna('Unknown')
            print(f"  {var}: 将 {missing_count} 个缺失值填充为 'Unknown'")

# 对于诊断变量，缺失值可以保留为NaN或编码为"Unknown"
diagnosis_vars = ['diag_1', 'diag_2', 'diag_3']
for var in diagnosis_vars:
    if var in df.columns:
        missing_count = df[var].isnull().sum()
        if missing_count > 0:
            df[var] = df[var].fillna('Unknown')
            print(f"  {var}: 将 {missing_count} 个缺失值填充为 'Unknown'")

# 8. 变量编码和分组
print("\n8. 变量编码和分组...")

# age已经是分组形式，但可以进一步简化
if 'age' in df.columns:
    print(f"age变量当前取值: {df['age'].unique()}")
    # age已经是分组，可以直接使用

# 创建一些衍生变量
print("\n9. 创建衍生变量...")

# 合并症数量（已有number_diagnoses，但可以基于诊断编码进一步分析）
# 这里先使用number_diagnoses作为代理

# 是否使用胰岛素（简化）
if 'insulin' in df.columns:
    df['insulin_use'] = (df['insulin'] != 'No').astype(int)
    print(f"insulin_use: {df['insulin_use'].sum():,} 人使用胰岛素")

# 药物变化情况
if 'change' in df.columns:
    df['medication_changed'] = (df['change'] == 'Ch').astype(int)
    print(f"medication_changed: {df['medication_changed'].sum():,} 人药物有变化")

# 10. 保存清洗后的数据
print("\n10. 保存清洗后的数据...")
cleaned_data_path = os.path.join(output_dir, 'diabetic_data_cleaned.csv')
df.to_csv(cleaned_data_path, index=False, encoding='utf-8')
print(f"已保存到: {cleaned_data_path}")
print(f"清洗后数据形状: {df.shape}")

# 11. 生成清洗报告
print("\n11. 生成数据清洗报告...")
report_path = os.path.join('../docs', '03_data_cleaning_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("数据清洗报告\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"原始数据记录数: {initial_count:,}\n")
    f.write(f"删除重复记录: {duplicate_count} 条\n")
    f.write(f"清洗后记录数: {len(df):,}\n\n")
    f.write("缺失值处理:\n")
    for var, count in missing_stats.head(20).items():
        pct = count / len(df) * 100
        f.write(f"  {var}: {count:,} ({pct:.2f}%)\n")
    f.write(f"\n因变量处理:\n")
    f.write(f"  readmitted_30d: 30天内再入院率 {df['readmitted_30d'].mean()*100:.2f}%\n")
    f.write(f"  time_in_hospital: 平均住院天数 {df['time_in_hospital'].mean():.2f} 天\n")

print(f"清洗报告已保存到: {report_path}")

print("\n" + "=" * 60)
print("数据清洗完成！")
print("=" * 60)

