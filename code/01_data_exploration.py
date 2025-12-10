"""
数据探索与理解
成员A - 第一步：理解数据结构和变量含义
"""

import pandas as pd
import numpy as np
import os

# 设置路径
data_dir = '../data/raw/diabetes_130_us_hospitals_1999_2008'
raw_data_path = os.path.join(data_dir, 'diabetic_data.csv')
mapping_path = os.path.join(data_dir, 'IDS_mapping.csv')

# 读取数据
print("=" * 60)
print("步骤1: 数据基本信息探索")
print("=" * 60)

df = pd.read_csv(raw_data_path)
print(f"\n数据形状: {df.shape}")
print(f"总记录数: {len(df):,}")
print(f"总变量数: {len(df.columns)}")

# 查看列名
print("\n所有变量列表:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

# 查看数据类型
print("\n数据类型分布:")
print(df.dtypes.value_counts())

# 查看缺失值
print("\n缺失值统计:")
missing_stats = df.isnull().sum()
missing_stats = missing_stats[missing_stats > 0].sort_values(ascending=False)
if len(missing_stats) > 0:
    print(f"有缺失值的变量数: {len(missing_stats)}")
    print("\n缺失值最多的前10个变量:")
    print(missing_stats.head(10))
else:
    print("没有发现缺失值（使用pandas的isnull方法）")

# 检查特殊值（如'?'）
print("\n检查特殊值（如'?'）:")
for col in df.columns:
    if df[col].dtype == 'object':
        unique_vals = df[col].unique()
        if '?' in unique_vals:
            count = (df[col] == '?').sum()
            print(f"{col}: {count} 个 '?' ({count/len(df)*100:.2f}%)")

# 查看关键变量的分布
print("\n关键变量 - readmitted 分布:")
print(df['readmitted'].value_counts())
print("\n关键变量 - time_in_hospital 统计:")
print(df['time_in_hospital'].describe())

# 保存基本信息到文件
output_dir = '../docs'
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, '01_data_basic_info.txt'), 'w', encoding='utf-8') as f:
    f.write("数据基本信息\n")
    f.write("=" * 60 + "\n")
    f.write(f"数据形状: {df.shape}\n")
    f.write(f"总记录数: {len(df):,}\n")
    f.write(f"总变量数: {len(df.columns)}\n\n")
    f.write("所有变量列表:\n")
    for i, col in enumerate(df.columns, 1):
        f.write(f"{i:2d}. {col}\n")
    f.write("\n数据类型分布:\n")
    f.write(str(df.dtypes.value_counts()) + "\n")

print("\n数据基本信息已保存到 docs/01_data_basic_info.txt")

