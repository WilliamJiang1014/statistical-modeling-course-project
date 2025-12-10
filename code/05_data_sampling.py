"""
数据抽样脚本 - 将数据控制在10MB以内
根据项目要求，数据规模需要控制在10MB以内
"""

import pandas as pd
import numpy as np
import os

print("=" * 60)
print("数据抽样 - 控制数据规模在10MB以内")
print("=" * 60)

# 设置路径
raw_data_path = '../data/raw/diabetes_130_us_hospitals_1999_2008/diabetic_data.csv'
output_dir = '../data/raw/diabetes_130_us_hospitals_1999_2008'
sampled_data_path = os.path.join(output_dir, 'diabetic_data_sampled.csv')

# 读取原始数据
print("\n1. 读取原始数据...")
df = pd.read_csv(raw_data_path)
print(f"原始数据: {len(df):,} 行, {len(df.columns)} 列")

# 估算文件大小（粗略）
estimated_size_mb = len(df) * len(df.columns) * 10 / 1024 / 1024
print(f"估算文件大小: {estimated_size_mb:.1f} MB")

# 计算需要保留的行数（目标：约8MB，留一些余量）
target_size_mb = 8  # 目标8MB，留2MB余量
sample_ratio = target_size_mb / estimated_size_mb
n_samples = int(len(df) * sample_ratio)

print(f"\n2. 计算抽样比例...")
print(f"目标大小: {target_size_mb} MB")
print(f"抽样比例: {sample_ratio:.3f}")
print(f"抽样后行数: {n_samples:,}")

# 分层抽样：确保再入院情况的分布保持
print("\n3. 进行分层抽样（保持再入院分布）...")
if 'readmitted' in df.columns:
    # 按再入院情况分层
    df_sampled = df.groupby('readmitted', group_keys=False).apply(
        lambda x: x.sample(frac=sample_ratio, random_state=42)
    ).reset_index(drop=True)
    
    print("分层抽样完成")
    print("\n原始数据再入院分布:")
    print(df['readmitted'].value_counts())
    print("\n抽样后数据再入院分布:")
    print(df_sampled['readmitted'].value_counts())
else:
    # 如果没有readmitted列，使用简单随机抽样
    df_sampled = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    print("简单随机抽样完成")

# 保存抽样后的数据
print(f"\n4. 保存抽样后的数据...")
df_sampled.to_csv(sampled_data_path, index=False)
print(f"已保存到: {sampled_data_path}")

# 检查文件大小
import os
file_size_mb = os.path.getsize(sampled_data_path) / 1024 / 1024
print(f"实际文件大小: {file_size_mb:.2f} MB")

if file_size_mb <= 10:
    print(f"✅ 文件大小符合要求（≤10MB）")
else:
    print(f"⚠️ 文件大小仍超过10MB，需要进一步减少")

# 显示抽样后的数据信息
print(f"\n5. 抽样后数据信息:")
print(f"行数: {len(df_sampled):,}")
print(f"列数: {len(df_sampled.columns)}")
print(f"抽样比例: {len(df_sampled)/len(df)*100:.1f}%")

print("\n" + "=" * 60)
print("数据抽样完成！")
print("=" * 60)
print(f"\n建议:")
print(f"1. 使用抽样后的数据: {sampled_data_path}")
print(f"2. 重新运行数据清洗脚本，使用抽样后的数据作为输入")
print(f"3. 更新README说明使用了数据抽样")

