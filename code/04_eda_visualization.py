"""
探索性数据分析（EDA）与可视化
成员A - 第四步：单变量和多变量可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置路径
data_dir = '../data/processed'
cleaned_data_path = os.path.join(data_dir, 'diabetic_data_cleaned.csv')
output_dir = '../docs/figures'
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("步骤4: 探索性数据分析（EDA）与可视化")
print("=" * 60)

# 读取清洗后的数据
print("\n读取清洗后的数据...")
df = pd.read_csv(cleaned_data_path)
print(f"数据形状: {df.shape}")

# ==================== 单变量分析 ====================

print("\n1. 单变量分析...")

# 1.1 因变量分布
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 再入院情况
readmitted_counts = df['readmitted'].value_counts()
axes[0].bar(readmitted_counts.index, readmitted_counts.values, color=['#2ecc71', '#e74c3c', '#f39c12'])
axes[0].set_title('Distribution of Readmission Status', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Readmission Status')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)
for i, v in enumerate(readmitted_counts.values):
    axes[0].text(i, v, str(v), ha='center', va='bottom')

# 30天内再入院（二分类）
readmitted_30d_counts = df['readmitted_30d'].value_counts()
axes[1].bar(['No (<30 days)', 'Yes (<30 days)'], readmitted_30d_counts.values, 
            color=['#3498db', '#e74c3c'])
axes[1].set_title('30-Day Readmission (Binary)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Count')
for i, v in enumerate(readmitted_30d_counts.values):
    axes[1].text(i, v, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '01_readmission_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 保存: 01_readmission_distribution.png")
print("  结论: 展示了再入院情况的整体分布，30天内再入院的比例为关键指标")

# 1.2 住院天数分布
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['time_in_hospital'], bins=50, color='#3498db', edgecolor='black', alpha=0.7)
axes[0].set_title('Distribution of Length of Stay', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Days in Hospital')
axes[0].set_ylabel('Frequency')
axes[0].axvline(df['time_in_hospital'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["time_in_hospital"].mean():.2f}')
axes[0].axvline(df['time_in_hospital'].median(), color='green', linestyle='--', 
                label=f'Median: {df["time_in_hospital"].median():.2f}')
axes[0].legend()

axes[1].boxplot(df['time_in_hospital'], vert=True)
axes[1].set_title('Boxplot of Length of Stay', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Days in Hospital')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '02_length_of_stay_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 保存: 02_length_of_stay_distribution.png")
print(f"  结论: 平均住院天数 {df['time_in_hospital'].mean():.2f} 天，中位数 {df['time_in_hospital'].median():.2f} 天")

# 1.3 人口学变量分布
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 年龄分布
age_counts = df['age'].value_counts().sort_index()
axes[0, 0].bar(range(len(age_counts)), age_counts.values, color='#9b59b6')
axes[0, 0].set_title('Age Distribution', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Age Group')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_xticks(range(len(age_counts)))
axes[0, 0].set_xticklabels(age_counts.index, rotation=45, ha='right')

# 性别分布
gender_counts = df['gender'].value_counts()
axes[0, 1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', 
               colors=['#3498db', '#e74c3c'])
axes[0, 1].set_title('Gender Distribution', fontsize=12, fontweight='bold')

# 种族分布
race_counts = df['race'].value_counts().head(10)  # 只显示前10
axes[1, 0].barh(range(len(race_counts)), race_counts.values, color='#16a085')
axes[1, 0].set_title('Race Distribution (Top 10)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Count')
axes[1, 0].set_yticks(range(len(race_counts)))
axes[1, 0].set_yticklabels(race_counts.index)

# 诊断数量分布
axes[1, 1].hist(df['number_diagnoses'], bins=20, color='#e67e22', edgecolor='black', alpha=0.7)
axes[1, 1].set_title('Number of Diagnoses Distribution', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Number of Diagnoses')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '03_demographic_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 保存: 03_demographic_distribution.png")
print("  结论: 展示了患者的基本人口学特征分布情况")

# ==================== 双变量分析 ====================

print("\n2. 双变量分析...")

# 2.1 再入院率在不同分组下的分布
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 按年龄分组的再入院率
age_readmit = df.groupby('age')['readmitted_30d'].mean().sort_index()
axes[0, 0].bar(range(len(age_readmit)), age_readmit.values * 100, color='#e74c3c')
axes[0, 0].set_title('30-Day Readmission Rate by Age Group', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Age Group')
axes[0, 0].set_ylabel('Readmission Rate (%)')
axes[0, 0].set_xticks(range(len(age_readmit)))
axes[0, 0].set_xticklabels(age_readmit.index, rotation=45, ha='right')
axes[0, 0].axhline(df['readmitted_30d'].mean() * 100, color='black', linestyle='--', 
                   label=f'Overall: {df["readmitted_30d"].mean()*100:.2f}%')
axes[0, 0].legend()

# 按性别分组的再入院率
gender_readmit = df.groupby('gender')['readmitted_30d'].mean()
axes[0, 1].bar(gender_readmit.index, gender_readmit.values * 100, color=['#3498db', '#e74c3c'])
axes[0, 1].set_title('30-Day Readmission Rate by Gender', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Readmission Rate (%)')
axes[0, 1].axhline(df['readmitted_30d'].mean() * 100, color='black', linestyle='--')
for i, v in enumerate(gender_readmit.values):
    axes[0, 1].text(i, v*100, f'{v*100:.2f}%', ha='center', va='bottom')

# 按诊断数量分组的再入院率（将诊断数量分组）
df['num_diag_group'] = pd.cut(df['number_diagnoses'], bins=[0, 3, 6, 9, 20], 
                              labels=['1-3', '4-6', '7-9', '10+'])
diag_readmit = df.groupby('num_diag_group', observed=True)['readmitted_30d'].mean()
axes[1, 0].bar(range(len(diag_readmit)), diag_readmit.values * 100, color='#16a085')
axes[1, 0].set_title('30-Day Readmission Rate by Number of Diagnoses', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Number of Diagnoses Group')
axes[1, 0].set_ylabel('Readmission Rate (%)')
axes[1, 0].set_xticks(range(len(diag_readmit)))
axes[1, 0].set_xticklabels(diag_readmit.index)
axes[1, 0].axhline(df['readmitted_30d'].mean() * 100, color='black', linestyle='--')

# 按住院天数分组的再入院率（将住院天数分组）
df['los_group'] = pd.cut(df['time_in_hospital'], bins=[0, 3, 7, 14, 100], 
                         labels=['1-3 days', '4-7 days', '8-14 days', '15+ days'])
los_readmit = df.groupby('los_group', observed=True)['readmitted_30d'].mean()
axes[1, 1].bar(range(len(los_readmit)), los_readmit.values * 100, color='#f39c12')
axes[1, 1].set_title('30-Day Readmission Rate by Length of Stay', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Length of Stay Group')
axes[1, 1].set_ylabel('Readmission Rate (%)')
axes[1, 1].set_xticks(range(len(los_readmit)))
axes[1, 1].set_xticklabels(los_readmit.index, rotation=45, ha='right')
axes[1, 1].axhline(df['readmitted_30d'].mean() * 100, color='black', linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '04_readmission_by_groups.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 保存: 04_readmission_by_groups.png")
print("  结论: 不同年龄、性别、诊断数量和住院天数分组的再入院率存在差异")

# 2.2 住院天数在不同分组下的分布
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 按年龄分组的住院天数
age_los = df.groupby('age')['time_in_hospital'].mean().sort_index()
axes[0, 0].bar(range(len(age_los)), age_los.values, color='#3498db')
axes[0, 0].set_title('Average Length of Stay by Age Group', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Age Group')
axes[0, 0].set_ylabel('Average Days')
axes[0, 0].set_xticks(range(len(age_los)))
axes[0, 0].set_xticklabels(age_los.index, rotation=45, ha='right')
axes[0, 0].axhline(df['time_in_hospital'].mean(), color='red', linestyle='--', 
                   label=f'Overall: {df["time_in_hospital"].mean():.2f}')
axes[0, 0].legend()

# 按性别分组的住院天数
gender_los = df.groupby('gender')['time_in_hospital'].mean()
axes[0, 1].bar(gender_los.index, gender_los.values, color=['#3498db', '#e74c3c'])
axes[0, 1].set_title('Average Length of Stay by Gender', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Average Days')
axes[0, 1].axhline(df['time_in_hospital'].mean(), color='red', linestyle='--')

# 按再入院情况分组的住院天数
readmit_los = df.groupby('readmitted_30d')['time_in_hospital'].mean()
axes[1, 0].bar(['No', 'Yes'], readmit_los.values, color=['#2ecc71', '#e74c3c'])
axes[1, 0].set_title('Average Length of Stay by Readmission Status', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Average Days')
for i, v in enumerate(readmit_los.values):
    axes[1, 0].text(i, v, f'{v:.2f}', ha='center', va='bottom')

# 按诊断数量分组的住院天数
diag_los = df.groupby('num_diag_group', observed=True)['time_in_hospital'].mean()
axes[1, 1].bar(range(len(diag_los)), diag_los.values, color='#9b59b6')
axes[1, 1].set_title('Average Length of Stay by Number of Diagnoses', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Number of Diagnoses Group')
axes[1, 1].set_ylabel('Average Days')
axes[1, 1].set_xticks(range(len(diag_los)))
axes[1, 1].set_xticklabels(diag_los.index)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '05_length_of_stay_by_groups.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 保存: 05_length_of_stay_by_groups.png")
print("  结论: 不同特征分组的平均住院天数存在差异，再入院患者的住院天数可能更长")

# 2.3 相关性热力图（连续变量）
print("\n3. 连续变量相关性分析...")
continuous_vars = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                   'num_medications', 'number_outpatient', 'number_emergency', 
                   'number_inpatient', 'number_diagnoses']
continuous_vars = [v for v in continuous_vars if v in df.columns]

corr_matrix = df[continuous_vars].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap of Continuous Variables', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '06_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 保存: 06_correlation_heatmap.png")
print("  结论: 展示了连续变量之间的相关性，有助于识别多重共线性问题")

# 生成EDA总结报告
print("\n4. 生成EDA总结报告...")
report_path = os.path.join('../docs', '04_eda_summary.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("探索性数据分析（EDA）总结\n")
    f.write("=" * 60 + "\n\n")
    f.write("1. 因变量分布\n")
    f.write(f"   - 30天内再入院率: {df['readmitted_30d'].mean()*100:.2f}%\n")
    f.write(f"   - 平均住院天数: {df['time_in_hospital'].mean():.2f} 天\n")
    f.write(f"   - 住院天数中位数: {df['time_in_hospital'].median():.2f} 天\n\n")
    
    f.write("2. 关键发现\n")
    f.write("   - 不同年龄组的再入院率存在差异\n")
    f.write("   - 诊断数量与再入院率和住院天数相关\n")
    f.write("   - 住院天数与再入院情况可能存在关联\n\n")
    
    f.write("3. 生成的图表\n")
    f.write("   - 01_readmission_distribution.png: 再入院情况分布\n")
    f.write("   - 02_length_of_stay_distribution.png: 住院天数分布\n")
    f.write("   - 03_demographic_distribution.png: 人口学特征分布\n")
    f.write("   - 04_readmission_by_groups.png: 不同分组的再入院率\n")
    f.write("   - 05_length_of_stay_by_groups.png: 不同分组的住院天数\n")
    f.write("   - 06_correlation_heatmap.png: 连续变量相关性热力图\n")

print(f"EDA总结报告已保存到: {report_path}")

print("\n" + "=" * 60)
print("EDA与可视化完成！")
print("=" * 60)
print(f"\n所有图表已保存到: {output_dir}/")
print(f"共生成 6 张图表")

