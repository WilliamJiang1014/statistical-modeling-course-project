"""
变量说明表生成
成员A - 第二步：整理变量含义，生成变量说明表
"""

import pandas as pd
import os

# 读取数据
data_dir = '../data/raw/diabetes_130_us_hospitals_1999_2008'
# 使用抽样后的数据（符合10MB要求）
raw_data_path = os.path.join(data_dir, 'diabetic_data_sampled.csv')
# 如果抽样数据不存在，使用原始数据
if not os.path.exists(raw_data_path):
    raw_data_path = os.path.join(data_dir, 'diabetic_data.csv')
mapping_path = os.path.join(data_dir, 'IDS_mapping.csv')

df = pd.read_csv(raw_data_path)

# 读取映射文件
mapping_df = pd.read_csv(mapping_path, header=None)

# 解析映射文件
admission_type_map = {}
discharge_disposition_map = {}
admission_source_map = {}

current_map = None
for idx, row in mapping_df.iterrows():
    if pd.isna(row[0]) or row[0] == '':
        continue
    if 'admission_type_id' in str(row[0]):
        current_map = 'admission_type'
    elif 'discharge_disposition_id' in str(row[0]):
        current_map = 'discharge_disposition'
    elif 'admission_source_id' in str(row[0]):
        current_map = 'admission_source'
    else:
        if current_map == 'admission_type' and len(row) >= 2:
            try:
                key = int(row[0])
                admission_type_map[key] = row[1]
            except:
                pass
        elif current_map == 'discharge_disposition' and len(row) >= 2:
            try:
                key = int(row[0])
                discharge_disposition_map[key] = row[1]
            except:
                pass
        elif current_map == 'admission_source' and len(row) >= 2:
            try:
                key = int(row[0])
                admission_source_map[key] = row[1]
            except:
                pass

# 生成变量说明表
variable_doc = []

# 因变量
variable_doc.append({
    '变量名': 'readmitted',
    '变量类型': '分类变量（因变量-分类任务）',
    '取值说明': 'NO: 未再入院; <30: 30天内再入院; >30: 30天后再入院',
    '缺失值': '无',
    '备注': '需要转换为二分类：30天内再入院（1）vs 其他（0）'
})

variable_doc.append({
    '变量名': 'time_in_hospital',
    '变量类型': '连续变量（因变量-回归任务）',
    '取值说明': '本次住院天数（整数）',
    '缺失值': '无',
    '备注': '需要检查异常值（如过大值）'
})

# 人口学变量
variable_doc.append({
    '变量名': 'age',
    '变量类型': '分类变量',
    '取值说明': '年龄分组，如[0-10), [10-20), [20-30)等',
    '缺失值': '无',
    '备注': '已分组，可直接使用'
})

variable_doc.append({
    '变量名': 'gender',
    '变量类型': '分类变量',
    '取值说明': '性别：Male, Female',
    '缺失值': '可能有缺失',
    '备注': '需要检查缺失值处理'
})

variable_doc.append({
    '变量名': 'race',
    '变量类型': '分类变量',
    '取值说明': '种族：Caucasian, AfricanAmerican, Asian, Hispanic, Other等',
    '缺失值': '可能有"?"表示缺失',
    '备注': '需要处理缺失值'
})

# 住院相关变量
variable_doc.append({
    '变量名': 'admission_type_id',
    '变量类型': '分类变量',
    '取值说明': '入院类型ID（需参考映射表）',
    '缺失值': '无',
    '备注': f'映射关系已从IDS_mapping.csv读取，共{len(admission_type_map)}种类型'
})

variable_doc.append({
    '变量名': 'discharge_disposition_id',
    '变量类型': '分类变量',
    '取值说明': '出院去向ID（需参考映射表）',
    '缺失值': '无',
    '备注': f'映射关系已从IDS_mapping.csv读取，共{len(discharge_disposition_map)}种类型'
})

variable_doc.append({
    '变量名': 'admission_source_id',
    '变量类型': '分类变量',
    '取值说明': '入院来源ID（需参考映射表）',
    '缺失值': '无',
    '备注': f'映射关系已从IDS_mapping.csv读取，共{len(admission_source_map)}种类型'
})

# 既往就医
variable_doc.append({
    '变量名': 'number_outpatient',
    '变量类型': '连续变量',
    '取值说明': '既往门诊次数',
    '缺失值': '无',
    '备注': '可能需要检查异常值'
})

variable_doc.append({
    '变量名': 'number_emergency',
    '变量类型': '连续变量',
    '取值说明': '既往急诊次数',
    '缺失值': '无',
    '备注': '可能需要检查异常值'
})

variable_doc.append({
    '变量名': 'number_inpatient',
    '变量类型': '连续变量',
    '取值说明': '既往住院次数',
    '缺失值': '无',
    '备注': '可能需要检查异常值'
})

# 诊断相关
variable_doc.append({
    '变量名': 'diag_1',
    '变量类型': '分类变量',
    '取值说明': '主要诊断（ICD-9编码）',
    '缺失值': '可能有"?"表示缺失',
    '备注': '可能需要分组（如糖尿病相关、心血管相关等）'
})

variable_doc.append({
    '变量名': 'diag_2',
    '变量类型': '分类变量',
    '取值说明': '次要诊断1（ICD-9编码）',
    '缺失值': '可能有"?"表示缺失',
    '备注': '同diag_1'
})

variable_doc.append({
    '变量名': 'diag_3',
    '变量类型': '分类变量',
    '取值说明': '次要诊断2（ICD-9编码）',
    '缺失值': '可能有"?"表示缺失',
    '备注': '同diag_1'
})

variable_doc.append({
    '变量名': 'number_diagnoses',
    '变量类型': '连续变量',
    '取值说明': '诊断数量',
    '缺失值': '无',
    '备注': '可作为合并症数量的代理指标'
})

# 检查与治疗
variable_doc.append({
    '变量名': 'num_lab_procedures',
    '变量类型': '连续变量',
    '取值说明': '实验室检查次数',
    '缺失值': '无',
    '备注': ''
})

variable_doc.append({
    '变量名': 'num_procedures',
    '变量类型': '连续变量',
    '取值说明': '操作/手术次数',
    '缺失值': '无',
    '备注': ''
})

variable_doc.append({
    '变量名': 'num_medications',
    '变量类型': '连续变量',
    '取值说明': '用药数量',
    '缺失值': '无',
    '备注': ''
})

# 药物相关（多个变量，统一说明）
variable_doc.append({
    '变量名': 'insulin, metformin, glipizide等',
    '变量类型': '分类变量',
    '取值说明': '各种糖尿病药物的使用情况：No, Steady, Up, Down',
    '缺失值': '无',
    '备注': '可能需要合并为"是否使用胰岛素"等二分类变量'
})

variable_doc.append({
    '变量名': 'diabetesMed',
    '变量类型': '分类变量',
    '取值说明': '是否使用糖尿病药物：Yes, No',
    '缺失值': '无',
    '备注': ''
})

# 转换为DataFrame并保存
var_df = pd.DataFrame(variable_doc)

output_dir = '../docs'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, '02_variable_documentation.csv')
var_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print("=" * 60)
print("变量说明表已生成")
print("=" * 60)
print(f"\n已保存到: {output_path}")
print(f"\n共整理了 {len(variable_doc)} 个变量/变量组的说明")
print("\n变量说明表预览:")
print(var_df.to_string())

