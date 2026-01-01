# ============================================================================
# 数据准备与特征工程
# 功能：数据加载、特征工程、数据分割
# ============================================================================

# 加载必要的包
library(dplyr)
library(readr)
library(caret)

# 设置随机种子
set.seed(42)

# ==================== 1. 数据加载 ====================
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("阶段1：数据准备与特征工程\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

cat("\n[1.1] 加载清洗后的数据...\n")
df <- read_csv('data/processed/diabetic_data_cleaned.csv', 
               locale = locale(encoding = "UTF-8"))
cat("数据形状:", nrow(df), "行,", ncol(df), "列\n")
cat("数据列数:", ncol(df), "\n")

# 查看基本信息
cat("\n[1.2] 数据基本信息:\n")
cat("- 总记录数:", format(nrow(df), big.mark = ","), "\n")
cat("- 总变量数:", ncol(df), "\n")
cat("- 因变量 - 再入院率:", sprintf("%.2f%%", mean(df$readmitted_30d) * 100), "\n")
cat("- 因变量 - 平均住院天数:", sprintf("%.2f天", mean(df$time_in_hospital)), "\n")

# ==================== 2. 特征工程函数 ====================

# 将ICD-9诊断编码分组
categorize_icd9 <- function(diag_code) {
  # 处理缺失值或Unknown
  if (is.na(diag_code) || diag_code == 'Unknown') {
    return('Unknown')
  }
  
  # 转换为字符串
  diag_str <- as.character(diag_code)
  
  # 尝试转换为数字
  diag_num <- suppressWarnings(as.numeric(diag_str))
  
  if (is.na(diag_num)) {
    return('Unknown')
  }
  
  # ICD-9编码分组
  if (diag_num >= 250 && diag_num < 251) {
    return('Diabetes')
  } else if ((diag_num >= 390 && diag_num < 460) || (diag_num >= 785 && diag_num < 786)) {
    return('Circulatory')
  } else if (diag_num >= 460 && diag_num < 520) {
    return('Respiratory')
  } else if (diag_num >= 520 && diag_num < 580) {
    return('Digestive')
  } else if (diag_num >= 580 && diag_num < 630) {
    return('Genitourinary')
  } else if (diag_num >= 800 && diag_num < 1000) {
    return('Injury')
  } else if (diag_num >= 140 && diag_num < 240) {
    return('Neoplasms')
  } else {
    return('Other')
  }
}

# 将出院去向分组
group_discharge_disposition <- function(discharge_id) {
  if (discharge_id == 1) {
    return('Home')
  } else if (discharge_id %in% c(2, 3, 4, 5, 6)) {
    return('Transfer')
  } else if (discharge_id %in% c(7, 8, 9, 10)) {
    return('Other_Facility')
  } else if (discharge_id %in% c(11, 12, 13, 14, 15, 16, 17, 18, 19)) {
    return('Other')
  } else if (discharge_id %in% c(20, 21, 22, 23, 24, 25, 26, 27, 28, 29)) {
    return('Death_Other')
  } else {
    return('Unknown')
  }
}

# 将入院来源分组
group_admission_source <- function(source_id) {
  if (source_id %in% c(1, 2, 3)) {
    return('Referral')
  } else if (source_id %in% c(4, 5, 6, 7, 8)) {
    return('Transfer')
  } else if (source_id %in% c(9, 10, 11, 12, 13, 14, 15, 16, 17)) {
    return('Emergency_Other')
  } else {
    return('Other')
  }
}

# ==================== 3. 特征工程 ====================
cat("\n[2] 开始特征工程...\n")

# 创建数据副本
df_fe <- df

# 3.1 诊断编码分组
cat("\n[2.1] 处理诊断编码（ICD-9分组）...\n")
df_fe$diag_1_group <- sapply(df_fe$diag_1, categorize_icd9)
df_fe$diag_2_group <- sapply(df_fe$diag_2, categorize_icd9)
df_fe$diag_3_group <- sapply(df_fe$diag_3, categorize_icd9)

cat("主要诊断分组分布:\n")
diag1_table <- table(df_fe$diag_1_group)
print(sort(diag1_table, decreasing = TRUE)[1:min(10, length(diag1_table))])

# 3.2 出院去向分组
cat("\n[2.2] 处理出院去向分组...\n")
df_fe$discharge_disposition_group <- sapply(df_fe$discharge_disposition_id, 
                                            group_discharge_disposition)
cat("出院去向分组分布:\n")
print(table(df_fe$discharge_disposition_group))

# 3.3 入院来源分组
cat("\n[2.3] 处理入院来源分组...\n")
df_fe$admission_source_group <- sapply(df_fe$admission_source_id, 
                                       group_admission_source)
cat("入院来源分组分布:\n")
print(table(df_fe$admission_source_group))

# 3.4 创建交互项（可选，在建模时再决定是否使用）
cat("\n[2.4] 创建交互项特征...\n")
# 年龄与诊断数量的交互（年龄分组需要先转换为数值）
# 这里先创建，后续在建模时再决定是否使用

# 3.5 处理高缺失率变量的指示变量（可选）
cat("\n[2.5] 创建高缺失率变量的指示变量...\n")
df_fe$has_weight <- !is.na(df_fe$weight) & df_fe$weight != 'Unknown'
df_fe$has_glucose <- !is.na(df_fe$max_glu_serum) & df_fe$max_glu_serum != 'Unknown'
df_fe$has_A1C <- !is.na(df_fe$A1Cresult) & df_fe$A1Cresult != 'Unknown'

cat("有体重数据:", sum(df_fe$has_weight), 
    sprintf("(%.2f%%)", mean(df_fe$has_weight) * 100), "\n")
cat("有血糖数据:", sum(df_fe$has_glucose), 
    sprintf("(%.2f%%)", mean(df_fe$has_glucose) * 100), "\n")
cat("有A1C数据:", sum(df_fe$has_A1C), 
    sprintf("(%.2f%%)", mean(df_fe$has_A1C) * 100), "\n")

# ==================== 4. 准备建模变量 ====================
cat("\n[3] 准备建模变量...\n")

# 4.1 回归模型变量（预测住院天数）
regression_features_base <- c(
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
)

regression_features_enhanced <- c(
  regression_features_base,
  # 增强特征
  'diag_1_group', 'diag_2_group', 'diag_3_group',
  'discharge_disposition_group', 'admission_source_group',
  'insulin_use', 'diabetesMed', 'medication_changed'
)

# 4.2 分类模型变量（预测30天再入院）
classification_features_base <- c(
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
)

classification_features_enhanced <- c(
  classification_features_base,
  # 增强特征
  'diag_1_group', 'diag_2_group', 'diag_3_group',
  'discharge_disposition_group', 'admission_source_group',
  'medication_changed',
  'num_lab_procedures', 'num_procedures'
)

cat("\n回归模型 - 基准特征数:", length(regression_features_base), "\n")
cat("回归模型 - 增强特征数:", length(regression_features_enhanced), "\n")
cat("分类模型 - 基准特征数:", length(classification_features_base), "\n")
cat("分类模型 - 增强特征数:", length(classification_features_enhanced), "\n")

# ==================== 5. 数据分割 ====================
cat("\n[4] 数据分割（训练集/测试集）...\n")

# 5.1 回归任务数据分割
X_reg <- df_fe[, regression_features_enhanced, drop = FALSE]
y_reg <- df_fe$time_in_hospital

# 5.2 分类任务数据分割
X_clf <- df_fe[, classification_features_enhanced, drop = FALSE]
y_clf <- df_fe$readmitted_30d

# 分割数据（80/20）
# 回归任务
trainIndex_reg <- createDataPartition(y_reg, p = 0.8, list = FALSE, times = 1)
X_reg_train <- X_reg[trainIndex_reg, , drop = FALSE]
X_reg_test <- X_reg[-trainIndex_reg, , drop = FALSE]
y_reg_train <- y_reg[trainIndex_reg]
y_reg_test <- y_reg[-trainIndex_reg]

# 分类任务（分层抽样）
trainIndex_clf <- createDataPartition(y_clf, p = 0.8, list = FALSE, times = 1)
X_clf_train <- X_clf[trainIndex_clf, , drop = FALSE]
X_clf_test <- X_clf[-trainIndex_clf, , drop = FALSE]
y_clf_train <- y_clf[trainIndex_clf]
y_clf_test <- y_clf[-trainIndex_clf]

cat("\n回归任务:\n")
cat("  训练集:", format(nrow(X_reg_train), big.mark = ","), "样本\n")
cat("  测试集:", format(nrow(X_reg_test), big.mark = ","), "样本\n")
cat("  训练集平均住院天数:", sprintf("%.2f天", mean(y_reg_train)), "\n")
cat("  测试集平均住院天数:", sprintf("%.2f天", mean(y_reg_test)), "\n")

cat("\n分类任务:\n")
cat("  训练集:", format(nrow(X_clf_train), big.mark = ","), "样本\n")
cat("  测试集:", format(nrow(X_clf_test), big.mark = ","), "样本\n")
cat("  训练集再入院率:", sprintf("%.2f%%", mean(y_clf_train) * 100), "\n")
cat("  测试集再入院率:", sprintf("%.2f%%", mean(y_clf_test) * 100), "\n")

# ==================== 6. 保存处理后的数据 ====================
cat("\n[5] 保存处理后的数据...\n")

# 保存特征工程后的完整数据
write_csv(df_fe, 'data/processed/diabetic_data_featured.csv')
cat("✓ 已保存特征工程后的数据: data/processed/diabetic_data_featured.csv\n")

# 保存训练集和测试集（用于后续建模）
# 创建数据分割列表
data_splits <- list(
  regression = list(
    X_train = X_reg_train,
    X_test = X_reg_test,
    y_train = y_reg_train,
    y_test = y_reg_test,
    features_base = regression_features_base,
    features_enhanced = regression_features_enhanced
  ),
  classification = list(
    X_train = X_clf_train,
    X_test = X_clf_test,
    y_train = y_clf_train,
    y_test = y_clf_test,
    features_base = classification_features_base,
    features_enhanced = classification_features_enhanced
  )
)

# 保存为RDS格式
saveRDS(data_splits, 'data/processed/data_splits.rds')
cat("✓ 已保存数据分割: data/processed/data_splits.rds\n")

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("阶段1完成！数据准备与特征工程已完成。\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

