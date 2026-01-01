# ============================================================================
# 分类模型（30天再入院预测）
# 功能：Logistic回归模型构建、评估与解读
# ============================================================================

# 加载必要的包
library(dplyr)
library(caret)
library(pROC)
library(ggplot2)
library(gridExtra)
library(stats)

# 设置中文字体（Windows）
if (.Platform$OS.type == "windows") {
  windowsFonts(SimHei = windowsFont("SimHei"))
  par(family = "SimHei")
}

# ==================== 1. 加载数据 ====================
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("阶段3：分类模型开发（30天再入院预测）\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

cat("\n[1] 加载数据...\n")
data_splits <- readRDS('data/processed/data_splits.rds')

clf_data <- data_splits$classification
X_train <- clf_data$X_train
X_test <- clf_data$X_test
y_train <- clf_data$y_train
y_test <- clf_data$y_test
features_base <- clf_data$features_base
features_enhanced <- clf_data$features_enhanced

cat("训练集:", format(nrow(X_train), big.mark = ","), "样本\n")
cat("测试集:", format(nrow(X_test), big.mark = ","), "样本\n")
cat("训练集再入院率:", sprintf("%.2f%%", mean(y_train) * 100), "\n")
cat("测试集再入院率:", sprintf("%.2f%%", mean(y_test) * 100), "\n")

# ==================== 2. 特征编码 ====================
encode_features <- function(X, encoders = NULL, fit = TRUE) {
  # 对分类变量进行编码
  X_encoded <- X
  
  # 识别分类变量（字符型或因子型）
  categorical_cols <- names(X)[sapply(X, function(x) is.character(x) || is.factor(x))]
  
  if (is.null(encoders)) {
    encoders <- list()
  }
  
  for (col in categorical_cols) {
    if (fit) {
      # 训练时：创建编码器
      X_encoded[[col]] <- as.character(X[[col]])
      levels <- unique(X_encoded[[col]])
      X_encoded[[col]] <- as.numeric(factor(X_encoded[[col]], levels = levels))
      encoders[[col]] <- list(levels = levels)
    } else {
      # 测试时：使用已有编码器
      if (col %in% names(encoders)) {
        X_encoded[[col]] <- as.character(X[[col]])
        # 处理未见过的类别
        X_encoded[[col]] <- ifelse(
          X_encoded[[col]] %in% encoders[[col]]$levels,
          match(X_encoded[[col]], encoders[[col]]$levels) - 1,
          0
        )
      }
    }
  }
  
  return(list(X_encoded = X_encoded, encoders = encoders))
}

cat("\n[2] 特征编码...\n")
encoded_train <- encode_features(X_train, fit = TRUE)
X_train_encoded <- encoded_train$X_encoded
encoders <- encoded_train$encoders

encoded_test <- encode_features(X_test, encoders = encoders, fit = FALSE)
X_test_encoded <- encoded_test$X_encoded

# ==================== 3. 处理类别不平衡 ====================
cat("\n[3] 计算类别权重...\n")
# 计算类别权重（平衡类别）
class_weights <- table(y_train)
class_weights <- 1 / (class_weights / sum(class_weights))
class_weight_dict <- as.list(class_weights)
names(class_weight_dict) <- names(class_weights)
cat("类别权重:", paste(names(class_weight_dict), "=", 
                       sprintf("%.4f", unlist(class_weight_dict)), collapse = ", "), "\n")

# 创建权重向量
weights_train <- ifelse(y_train == 1, 
                        class_weight_dict[["1"]], 
                        class_weight_dict[["0"]])

# ==================== 4. 基准模型 ====================
cat("\n[4] 构建基准模型...\n")

X_train_base <- X_train_encoded[, features_base, drop = FALSE]
X_test_base <- X_test_encoded[, features_base, drop = FALSE]

# 标准化
preProc_base <- preProcess(X_train_base, method = c("center", "scale"))
X_train_base_scaled <- predict(preProc_base, X_train_base)
X_test_base_scaled <- predict(preProc_base, X_test_base)

# 准备数据框用于建模
train_data_base <- data.frame(X_train_base_scaled, y = as.factor(y_train))
test_data_base <- data.frame(X_test_base_scaled, y = as.factor(y_test))

# 构建公式
formula_base <- as.formula(paste("y ~", paste(features_base, collapse = " + ")))

# 训练模型（使用权重处理不平衡）
model_base <- glm(formula_base, data = train_data_base, 
                  family = binomial(link = "logit"), 
                  weights = weights_train)

# 预测
y_train_pred_base <- predict(model_base, newdata = train_data_base, type = "response")
y_test_pred_base <- predict(model_base, newdata = test_data_base, type = "response")
y_train_pred_class_base <- ifelse(y_train_pred_base > 0.5, 1, 0)
y_test_pred_class_base <- ifelse(y_test_pred_base > 0.5, 1, 0)

# 评估指标
cm_train_base <- confusionMatrix(factor(y_train_pred_class_base), 
                                  factor(y_train), positive = "1")
cm_test_base <- confusionMatrix(factor(y_test_pred_class_base), 
                                 factor(y_test), positive = "1")

train_acc_base <- as.numeric(cm_train_base$overall["Accuracy"])
test_acc_base <- as.numeric(cm_test_base$overall["Accuracy"])
train_prec_base <- as.numeric(cm_train_base$byClass["Precision"])
test_prec_base <- as.numeric(cm_test_base$byClass["Precision"])
train_rec_base <- as.numeric(cm_train_base$byClass["Recall"])
test_rec_base <- as.numeric(cm_test_base$byClass["Recall"])
train_f1_base <- as.numeric(cm_train_base$byClass["F1"])
test_f1_base <- as.numeric(cm_test_base$byClass["F1"])

# AUC
train_auc_base <- as.numeric(auc(roc(y_train, y_train_pred_base)))
test_auc_base <- as.numeric(auc(roc(y_test, y_test_pred_base)))

cat("\n基准模型评估结果:\n")
cat("训练集 - 准确率:", sprintf("%.4f", train_acc_base), 
    ", 精确率:", sprintf("%.4f", train_prec_base), 
    ", 召回率:", sprintf("%.4f", train_rec_base), 
    ", F1:", sprintf("%.4f", train_f1_base), 
    ", AUC:", sprintf("%.4f", train_auc_base), "\n")
cat("测试集 - 准确率:", sprintf("%.4f", test_acc_base), 
    ", 精确率:", sprintf("%.4f", test_prec_base), 
    ", 召回率:", sprintf("%.4f", test_rec_base), 
    ", F1:", sprintf("%.4f", test_f1_base), 
    ", AUC:", sprintf("%.4f", test_auc_base), "\n")

# ==================== 5. 改进模型 ====================
cat("\n[5] 构建改进模型...\n")

X_train_enhanced <- X_train_encoded[, features_enhanced, drop = FALSE]
X_test_enhanced <- X_test_encoded[, features_enhanced, drop = FALSE]

# 标准化
preProc_enhanced <- preProcess(X_train_enhanced, method = c("center", "scale"))
X_train_enhanced_scaled <- predict(preProc_enhanced, X_train_enhanced)
X_test_enhanced_scaled <- predict(preProc_enhanced, X_test_enhanced)

# 准备数据框
train_data_enhanced <- data.frame(X_train_enhanced_scaled, y = as.factor(y_train))
test_data_enhanced <- data.frame(X_test_enhanced_scaled, y = as.factor(y_test))

# 构建公式
formula_enhanced <- as.formula(paste("y ~", paste(features_enhanced, collapse = " + ")))

# 训练模型
model_enhanced <- glm(formula_enhanced, data = train_data_enhanced, 
                      family = binomial(link = "logit"), 
                      weights = weights_train)

# 预测
y_train_pred_enhanced <- predict(model_enhanced, newdata = train_data_enhanced, type = "response")
y_test_pred_enhanced <- predict(model_enhanced, newdata = test_data_enhanced, type = "response")
y_train_pred_class_enhanced <- ifelse(y_train_pred_enhanced > 0.5, 1, 0)
y_test_pred_class_enhanced <- ifelse(y_test_pred_enhanced > 0.5, 1, 0)

# 评估指标
cm_train_enhanced <- confusionMatrix(factor(y_train_pred_class_enhanced), 
                                     factor(y_train), positive = "1")
cm_test_enhanced <- confusionMatrix(factor(y_test_pred_class_enhanced), 
                                    factor(y_test), positive = "1")

train_acc_enhanced <- as.numeric(cm_train_enhanced$overall["Accuracy"])
test_acc_enhanced <- as.numeric(cm_test_enhanced$overall["Accuracy"])
train_prec_enhanced <- as.numeric(cm_train_enhanced$byClass["Precision"])
test_prec_enhanced <- as.numeric(cm_test_enhanced$byClass["Precision"])
train_rec_enhanced <- as.numeric(cm_train_enhanced$byClass["Recall"])
test_rec_enhanced <- as.numeric(cm_test_enhanced$byClass["Recall"])
train_f1_enhanced <- as.numeric(cm_train_enhanced$byClass["F1"])
test_f1_enhanced <- as.numeric(cm_test_enhanced$byClass["F1"])

# AUC
train_auc_enhanced <- as.numeric(auc(roc(y_train, y_train_pred_enhanced)))
test_auc_enhanced <- as.numeric(auc(roc(y_test, y_test_pred_enhanced)))

cat("\n改进模型评估结果:\n")
cat("训练集 - 准确率:", sprintf("%.4f", train_acc_enhanced), 
    ", 精确率:", sprintf("%.4f", train_prec_enhanced), 
    ", 召回率:", sprintf("%.4f", train_rec_enhanced), 
    ", F1:", sprintf("%.4f", train_f1_enhanced), 
    ", AUC:", sprintf("%.4f", train_auc_enhanced), "\n")
cat("测试集 - 准确率:", sprintf("%.4f", test_acc_enhanced), 
    ", 精确率:", sprintf("%.4f", test_prec_enhanced), 
    ", 召回率:", sprintf("%.4f", test_rec_enhanced), 
    ", F1:", sprintf("%.4f", test_f1_enhanced), 
    ", AUC:", sprintf("%.4f", test_auc_enhanced), "\n")

# ==================== 6. OR值计算 ====================
cat("\n[6] 计算OR值（优势比）...\n")

# OR = exp(系数)
coef_base <- coef(model_base)
or_base <- data.frame(
  Feature = names(coef_base)[-1],  # 排除截距
  Coefficient = as.numeric(coef_base[-1]),
  OR = exp(as.numeric(coef_base[-1]))
)
or_base <- or_base[order(abs(or_base$OR - 1), decreasing = TRUE), ]

coef_enhanced <- coef(model_enhanced)
or_enhanced <- data.frame(
  Feature = names(coef_enhanced)[-1],
  Coefficient = as.numeric(coef_enhanced[-1]),
  OR = exp(as.numeric(coef_enhanced[-1]))
)
or_enhanced <- or_enhanced[order(abs(or_enhanced$OR - 1), decreasing = TRUE), ]

cat("\n基准模型 - 主要变量OR值:\n")
print(head(or_base, 10))

cat("\n改进模型 - 主要变量OR值:\n")
print(head(or_enhanced, 10))

# ==================== 7. 结果可视化 ====================
cat("\n[7] 生成可视化图表...\n")

if (!dir.exists('docs/figures')) {
  dir.create('docs/figures', recursive = TRUE)
}

# 7.1 OR值条形图
or_base_top <- head(or_base, 15)
or_enhanced_top <- head(or_enhanced, 15)

p1 <- ggplot(or_base_top, aes(x = reorder(Feature, OR), y = OR)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(x = "变量", y = "OR值", title = "基准模型 - 主要变量OR值") +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  theme_minimal() +
  theme(text = element_text(family = "SimHei", size = 10))

p2 <- ggplot(or_enhanced_top, aes(x = reorder(Feature, OR), y = OR)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(x = "变量", y = "OR值", title = "改进模型 - 主要变量OR值") +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  theme_minimal() +
  theme(text = element_text(family = "SimHei", size = 10))

png('docs/figures/09_classification_or_values.png', width = 16, height = 6, units = "in", res = 300)
grid.arrange(p1, p2, ncol = 2)
dev.off()
cat("✓ 已保存: docs/figures/09_classification_or_values.png\n")

# 7.2 混淆矩阵
cm_base_matrix <- as.matrix(cm_test_base$table)
cm_enhanced_matrix <- as.matrix(cm_test_enhanced$table)

png('docs/figures/10_classification_confusion_matrix.png', width = 14, height = 5, units = "in", res = 300)
par(mfrow = c(1, 2), family = "SimHei")

# 基准模型混淆矩阵
image(cm_base_matrix, col = heat.colors(10), 
      main = "基准模型 - 混淆矩阵", 
      xlab = "预测值", ylab = "实际值")
text(expand.grid(1:ncol(cm_base_matrix), 1:nrow(cm_base_matrix)), 
     labels = as.character(cm_base_matrix))

# 改进模型混淆矩阵
image(cm_enhanced_matrix, col = heat.colors(10), 
      main = "改进模型 - 混淆矩阵", 
      xlab = "预测值", ylab = "实际值")
text(expand.grid(1:ncol(cm_enhanced_matrix), 1:nrow(cm_enhanced_matrix)), 
     labels = as.character(cm_enhanced_matrix))

dev.off()
cat("✓ 已保存: docs/figures/10_classification_confusion_matrix.png\n")

# 7.3 ROC曲线
roc_base <- roc(y_test, y_test_pred_base)
roc_enhanced <- roc(y_test, y_test_pred_enhanced)

png('docs/figures/11_classification_roc_curve.png', width = 8, height = 6, units = "in", res = 300)
plot(roc_base, col = "blue", lwd = 2, 
     main = "ROC曲线", 
     xlab = "假阳性率 (FPR)", ylab = "真阳性率 (TPR)")
lines(roc_enhanced, col = "red", lwd = 2)
abline(0, 1, lty = 2, col = "black")
legend("bottomright", 
       legend = c(paste0("基准模型 (AUC=", sprintf("%.3f", test_auc_base), ")"),
                  paste0("改进模型 (AUC=", sprintf("%.3f", test_auc_enhanced), ")"),
                  "随机分类器"),
       col = c("blue", "red", "black"), lty = c(1, 1, 2), lwd = 2)
dev.off()
cat("✓ 已保存: docs/figures/11_classification_roc_curve.png\n")

# 7.4 预测概率分布
png('docs/figures/12_classification_probability_distribution.png', width = 14, height = 5, units = "in", res = 300)
par(mfrow = c(1, 2), family = "SimHei")

# 基准模型
hist(y_test_pred_base[y_test == 0], breaks = 50, col = rgb(0, 0, 1, 0.7), 
     xlab = "预测概率", ylab = "频数", 
     main = "基准模型 - 预测概率分布", border = "black")
hist(y_test_pred_base[y_test == 1], breaks = 50, col = rgb(1, 0, 0, 0.7), add = TRUE)
legend("topright", legend = c("未再入院", "再入院"), 
       fill = c(rgb(0, 0, 1, 0.7), rgb(1, 0, 0, 0.7)))

# 改进模型
hist(y_test_pred_enhanced[y_test == 0], breaks = 50, col = rgb(0, 0, 1, 0.7), 
     xlab = "预测概率", ylab = "频数", 
     main = "改进模型 - 预测概率分布", border = "black")
hist(y_test_pred_enhanced[y_test == 1], breaks = 50, col = rgb(1, 0, 0, 0.7), add = TRUE)
legend("topright", legend = c("未再入院", "再入院"), 
       fill = c(rgb(0, 0, 1, 0.7), rgb(1, 0, 0, 0.7)))

dev.off()
cat("✓ 已保存: docs/figures/12_classification_probability_distribution.png\n")

# ==================== 8. 保存模型 ====================
cat("\n[8] 保存模型...\n")

if (!dir.exists('models')) {
  dir.create('models', recursive = TRUE)
}

model_info <- list(
  model_base = model_base,
  model_enhanced = model_enhanced,
  preProc_base = preProc_base,
  preProc_enhanced = preProc_enhanced,
  encoders = encoders,
  features_base = features_base,
  features_enhanced = features_enhanced,
  class_weight_dict = class_weight_dict,
  or_base = or_base,
  or_enhanced = or_enhanced,
  metrics_base = list(
    train_acc = train_acc_base,
    test_acc = test_acc_base,
    train_prec = train_prec_base,
    test_prec = test_prec_base,
    train_rec = train_rec_base,
    test_rec = test_rec_base,
    train_f1 = train_f1_base,
    test_f1 = test_f1_base,
    train_auc = train_auc_base,
    test_auc = test_auc_base
  ),
  metrics_enhanced = list(
    train_acc = train_acc_enhanced,
    test_acc = test_acc_enhanced,
    train_prec = train_prec_enhanced,
    test_prec = test_prec_enhanced,
    train_rec = train_rec_enhanced,
    test_rec = test_rec_enhanced,
    train_f1 = train_f1_enhanced,
    test_f1 = test_f1_enhanced,
    train_auc = train_auc_enhanced,
    test_auc = test_auc_enhanced
  )
)

saveRDS(model_info, 'models/classification_model.rds')
cat("✓ 已保存: models/classification_model.rds\n")

# ==================== 9. 输出结果摘要 ====================
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("分类模型结果摘要\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("\n基准模型 (特征数:", length(features_base), "):\n")
cat("  测试集 - 准确率:", sprintf("%.4f", test_acc_base), 
    ", 精确率:", sprintf("%.4f", test_prec_base), "\n")
cat("  测试集 - 召回率:", sprintf("%.4f", test_rec_base), 
    ", F1:", sprintf("%.4f", test_f1_base), 
    ", AUC:", sprintf("%.4f", test_auc_base), "\n")

cat("\n改进模型 (特征数:", length(features_enhanced), "):\n")
cat("  测试集 - 准确率:", sprintf("%.4f", test_acc_enhanced), 
    ", 精确率:", sprintf("%.4f", test_prec_enhanced), "\n")
cat("  测试集 - 召回率:", sprintf("%.4f", test_rec_enhanced), 
    ", F1:", sprintf("%.4f", test_f1_enhanced), 
    ", AUC:", sprintf("%.4f", test_auc_enhanced), "\n")

cat("\n模型改进:\n")
cat("  AUC提升:", sprintf("%.4f", test_auc_enhanced - test_auc_base), "\n")
cat("  F1提升:", sprintf("%.4f", test_f1_enhanced - test_f1_base), "\n")

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("阶段3完成！分类模型已构建并评估。\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

