# ============================================================================
# 回归模型（住院天数预测）
# 功能：多元线性回归模型构建、评估与诊断
# ============================================================================

# 加载必要的包
library(dplyr)
library(caret)
library(car)          # VIF检查
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
cat("阶段2：回归模型开发（住院天数预测）\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

cat("\n[1] 加载数据...\n")
data_splits <- readRDS('data/processed/data_splits.rds')

reg_data <- data_splits$regression
X_train <- reg_data$X_train
X_test <- reg_data$X_test
y_train <- reg_data$y_train
y_test <- reg_data$y_test
features_base <- reg_data$features_base
features_enhanced <- reg_data$features_enhanced

cat("训练集:", format(nrow(X_train), big.mark = ","), "样本\n")
cat("测试集:", format(nrow(X_test), big.mark = ","), "样本\n")

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

# ==================== 3. 基准模型 ====================
cat("\n[3] 构建基准模型...\n")

# 使用基准特征集
X_train_base <- X_train_encoded[, features_base, drop = FALSE]
X_test_base <- X_test_encoded[, features_base, drop = FALSE]

# 标准化
preProc_base <- preProcess(X_train_base, method = c("center", "scale"))
X_train_base_scaled <- predict(preProc_base, X_train_base)
X_test_base_scaled <- predict(preProc_base, X_test_base)

# 准备数据框用于建模
train_data_base <- data.frame(X_train_base_scaled, y = y_train)
test_data_base <- data.frame(X_test_base_scaled, y = y_test)

# 构建公式
formula_base <- as.formula(paste("y ~", paste(features_base, collapse = " + ")))

# 训练模型
model_base <- lm(formula_base, data = train_data_base)

# 预测
y_train_pred_base <- predict(model_base, newdata = train_data_base)
y_test_pred_base <- predict(model_base, newdata = test_data_base)

# 评估指标
train_r2_base <- cor(y_train, y_train_pred_base)^2
test_r2_base <- cor(y_test, y_test_pred_base)^2
train_rmse_base <- sqrt(mean((y_train - y_train_pred_base)^2))
test_rmse_base <- sqrt(mean((y_test - y_test_pred_base)^2))
train_mae_base <- mean(abs(y_train - y_train_pred_base))
test_mae_base <- mean(abs(y_test - y_test_pred_base))

cat("\n基准模型评估结果:\n")
cat("训练集 - R²:", sprintf("%.4f", train_r2_base), 
    ", RMSE:", sprintf("%.4f", train_rmse_base), 
    ", MAE:", sprintf("%.4f", train_mae_base), "\n")
cat("测试集 - R²:", sprintf("%.4f", test_r2_base), 
    ", RMSE:", sprintf("%.4f", test_rmse_base), 
    ", MAE:", sprintf("%.4f", test_mae_base), "\n")

# ==================== 4. 改进模型 ====================
cat("\n[4] 构建改进模型...\n")

X_train_enhanced <- X_train_encoded[, features_enhanced, drop = FALSE]
X_test_enhanced <- X_test_encoded[, features_enhanced, drop = FALSE]

# 标准化
preProc_enhanced <- preProcess(X_train_enhanced, method = c("center", "scale"))
X_train_enhanced_scaled <- predict(preProc_enhanced, X_train_enhanced)
X_test_enhanced_scaled <- predict(preProc_enhanced, X_test_enhanced)

# 准备数据框
train_data_enhanced <- data.frame(X_train_enhanced_scaled, y = y_train)
test_data_enhanced <- data.frame(X_test_enhanced_scaled, y = y_test)

# 构建公式
formula_enhanced <- as.formula(paste("y ~", paste(features_enhanced, collapse = " + ")))

# 训练模型
model_enhanced <- lm(formula_enhanced, data = train_data_enhanced)

# 预测
y_train_pred_enhanced <- predict(model_enhanced, newdata = train_data_enhanced)
y_test_pred_enhanced <- predict(model_enhanced, newdata = test_data_enhanced)

# 评估指标
train_r2_enhanced <- cor(y_train, y_train_pred_enhanced)^2
test_r2_enhanced <- cor(y_test, y_test_pred_enhanced)^2
train_rmse_enhanced <- sqrt(mean((y_train - y_train_pred_enhanced)^2))
test_rmse_enhanced <- sqrt(mean((y_test - y_test_pred_enhanced)^2))
train_mae_enhanced <- mean(abs(y_train - y_train_pred_enhanced))
test_mae_enhanced <- mean(abs(y_test - y_test_pred_enhanced))

cat("\n改进模型评估结果:\n")
cat("训练集 - R²:", sprintf("%.4f", train_r2_enhanced), 
    ", RMSE:", sprintf("%.4f", train_rmse_enhanced), 
    ", MAE:", sprintf("%.4f", train_mae_enhanced), "\n")
cat("测试集 - R²:", sprintf("%.4f", test_r2_enhanced), 
    ", RMSE:", sprintf("%.4f", test_rmse_enhanced), 
    ", MAE:", sprintf("%.4f", test_mae_enhanced), "\n")

# ==================== 5. 模型诊断 ====================
cat("\n[5] 模型诊断...\n")

# 使用改进模型进行诊断
residuals <- y_test - y_test_pred_enhanced

# 5.1 残差正态性检验（Shapiro-Wilk，样本量大时用前5000个）
cat("\n残差正态性检验 (Shapiro-Wilk):\n")
n_test <- min(5000, length(residuals))
shapiro_test <- shapiro.test(residuals[1:n_test])
cat("  统计量:", sprintf("%.4f", shapiro_test$statistic), 
    ", p值:", sprintf("%.4f", shapiro_test$p.value), "\n")

# 5.2 多重共线性检查（VIF）
cat("\n多重共线性检查 (VIF):\n")
vif_values <- vif(model_enhanced)
vif_data <- data.frame(
  Variable = names(vif_values),
  VIF = as.numeric(vif_values)
)
vif_data <- vif_data[order(-vif_data$VIF), ]
print(head(vif_data, 10))
cat("\nVIF > 10 的变量数:", sum(vif_data$VIF > 10), "\n")

# ==================== 6. 结果可视化 ====================
cat("\n[6] 生成可视化图表...\n")

if (!dir.exists('docs/figures')) {
  dir.create('docs/figures', recursive = TRUE)
}

# 6.1 模型系数条形图
coef_base <- summary(model_base)$coefficients
coef_base_df <- data.frame(
  Feature = rownames(coef_base)[-1],  # 排除截距
  Coefficient = coef_base[-1, "Estimate"]
)
coef_base_df <- coef_base_df[order(abs(coef_base_df$Coefficient), decreasing = TRUE), ]
coef_base_df <- head(coef_base_df, 15)

coef_enhanced <- summary(model_enhanced)$coefficients
coef_enhanced_df <- data.frame(
  Feature = rownames(coef_enhanced)[-1],
  Coefficient = coef_enhanced[-1, "Estimate"]
)
coef_enhanced_df <- coef_enhanced_df[order(abs(coef_enhanced_df$Coefficient), decreasing = TRUE), ]
coef_enhanced_df <- head(coef_enhanced_df, 15)

p1 <- ggplot(coef_base_df, aes(x = reorder(Feature, Coefficient), y = Coefficient)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(x = "变量", y = "回归系数", title = "基准模型 - 主要变量系数") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  theme_minimal() +
  theme(text = element_text(family = "SimHei", size = 10))

p2 <- ggplot(coef_enhanced_df, aes(x = reorder(Feature, Coefficient), y = Coefficient)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(x = "变量", y = "回归系数", title = "改进模型 - 主要变量系数") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  theme_minimal() +
  theme(text = element_text(family = "SimHei", size = 10))

png('docs/figures/07_regression_coefficients.png', width = 16, height = 6, units = "in", res = 300)
grid.arrange(p1, p2, ncol = 2)
dev.off()
cat("✓ 已保存: docs/figures/07_regression_coefficients.png\n")

# 6.2 残差诊断图
png('docs/figures/08_regression_diagnostics.png', width = 14, height = 10, units = "in", res = 300)

par(mfrow = c(2, 2), family = "SimHei")

# 残差vs预测值
plot(y_test_pred_enhanced, residuals, 
     xlab = "预测值", ylab = "残差", 
     main = "残差 vs 预测值", pch = 20, cex = 0.5)
abline(h = 0, col = "red", lty = 2)

# Q-Q图
qqnorm(residuals, main = "残差Q-Q图（正态性检验）")
qqline(residuals, col = "red")

# 残差直方图
hist(residuals, breaks = 50, 
     xlab = "残差", ylab = "频数", 
     main = "残差分布", border = "black")

# 预测值vs实际值
plot(y_test, y_test_pred_enhanced, 
     xlab = "实际值", ylab = "预测值", 
     main = paste0("预测值 vs 实际值 (R²=", sprintf("%.3f", test_r2_enhanced), ")"),
     pch = 20, cex = 0.5)
abline(0, 1, col = "red", lty = 2, lwd = 2)

dev.off()
cat("✓ 已保存: docs/figures/08_regression_diagnostics.png\n")

# ==================== 7. 保存模型 ====================
cat("\n[7] 保存模型...\n")

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
  metrics_base = list(
    train_r2 = train_r2_base,
    test_r2 = test_r2_base,
    train_rmse = train_rmse_base,
    test_rmse = test_rmse_base,
    train_mae = train_mae_base,
    test_mae = test_mae_base
  ),
  metrics_enhanced = list(
    train_r2 = train_r2_enhanced,
    test_r2 = test_r2_enhanced,
    train_rmse = train_rmse_enhanced,
    test_rmse = test_rmse_enhanced,
    train_mae = train_mae_enhanced,
    test_mae = test_mae_enhanced
  ),
  vif_data = vif_data
)

saveRDS(model_info, 'models/regression_model.rds')
cat("✓ 已保存: models/regression_model.rds\n")

# ==================== 8. 输出结果摘要 ====================
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("回归模型结果摘要\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("\n基准模型 (特征数:", length(features_base), "):\n")
cat("  测试集 R²:", sprintf("%.4f", test_r2_base), "\n")
cat("  测试集 RMSE:", sprintf("%.4f", test_rmse_base), "天\n")
cat("  测试集 MAE:", sprintf("%.4f", test_mae_base), "天\n")

cat("\n改进模型 (特征数:", length(features_enhanced), "):\n")
cat("  测试集 R²:", sprintf("%.4f", test_r2_enhanced), "\n")
cat("  测试集 RMSE:", sprintf("%.4f", test_rmse_enhanced), "天\n")
cat("  测试集 MAE:", sprintf("%.4f", test_mae_enhanced), "天\n")

cat("\n模型改进:\n")
cat("  R²提升:", sprintf("%.4f", test_r2_enhanced - test_r2_base), "\n")
cat("  RMSE降低:", sprintf("%.4f", test_rmse_base - test_rmse_enhanced), "天\n")

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("阶段2完成！回归模型已构建并评估。\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

