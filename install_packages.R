# ============================================================================
# R包安装脚本
# 功能：安装项目所需的所有R包
# 使用方法：在R或RStudio中运行 source("install_packages.R")
# ============================================================================

# 需要安装的R包列表
packages <- c(
  # 数据处理
  "dplyr",
  "tidyr",
  "readr",
  
  # 建模
  "caret",
  "car",           # VIF检查
  "pROC",          # ROC曲线
  
  # 可视化
  "ggplot2",
  "gridExtra",
  
  # Shiny应用
  "shiny",
  "shinydashboard",
  "DT"             # 数据表格
)

# 安装缺失的包
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]

if(length(new_packages)) {
  cat("正在安装以下R包:\n")
  cat(paste(new_packages, collapse = ", "), "\n\n")
  install.packages(new_packages, dependencies = TRUE)
} else {
  cat("所有必需的R包已安装。\n")
}

# 验证安装
cat("\n验证包安装状态:\n")
for (pkg in packages) {
  if (require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat("✓", pkg, "\n")
  } else {
    cat("✗", pkg, "- 安装失败\n")
  }
}

cat("\n安装完成！\n")

