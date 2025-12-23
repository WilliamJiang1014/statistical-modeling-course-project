# ============================================================================
# 阶段4：原型系统开发
# 功能：Shiny交互式Web系统
# 作者：成员B
# ============================================================================

library(shiny)
library(shinydashboard)
library(DT)
library(ggplot2)
library(dplyr)
library(readr)

# ============================================================================
# UI界面定义
# ============================================================================

ui <- dashboardPage(
  dashboardHeader(
    title = "糖尿病再入院预测系统",
    titleWidth = 350,
    tags$li(class = "dropdown",
            tags$style(HTML("
              .main-header .logo {
                margin-left: 50px;
              }
              .main-header .navbar {
                margin-left: 0;
              }
            "))
    )
  ),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("首页", tabName = "home", icon = icon("home")),
      menuItem("数据概览", tabName = "overview", icon = icon("chart-bar")),
      menuItem("预处理展示", tabName = "preprocessing", icon = icon("cog")),
      menuItem("模型结果", tabName = "models", icon = icon("chart-line")),
      menuItem("风险预测", tabName = "prediction", icon = icon("calculator"))
    )
  ),
  
  dashboardBody(
    tags$head(
      tags$style(HTML("
        /* 标题样式 */
        .main-header .logo {
          font-weight: bold;
          font-size: 18px;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
          width: 100%;
          margin-left: 50px !important;
        }
        .main-header .navbar-brand {
          max-width: 350px;
        }
        /* 侧边栏按钮位置 */
        .main-header .sidebar-toggle {
          position: absolute;
          left: 0;
        }
        /* 内容区域 */
        .content-wrapper, .right-side {
          background-color: #f4f4f4;
        }
        /* 图片响应式显示 */
        .shiny-image-output {
          max-width: 100% !important;
          height: auto !important;
          width: 100% !important;
          display: block;
        }
        .shiny-image-output img {
          max-width: 100% !important;
          height: auto !important;
          width: 100% !important;
          object-fit: contain;
        }
        /* Box内容溢出处理 */
        .box-body {
          overflow-x: auto;
          overflow-y: auto;
          max-width: 100%;
        }
        /* 确保图片容器不溢出 */
        .box {
          overflow: hidden;
        }
      "))
    ),
    
    tabItems(
      # ==================== 首页 ====================
      tabItem(tabName = "home",
        h2("糖尿病再入院风险与住院天数预测系统"),
        hr(),
        fluidRow(
          box(
            width = 12,
            status = "primary",
            solidHeader = TRUE,
            title = "项目简介",
            p("本项目利用统计建模方法，分析影响糖尿病患者",
              strong("30天内再入院风险"), "和", strong("住院天数"), 
              "的关键因素，构建可解释的预测模型，为临床决策提供支持。")
          )
        ),
        fluidRow(
          valueBoxOutput("total_records"),
          valueBoxOutput("total_vars"),
          valueBoxOutput("readmission_rate"),
          valueBoxOutput("avg_los")
        ),
        hr(),
        fluidRow(
          box(
            width = 12,
            title = "模型方法",
            p(strong("多元线性回归"), "（住院天数预测） | ",
              strong("Logistic回归"), "（再入院风险预测）")
          )
        )
      ),
      
      # ==================== 数据概览 ====================
      tabItem(tabName = "overview",
        h2("数据概览与可视化"),
        hr(),
        fluidRow(
          valueBoxOutput("overview_total"),
          valueBoxOutput("overview_vars"),
          valueBoxOutput("overview_readmission"),
          valueBoxOutput("overview_los")
        ),
        hr(),
        fluidRow(
          box(
            width = 6,
            title = "再入院情况分布",
            DT::dataTableOutput("readmission_table")
          ),
          box(
            width = 6,
            title = "住院天数统计",
            DT::dataTableOutput("los_table")
          )
        ),
        hr(),
        # 探索性数据分析图表
        fluidRow(
          box(width = 6, title = "再入院情况分布",
              imageOutput("eda_img1", height = "auto")),
          box(width = 6, title = "住院天数分布",
              imageOutput("eda_img2", height = "auto"))
        ),
        fluidRow(
          box(width = 6, title = "人口学特征分布",
              imageOutput("eda_img3", height = "auto")),
          box(width = 6, title = "不同分组的再入院率",
              imageOutput("eda_img4", height = "auto"))
        ),
        fluidRow(
          box(width = 6, title = "不同分组的住院天数",
              imageOutput("eda_img5", height = "auto")),
          box(width = 6, title = "连续变量相关性热力图",
              imageOutput("eda_img6", height = "auto"))
        )
      ),
      
      # ==================== 预处理展示 ====================
      tabItem(tabName = "preprocessing",
        h2("数据预处理展示"),
        hr(),
        fluidRow(
          box(
            width = 12,
            title = "数据清洗流程",
            tags$ul(
              tags$li("删除重复记录：检查并删除重复的住院记录"),
              tags$li("处理特殊值：将'?'替换为NaN后统一处理"),
              tags$li("创建因变量：readmitted_30d（30天内再入院，0/1）、time_in_hospital（住院天数）"),
              tags$li("处理缺失值：关键变量无缺失值，分类变量缺失值填充为'Unknown'"),
              tags$li("创建衍生变量：insulin_use、medication_changed")
            )
          )
        ),
        hr(),
        fluidRow(
          box(
            width = 12,
            title = "特征工程",
            tags$ul(
              tags$li("诊断编码分组（ICD-9）：将ICD-9编码分组为糖尿病相关、心血管相关、呼吸系统相关等类别"),
              tags$li("分类变量分组：出院去向（30类→5类）、入院来源（25类→4类）")
            )
          )
        ),
        hr(),
        fluidRow(
          box(
            width = 6,
            title = "关键变量缺失值统计",
            DT::dataTableOutput("missing_table")
          ),
          box(
            width = 6,
            title = "数据基本信息",
            DT::dataTableOutput("info_table")
          )
        )
      ),
      
      # ==================== 模型结果 ====================
      tabItem(tabName = "models",
        h2("模型结果展示"),
        hr(),
        tabsetPanel(
          # 回归模型
          tabPanel("回归模型（住院天数预测）",
            fluidRow(
              box(
                width = 12,
                title = "模型性能对比",
                DT::dataTableOutput("regression_comparison")
              )
            ),
            fluidRow(
              valueBoxOutput("reg_r2"),
              valueBoxOutput("reg_rmse"),
              valueBoxOutput("reg_mae")
            ),
            hr(),
            fluidRow(
              box(
                width = 12,
                title = "回归系数分析",
                imageOutput("regression_coefficients", height = "auto")
              )
            ),
            fluidRow(
              box(
                width = 12,
                title = "残差诊断",
                imageOutput("regression_diagnostics", height = "auto")
              )
            )
          ),
          
          # 分类模型
          tabPanel("分类模型（再入院预测）",
            fluidRow(
              box(
                width = 12,
                title = "模型性能对比",
                DT::dataTableOutput("classification_comparison")
              )
            ),
            fluidRow(
              valueBoxOutput("clf_acc"),
              valueBoxOutput("clf_prec"),
              valueBoxOutput("clf_rec"),
              valueBoxOutput("clf_f1"),
              valueBoxOutput("clf_auc")
            ),
            hr(),
            fluidRow(
              box(
                width = 12,
                title = "OR值（优势比）分析",
                imageOutput("classification_or", height = "auto")
              )
            ),
            fluidRow(
              box(
                width = 12,
                title = "混淆矩阵",
                imageOutput("classification_cm", height = "auto")
              )
            ),
            fluidRow(
              box(
                width = 12,
                title = "ROC曲线",
                imageOutput("classification_roc", height = "auto")
              )
            )
          )
        )
      ),
      
      # ==================== 风险预测 ====================
      tabItem(tabName = "prediction",
        h2("交互式风险预测"),
        hr(),
        fluidRow(
          box(
            width = 12,
            status = "info",
            solidHeader = TRUE,
            title = "使用说明",
            p("输入患者特征信息，系统将基于训练好的统计模型预测患者的",
              strong("30天再入院风险"), "和", strong("预计住院天数"), 
              "，为临床决策提供数据支持。")
          )
        ),
        hr(),
        fluidRow(
          box(
            width = 6,
            title = "基本信息",
            selectInput("pred_age", "年龄组", 
                       choices = c("[0-10)", "[10-20)", "[20-30)", "[30-40)", 
                                 "[40-50)", "[50-60)", "[60-70)", "[70-80)", 
                                 "[80-90)", "[90-100)")),
            selectInput("pred_gender", "性别", 
                       choices = c("Male", "Female", "Unknown/Invalid")),
            selectInput("pred_race", "种族", 
                       choices = c("Caucasian", "AfricanAmerican", "Asian", 
                                 "Hispanic", "Other", "Unknown"))
          ),
          box(
            width = 6,
            title = "诊断与治疗信息",
            sliderInput("pred_number_diagnoses", "诊断数量", 
                       min = 1, max = 16, value = 7),
            sliderInput("pred_num_medications", "用药数量", 
                       min = 1, max = 81, value = 15),
            selectInput("pred_insulin_use", "是否使用胰岛素", 
                       choices = list("否" = 0, "是" = 1)),
            selectInput("pred_diabetesMed", "是否使用糖尿病药物", 
                       choices = c("Yes", "No"))
          )
        ),
        fluidRow(
          box(
            width = 6,
            title = "既往就医史",
            numericInput("pred_number_outpatient", "既往门诊次数", 
                        value = 0, min = 0, max = 50),
            numericInput("pred_number_emergency", "既往急诊次数", 
                        value = 0, min = 0, max = 50),
            numericInput("pred_number_inpatient", "既往住院次数", 
                        value = 0, min = 0, max = 20)
          ),
          box(
            width = 6,
            title = "本次住院信息",
            sliderInput("pred_num_lab_procedures", "实验室检查次数", 
                       min = 1, max = 132, value = 40),
            sliderInput("pred_num_procedures", "操作/手术次数", 
                       min = 0, max = 6, value = 0),
            selectInput("pred_admission_type_id", "入院类型", 
                       choices = 1:8),
            selectInput("pred_admission_source_id", "入院来源", 
                       choices = 1:25)
          )
        ),
        hr(),
        fluidRow(
          actionButton("predict_btn", "开始预测", 
                      class = "btn-primary", 
                      style = "width: 100%; font-size: 18px; padding: 10px;")
        ),
        hr(),
        fluidRow(
          box(
            width = 6,
            title = "预测住院天数",
            valueBoxOutput("pred_los", width = 12),
            uiOutput("pred_los_status"),
            plotOutput("pred_los_plot", height = "200px")
          ),
          box(
            width = 6,
            title = "30天再入院风险",
            valueBoxOutput("pred_readmission", width = 12),
            uiOutput("pred_readmission_status"),
            plotOutput("pred_readmission_plot", height = "200px")
          )
        ),
        hr(),
        fluidRow(
          box(
            width = 12,
            title = "主要风险因素",
            tags$ul(
              tags$li(strong("诊断数量"), "：诊断数量越多，再入院风险和住院天数通常越高"),
              tags$li(strong("既往就医史"), "：既往住院/急诊次数多的患者，再入院风险显著更高"),
              tags$li(strong("用药情况"), "：使用胰岛素的患者可能需要更长的住院时间"),
              tags$li(strong("年龄"), "：老年患者通常住院时间更长，再入院风险更高")
            )
          )
        )
      )
    )
  )
)

# ============================================================================
# 服务器逻辑
# ============================================================================

server <- function(input, output, session) {
  
  # 获取项目根目录路径
  # 尝试多种可能的路径
  possible_roots <- c(
    getwd(),  # 当前工作目录
    normalizePath(file.path(getwd(), "..")),  # 上一级目录
    normalizePath(file.path(getwd(), "..", "..")),  # 上两级目录
    normalizePath(file.path(dirname(normalizePath("app.R", mustWork = FALSE)), "..", ".."))
  )
  
  project_root <- NULL
  for (root in possible_roots) {
    if (file.exists(file.path(root, "data", "processed", "diabetic_data_cleaned.csv"))) {
      project_root <- root
      break
    }
  }
  
  # 如果找不到，使用当前工作目录
  if (is.null(project_root)) {
    project_root <- getwd()
  }
  
  # 输出调试信息（可以在RStudio控制台看到）
  cat("项目根目录:", project_root, "\n")
  cat("数据文件存在:", file.exists(file.path(project_root, "data", "processed", "diabetic_data_cleaned.csv")), "\n")
  
  # 加载数据（缓存）
  df_data <- reactive({
    tryCatch({
      data_path <- file.path(project_root, 'data/processed/diabetic_data_cleaned.csv')
      if (file.exists(data_path)) {
        read_csv(data_path, locale = locale(encoding = "UTF-8"))
      } else {
        # 尝试相对路径
        read_csv('data/processed/diabetic_data_cleaned.csv', 
                 locale = locale(encoding = "UTF-8"))
      }
    }, error = function(e) {
      cat("数据加载错误:", e$message, "\n")
      return(NULL)
    })
  })
  
  # 加载模型（缓存）
  regression_model <- reactive({
    model_path1 <- file.path(project_root, 'models/regression_model.rds')
    model_path2 <- 'models/regression_model.rds'
    
    if (file.exists(model_path1)) {
      readRDS(model_path1)
    } else if (file.exists(model_path2)) {
      readRDS(model_path2)
    } else {
      NULL
    }
  })
  
  classification_model <- reactive({
    model_path1 <- file.path(project_root, 'models/classification_model.rds')
    model_path2 <- 'models/classification_model.rds'
    
    if (file.exists(model_path1)) {
      readRDS(model_path1)
    } else if (file.exists(model_path2)) {
      readRDS(model_path2)
    } else {
      NULL
    }
  })
  
  # ==================== 首页 ====================
  output$total_records <- renderValueBox({
    df <- df_data()
    if (!is.null(df)) {
      valueBox(
        format(nrow(df), big.mark = ","),
        "数据规模",
        icon = icon("database"),
        color = "blue"
      )
    } else {
      valueBox("加载中...", "数据规模", color = "blue")
    }
  })
  
  output$total_vars <- renderValueBox({
    df <- df_data()
    if (!is.null(df)) {
      valueBox(
        ncol(df),
        "变量数",
        icon = icon("list"),
        color = "green"
      )
    } else {
      valueBox("加载中...", "变量数", color = "green")
    }
  })
  
  output$readmission_rate <- renderValueBox({
    df <- df_data()
    if (!is.null(df)) {
      valueBox(
        sprintf("%.2f%%", mean(df$readmitted_30d) * 100),
        "再入院率",
        icon = icon("exclamation-triangle"),
        color = "yellow"
      )
    } else {
      valueBox("加载中...", "再入院率", color = "yellow")
    }
  })
  
  output$avg_los <- renderValueBox({
    df <- df_data()
    if (!is.null(df)) {
      valueBox(
        sprintf("%.2f天", mean(df$time_in_hospital)),
        "平均住院天数",
        icon = icon("clock"),
        color = "red"
      )
    } else {
      valueBox("加载中...", "平均住院天数", color = "red")
    }
  })
  
  # ==================== 数据概览 ====================
  output$overview_total <- renderValueBox({
    df <- df_data()
    if (!is.null(df)) {
      valueBox(format(nrow(df), big.mark = ","), "总记录数", color = "blue")
    } else {
      valueBox("加载中...", "总记录数", color = "blue")
    }
  })
  
  output$overview_vars <- renderValueBox({
    df <- df_data()
    if (!is.null(df)) {
      valueBox(ncol(df), "变量数", color = "green")
    } else {
      valueBox("加载中...", "变量数", color = "green")
    }
  })
  
  output$overview_readmission <- renderValueBox({
    df <- df_data()
    if (!is.null(df)) {
      valueBox(sprintf("%.2f%%", mean(df$readmitted_30d) * 100), 
               "30天再入院率", color = "yellow")
    } else {
      valueBox("加载中...", "30天再入院率", color = "yellow")
    }
  })
  
  output$overview_los <- renderValueBox({
    df <- df_data()
    if (!is.null(df)) {
      valueBox(sprintf("%.2f天", mean(df$time_in_hospital)), 
               "平均住院天数", color = "red")
    } else {
      valueBox("加载中...", "平均住院天数", color = "red")
    }
  })
  
  output$readmission_table <- DT::renderDataTable({
    df <- df_data()
    if (!is.null(df)) {
      readmission_counts <- table(df$readmitted_30d)
      data.frame(
        类别 = c("未再入院", "30天内再入院"),
        数量 = c(readmission_counts["0"], readmission_counts["1"]),
        比例 = c(
          sprintf("%.2f%%", readmission_counts["0"] / nrow(df) * 100),
          sprintf("%.2f%%", readmission_counts["1"] / nrow(df) * 100)
        )
      )
    }
  }, options = list(pageLength = 10, dom = 't'))
  
  output$los_table <- DT::renderDataTable({
    df <- df_data()
    if (!is.null(df)) {
      los_stats <- summary(df$time_in_hospital)
      data.frame(
        统计量 = c("最小值", "25%分位数", "中位数", "75%分位数", "最大值", "平均值"),
        天数 = c(
          sprintf("%.1f", los_stats["Min."]),
          sprintf("%.1f", los_stats["1st Qu."]),
          sprintf("%.1f", los_stats["Median"]),
          sprintf("%.1f", los_stats["3rd Qu."]),
          sprintf("%.1f", los_stats["Max."]),
          sprintf("%.1f", mean(df$time_in_hospital))
        )
      )
    }
  }, options = list(pageLength = 10, dom = 't'))
  
  # EDA图片显示（使用renderImage，更可靠）
  output$eda_img1 <- renderImage({
    fig_path1 <- file.path(project_root, 'docs/figures/01_readmission_distribution.png')
    fig_path2 <- 'docs/figures/01_readmission_distribution.png'
    fig_path <- if (file.exists(fig_path1)) fig_path1 else fig_path2
    list(src = fig_path, 
         width = "100%", 
         alt = "再入院情况分布")
  }, deleteFile = FALSE)
  
  output$eda_img2 <- renderImage({
    fig_path1 <- file.path(project_root, 'docs/figures/02_length_of_stay_distribution.png')
    fig_path2 <- 'docs/figures/02_length_of_stay_distribution.png'
    fig_path <- if (file.exists(fig_path1)) fig_path1 else fig_path2
    list(src = fig_path, 
         width = "100%", 
         alt = "住院天数分布")
  }, deleteFile = FALSE)
  
  output$eda_img3 <- renderImage({
    fig_path1 <- file.path(project_root, 'docs/figures/03_demographic_distribution.png')
    fig_path2 <- 'docs/figures/03_demographic_distribution.png'
    fig_path <- if (file.exists(fig_path1)) fig_path1 else fig_path2
    list(src = fig_path, 
         width = "100%", 
         alt = "人口学特征分布")
  }, deleteFile = FALSE)
  
  output$eda_img4 <- renderImage({
    fig_path1 <- file.path(project_root, 'docs/figures/04_readmission_by_groups.png')
    fig_path2 <- 'docs/figures/04_readmission_by_groups.png'
    fig_path <- if (file.exists(fig_path1)) fig_path1 else fig_path2
    list(src = fig_path, 
         width = "100%", 
         alt = "不同分组的再入院率")
  }, deleteFile = FALSE)
  
  output$eda_img5 <- renderImage({
    fig_path1 <- file.path(project_root, 'docs/figures/05_length_of_stay_by_groups.png')
    fig_path2 <- 'docs/figures/05_length_of_stay_by_groups.png'
    fig_path <- if (file.exists(fig_path1)) fig_path1 else fig_path2
    list(src = fig_path, 
         width = "100%", 
         alt = "不同分组的住院天数")
  }, deleteFile = FALSE)
  
  output$eda_img6 <- renderImage({
    fig_path1 <- file.path(project_root, 'docs/figures/06_correlation_heatmap.png')
    fig_path2 <- 'docs/figures/06_correlation_heatmap.png'
    fig_path <- if (file.exists(fig_path1)) fig_path1 else fig_path2
    list(src = fig_path, 
         width = "100%", 
         alt = "连续变量相关性热力图")
  }, deleteFile = FALSE)
  
  # 保留原有的uiOutput用于兼容
  output$eda_images <- renderUI({
    NULL  # 现在使用单独的renderImage输出
  })
  
  # ==================== 预处理展示 ====================
  output$missing_table <- DT::renderDataTable({
    df <- df_data()
    if (!is.null(df)) {
      key_vars <- c('readmitted_30d', 'time_in_hospital', 'age', 'gender', 
                   'number_diagnoses', 'num_medications')
      data.frame(
        变量 = key_vars,
        缺失值数 = sapply(key_vars, function(x) sum(is.na(df[[x]]))),
        缺失率 = sapply(key_vars, function(x) 
          sprintf("%.2f%%", sum(is.na(df[[x]])) / nrow(df) * 100))
      )
    }
  }, options = list(pageLength = 10, dom = 't'))
  
  output$info_table <- DT::renderDataTable({
    df <- df_data()
    if (!is.null(df)) {
      data.frame(
        指标 = c('总记录数', '变量数', '30天再入院数', '平均住院天数'),
        数值 = c(
          format(nrow(df), big.mark = ","),
          ncol(df),
          format(sum(df$readmitted_30d), big.mark = ","),
          sprintf("%.2f天", mean(df$time_in_hospital))
        )
      )
    }
  }, options = list(pageLength = 10, dom = 't'))
  
  # ==================== 模型结果 ====================
  # 回归模型
  output$regression_comparison <- DT::renderDataTable({
    model <- regression_model()
    if (!is.null(model)) {
      metrics_base <- model$metrics_base
      metrics_enhanced <- model$metrics_enhanced
      data.frame(
        指标 = c('R²', 'RMSE', 'MAE'),
        基准模型 = c(
          sprintf("%.4f", metrics_base$test_r2),
          sprintf("%.4f", metrics_base$test_rmse),
          sprintf("%.4f", metrics_base$test_mae)
        ),
        改进模型 = c(
          sprintf("%.4f", metrics_enhanced$test_r2),
          sprintf("%.4f", metrics_enhanced$test_rmse),
          sprintf("%.4f", metrics_enhanced$test_mae)
        ),
        提升 = c(
          sprintf("+%.4f", metrics_enhanced$test_r2 - metrics_base$test_r2),
          sprintf("-%.4f", metrics_base$test_rmse - metrics_enhanced$test_rmse),
          sprintf("-%.4f", metrics_base$test_mae - metrics_enhanced$test_mae)
        )
      )
    }
  }, options = list(pageLength = 10, dom = 't'))
  
  output$reg_r2 <- renderValueBox({
    model <- regression_model()
    if (!is.null(model)) {
      valueBox(sprintf("%.4f", model$metrics_enhanced$test_r2), 
               "测试集 R²", color = "blue")
    } else {
      valueBox("未加载", "测试集 R²", color = "blue")
    }
  })
  
  output$reg_rmse <- renderValueBox({
    model <- regression_model()
    if (!is.null(model)) {
      valueBox(sprintf("%.4f天", model$metrics_enhanced$test_rmse), 
               "测试集 RMSE", color = "red")
    } else {
      valueBox("未加载", "测试集 RMSE", color = "red")
    }
  })
  
  output$reg_mae <- renderValueBox({
    model <- regression_model()
    if (!is.null(model)) {
      valueBox(sprintf("%.4f天", model$metrics_enhanced$test_mae), 
               "测试集 MAE", color = "yellow")
    } else {
      valueBox("未加载", "测试集 MAE", color = "yellow")
    }
  })
  
  output$regression_coefficients <- renderImage({
    fig_path1 <- file.path(project_root, 'docs/figures/07_regression_coefficients.png')
    fig_path2 <- 'docs/figures/07_regression_coefficients.png'
    fig_path <- if (file.exists(fig_path1)) fig_path1 else fig_path2
    
    list(src = fig_path,
         width = "100%",
         alt = "回归系数图")
  }, deleteFile = FALSE)
  
  output$regression_diagnostics <- renderImage({
    fig_path1 <- file.path(project_root, 'docs/figures/08_regression_diagnostics.png')
    fig_path2 <- 'docs/figures/08_regression_diagnostics.png'
    fig_path <- if (file.exists(fig_path1)) fig_path1 else fig_path2
    
    list(src = fig_path,
         width = "100%",
         alt = "残差诊断图")
  }, deleteFile = FALSE)
  
  # 分类模型
  output$classification_comparison <- DT::renderDataTable({
    model <- classification_model()
    if (!is.null(model)) {
      metrics_base <- model$metrics_base
      metrics_enhanced <- model$metrics_enhanced
      data.frame(
        指标 = c('准确率', '精确率', '召回率', 'F1值', 'AUC'),
        基准模型 = c(
          sprintf("%.4f", metrics_base$test_acc),
          sprintf("%.4f", metrics_base$test_prec),
          sprintf("%.4f", metrics_base$test_rec),
          sprintf("%.4f", metrics_base$test_f1),
          sprintf("%.4f", metrics_base$test_auc)
        ),
        改进模型 = c(
          sprintf("%.4f", metrics_enhanced$test_acc),
          sprintf("%.4f", metrics_enhanced$test_prec),
          sprintf("%.4f", metrics_enhanced$test_rec),
          sprintf("%.4f", metrics_enhanced$test_f1),
          sprintf("%.4f", metrics_enhanced$test_auc)
        ),
        提升 = c(
          sprintf("%+.4f", metrics_enhanced$test_acc - metrics_base$test_acc),
          sprintf("%+.4f", metrics_enhanced$test_prec - metrics_base$test_prec),
          sprintf("%+.4f", metrics_enhanced$test_rec - metrics_base$test_rec),
          sprintf("%+.4f", metrics_enhanced$test_f1 - metrics_base$test_f1),
          sprintf("%+.4f", metrics_enhanced$test_auc - metrics_base$test_auc)
        )
      )
    }
  }, options = list(pageLength = 10, dom = 't'))
  
  output$clf_acc <- renderValueBox({
    model <- classification_model()
    if (!is.null(model)) {
      valueBox(sprintf("%.4f", model$metrics_enhanced$test_acc), 
               "准确率", color = "blue")
    } else {
      valueBox("未加载", "准确率", color = "blue")
    }
  })
  
  output$clf_prec <- renderValueBox({
    model <- classification_model()
    if (!is.null(model)) {
      valueBox(sprintf("%.4f", model$metrics_enhanced$test_prec), 
               "精确率", color = "green")
    } else {
      valueBox("未加载", "精确率", color = "green")
    }
  })
  
  output$clf_rec <- renderValueBox({
    model <- classification_model()
    if (!is.null(model)) {
      valueBox(sprintf("%.4f", model$metrics_enhanced$test_rec), 
               "召回率", color = "yellow")
    } else {
      valueBox("未加载", "召回率", color = "yellow")
    }
  })
  
  output$clf_f1 <- renderValueBox({
    model <- classification_model()
    if (!is.null(model)) {
      valueBox(sprintf("%.4f", model$metrics_enhanced$test_f1), 
               "F1值", color = "orange")
    } else {
      valueBox("未加载", "F1值", color = "orange")
    }
  })
  
  output$clf_auc <- renderValueBox({
    model <- classification_model()
    if (!is.null(model)) {
      valueBox(sprintf("%.4f", model$metrics_enhanced$test_auc), 
               "AUC", color = "red")
    } else {
      valueBox("未加载", "AUC", color = "red")
    }
  })
  
  output$classification_or <- renderImage({
    fig_path1 <- file.path(project_root, 'docs/figures/09_classification_or_values.png')
    fig_path2 <- 'docs/figures/09_classification_or_values.png'
    fig_path <- if (file.exists(fig_path1)) fig_path1 else fig_path2
    
    list(src = fig_path,
         width = "100%",
         alt = "OR值图")
  }, deleteFile = FALSE)
  
  output$classification_cm <- renderImage({
    fig_path1 <- file.path(project_root, 'docs/figures/10_classification_confusion_matrix.png')
    fig_path2 <- 'docs/figures/10_classification_confusion_matrix.png'
    fig_path <- if (file.exists(fig_path1)) fig_path1 else fig_path2
    
    list(src = fig_path,
         width = "100%",
         alt = "混淆矩阵")
  }, deleteFile = FALSE)
  
  output$classification_roc <- renderImage({
    fig_path1 <- file.path(project_root, 'docs/figures/11_classification_roc_curve.png')
    fig_path2 <- 'docs/figures/11_classification_roc_curve.png'
    fig_path <- if (file.exists(fig_path1)) fig_path1 else fig_path2
    
    list(src = fig_path,
         width = "100%",
         alt = "ROC曲线")
  }, deleteFile = FALSE)
  
  # ==================== 风险预测 ====================
  prediction_result <- eventReactive(input$predict_btn, {
    reg_model <- regression_model()
    clf_model <- classification_model()
    
    if (is.null(reg_model) || is.null(clf_model)) {
      return(list(error = "模型尚未加载，请先运行建模代码生成模型文件。"))
    }
    
    # 准备输入数据
    input_data <- data.frame(
      age = input$pred_age,
      gender = input$pred_gender,
      race = input$pred_race,
      number_outpatient = input$pred_number_outpatient,
      number_emergency = input$pred_number_emergency,
      number_inpatient = input$pred_number_inpatient,
      number_diagnoses = input$pred_number_diagnoses,
      num_lab_procedures = input$pred_num_lab_procedures,
      num_procedures = input$pred_num_procedures,
      num_medications = input$pred_num_medications,
      admission_type_id = as.numeric(input$pred_admission_type_id),
      admission_source_id = as.numeric(input$pred_admission_source_id),
      insulin_use = as.numeric(input$pred_insulin_use),
      diabetesMed = input$pred_diabetesMed,
      diag_1_group = "Other",
      diag_2_group = "Other",
      diag_3_group = "Other",
      discharge_disposition_group = "Home",
      admission_source_group = "Other",
      medication_changed = 0
    )
    
    # 回归模型预测
    reg_features <- reg_model$features_enhanced
    X_reg <- input_data[, reg_features, drop = FALSE]
    
    # 编码特征
    encoders <- reg_model$encoders
    for (col in names(X_reg)) {
      if (col %in% names(encoders)) {
        X_reg[[col]] <- as.character(X_reg[[col]])
        X_reg[[col]] <- ifelse(
          X_reg[[col]] %in% encoders[[col]]$levels,
          match(X_reg[[col]], encoders[[col]]$levels) - 1,
          0
        )
      }
    }
    
    # 标准化
    X_reg_scaled <- predict(reg_model$preProc_enhanced, X_reg)
    X_reg_scaled_df <- data.frame(X_reg_scaled)
    
    # 预测
    predicted_los <- predict(reg_model$model_enhanced, newdata = X_reg_scaled_df)
    
    # 分类模型预测
    clf_features <- clf_model$features_enhanced
    X_clf <- input_data[, clf_features, drop = FALSE]
    
    # 编码特征
    encoders_clf <- clf_model$encoders
    for (col in names(X_clf)) {
      if (col %in% names(encoders_clf)) {
        X_clf[[col]] <- as.character(X_clf[[col]])
        X_clf[[col]] <- ifelse(
          X_clf[[col]] %in% encoders_clf[[col]]$levels,
          match(X_clf[[col]], encoders_clf[[col]]$levels) - 1,
          0
        )
      }
    }
    
    # 标准化
    X_clf_scaled <- predict(clf_model$preProc_enhanced, X_clf)
    X_clf_scaled_df <- data.frame(X_clf_scaled)
    X_clf_scaled_df$y <- factor(0, levels = c("0", "1"))
    
    # 预测
    readmission_proba <- predict(clf_model$model_enhanced, 
                                 newdata = X_clf_scaled_df, 
                                 type = "response")
    
    return(list(
      predicted_los = as.numeric(predicted_los),
      readmission_proba = as.numeric(readmission_proba)
    ))
  })
  
  output$pred_los <- renderValueBox({
    result <- prediction_result()
    if (!is.null(result$error)) {
      valueBox("错误", result$error, color = "red")
    } else {
      valueBox(
        sprintf("%.1f天", result$predicted_los),
        "预测住院天数",
        icon = icon("calendar"),
        color = "blue"
      )
    }
  })
  
  output$pred_readmission <- renderValueBox({
    result <- prediction_result()
    if (!is.null(result$error)) {
      valueBox("错误", result$error, color = "red")
    } else {
      valueBox(
        sprintf("%.2f%%", result$readmission_proba * 100),
        "30天再入院风险",
        icon = icon("exclamation-triangle"),
        color = ifelse(result$readmission_proba > 0.2, "red", 
                      ifelse(result$readmission_proba > 0.1, "yellow", "green"))
      )
    }
  })
  
  output$pred_los_status <- renderUI({
    result <- prediction_result()
    if (!is.null(result$error)) {
      return(NULL)
    }
    if (result$predicted_los < 4) {
      tags$div(class = "alert alert-success", 
               "预计住院时间较短（<4天）")
    } else if (result$predicted_los < 7) {
      tags$div(class = "alert alert-info", 
               "预计住院时间中等（4-7天）")
    } else {
      tags$div(class = "alert alert-warning", 
               "预计住院时间较长（>7天）")
    }
  })
  
  output$pred_readmission_status <- renderUI({
    result <- prediction_result()
    if (!is.null(result$error)) {
      return(NULL)
    }
    if (result$readmission_proba < 0.1) {
      tags$div(class = "alert alert-success", 
               "再入院风险较低（<10%）")
    } else if (result$readmission_proba < 0.2) {
      tags$div(class = "alert alert-info", 
               "再入院风险中等（10-20%）")
    } else {
      tags$div(class = "alert alert-danger", 
               "再入院风险较高（>20%），建议加强随访")
    }
  })
  
  output$pred_los_plot <- renderPlot({
    result <- prediction_result()
    if (!is.null(result$error)) {
      return(NULL)
    }
    barplot(result$predicted_los, 
            names.arg = "", 
            ylab = "住院天数",
            col = "steelblue",
            ylim = c(0, max(15, result$predicted_los * 1.2)))
    abline(h = 4, col = "green", lty = 2, lwd = 2)
    abline(h = 7, col = "orange", lty = 2, lwd = 2)
  })
  
  output$pred_readmission_plot <- renderPlot({
    result <- prediction_result()
    if (!is.null(result$error)) {
      return(NULL)
    }
    color <- ifelse(result$readmission_proba < 0.1, "green",
                   ifelse(result$readmission_proba < 0.2, "orange", "red"))
    barplot(result$readmission_proba, 
            names.arg = "", 
            ylab = "再入院概率",
            col = color,
            ylim = c(0, 1))
    abline(h = 0.1, col = "green", lty = 2, lwd = 2)
    abline(h = 0.2, col = "orange", lty = 2, lwd = 2)
  })
}

# 运行应用
shinyApp(ui = ui, server = server)

