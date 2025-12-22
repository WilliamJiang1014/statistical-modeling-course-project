# 统计分析与建模课程项目（Statistical Modeling Course Project）

本仓库用于存放课程 **《统计分析与建模》期末项目** 的全部内容，包括数据、代码、文档和演示材料。

## 一、项目主题

- **选题方向**：方向 3——糖尿病患者 30 天再入院风险与住院天数建模  
- **研究目标**：
  - 利用公开的糖尿病住院数据集，分析影响 **30 天内再入院** 与 **住院天数（Length of Stay, LOS）** 的主要因素；
  - 构建可解释的统计模型，包括：
    - **分类模型**：预测患者是否会在出院后 30 天内再次入院（Logistic回归）；
    - **回归模型**：预测本次住院天数（多元线性回归）；
  - 开发一个简单的原型系统，用于可视化数据和模型结果，并支持交互式风险评估。

## 二、项目结构

```
statistical-modeling-course-project/
├── code/                          # 代码文件目录
│   ├── 01_data_exploration.py    # 数据探索与基本信息统计
│   ├── 02_variable_documentation.py  # 变量说明表生成
│   ├── 03_data_cleaning.py       # 数据清洗流程
│   ├── 04_eda_visualization.py   # 探索性数据分析与可视化
│   ├── 05_data_sampling.py       # 数据抽样（控制数据规模）
│   ├── 06_feature_engineering.py # 特征工程与数据分割
│   ├── 07_regression_model.py    # 回归模型（住院天数预测）
│   ├── 08_classification_model.py # 分类模型（再入院预测）
│   └── 09_streamlit_app.py       # Streamlit交互式Web系统
│
├── data/                          # 数据文件目录
│   ├── raw/                       # 原始数据
│   │   └── diabetes_130_us_hospitals_1999_2008/
│   │       ├── diabetic_data.csv          # 原始数据（101,766条，18MB）
│   │       ├── diabetic_data_sampled.csv  # 抽样数据（16,777条，2.9MB）
│   │       └── IDS_mapping.csv            # ID映射表
│   └── processed/                 # 处理后数据
│       ├── diabetic_data_cleaned.csv      # 清洗后数据（16,777条，4.0MB）
│       ├── diabetic_data_featured.csv     # 特征工程后数据
│       └── data_splits.pkl                # 训练/测试集分割结果
│
├── models/                        # 模型文件目录
│   ├── regression_model.pkl      # 回归模型文件
│   └── classification_model.pkl  # 分类模型文件
│
├── docs/                          # 文档目录
│   ├── figures/                  # 可视化图表（12张PNG图片）
│   │   ├── 01_readmission_distribution.png
│   │   ├── 02_length_of_stay_distribution.png
│   │   ├── 03_demographic_distribution.png
│   │   ├── 04_readmission_by_groups.png
│   │   ├── 05_length_of_stay_by_groups.png
│   │   ├── 06_correlation_heatmap.png
│   │   ├── 07_regression_coefficients.png
│   │   ├── 08_regression_diagnostics.png
│   │   ├── 09_classification_or_values.png
│   │   ├── 10_classification_confusion_matrix.png
│   │   ├── 11_classification_roc_curve.png
│   │   └── 12_classification_probability_distribution.png
│   ├── 02_variable_documentation.csv  # 变量说明表
│   ├── 项目文件说明.md            # 项目文件结构说明
│   ├── 图表分析结论.md            # EDA图表分析结论
│   ├── 建模技术说明.md            # 建模技术详细说明
│   ├── 数据抽样说明.md            # 数据抽样方法说明
│   ├── 成员B_模型结果摘要.md      # 模型结果摘要
│   └── 给成员B_建模变量选择说明.md # 变量选择说明
│
├── README.md                      # 项目主README文件
└── requirements.txt               # Python依赖包列表
```

## 三、数据说明

- **数据集**：Diabetes 130-US hospitals for years 1999-2008
- **原始规模**：101,766条记录，50个变量，18MB
- **数据规模控制**：通过分层抽样将数据控制在10MB以内（抽样后：16,777条记录，2.9MB）
- 详细说明见 `docs/数据抽样说明.md`

## 四、运行说明

### 4.1 环境要求

- **Python版本**：Python 3.8 或更高版本
- **操作系统**：Windows / Linux / macOS
- **依赖包**：见 `requirements.txt`

主要依赖包包括：
- `pandas>=1.5.0` - 数据处理
- `numpy>=1.23.0` - 数值计算
- `scikit-learn>=1.2.0` - 机器学习
- `matplotlib>=3.6.0` - 数据可视化
- `seaborn>=0.12.0` - 统计可视化
- `scipy>=1.9.0` - 科学计算
- `statsmodels>=0.13.0` - 统计建模
- `streamlit>=1.25.0` - Web应用框架
- `Pillow>=9.0.0` - 图像处理

### 4.2 环境配置

#### 方法一：使用pip安装（推荐）
```bash
# 安装所有依赖包
pip install -r requirements.txt
```

#### 方法二：使用conda环境（可选）
```bash
# 创建conda环境
conda create -n statistical_modeling python=3.8
conda activate statistical_modeling

# 安装依赖包
pip install -r requirements.txt
```

### 4.3 运行步骤

#### 步骤1：数据准备（可选，数据已处理）
如果数据文件已存在，可跳过此步骤。如需重新处理数据：
```bash
# 数据探索
python code/01_data_exploration.py

# 生成变量说明表
python code/02_variable_documentation.py

# 数据清洗
python code/03_data_cleaning.py

# EDA可视化（生成6张图表）
python code/04_eda_visualization.py

# 数据抽样（如需要重新抽样）
python code/05_data_sampling.py
```

#### 步骤2：特征工程与数据分割
```bash
python code/06_feature_engineering.py
```
**输出**：
- `data/processed/diabetic_data_featured.csv` - 特征工程后的数据
- `data/processed/data_splits.pkl` - 训练/测试集分割结果

#### 步骤3：模型构建
```bash
# 回归模型（住院天数预测）
python code/07_regression_model.py

# 分类模型（30天再入院预测）
python code/08_classification_model.py
```
**输出**：
- `models/regression_model.pkl` - 回归模型文件
- `models/classification_model.pkl` - 分类模型文件
- `docs/figures/07-12_*.png` - 模型评估图表（6张）

#### 步骤4：启动交互式系统
```bash
streamlit run code/09_streamlit_app.py
```
系统将在浏览器中自动打开（默认地址：`http://localhost:8501`），提供以下功能：
- 数据概览与EDA图表展示
- 预处理结果展示
- 模型评估指标与可视化
- 交互式风险预测

### 4.4 快速开始（数据已处理）

如果数据文件已存在，可以直接从特征工程或模型构建开始：
```bash
# 特征工程（如需要）
python code/06_feature_engineering.py

# 模型构建
python code/07_regression_model.py
python code/08_classification_model.py

# 启动系统
streamlit run code/09_streamlit_app.py
```

### 4.5 注意事项

1. **路径设置**：所有代码文件中的路径均为相对路径，请确保在项目根目录下运行
2. **中文显示**：图表中的中文显示需要系统支持中文字体（如SimHei），Windows系统通常已自带
3. **模型文件**：如果模型文件不存在，需要先运行模型构建脚本（步骤3）
4. **数据文件**：如果数据文件不存在，需要先运行数据准备脚本（步骤1）
5. **Streamlit端口**：如果8501端口被占用，Streamlit会自动使用下一个可用端口

详细说明见 `docs/项目文件说明.md` 和 `docs/建模技术说明.md`。

## 五、与课程要求的对应关系

本项目在以下方面对应课程与期末项目要求：

- **数据可视化与探索分析**：对关键变量和关系进行可视化，生成12张图表，并给出明确结论（见 `docs/图表分析结论.md`）；
- **数据预处理与统计建模**：包含数据清洗、特征处理、回归模型与分类模型的构建与比较（见 `docs/建模技术说明.md`）；
- **模型评估与解读**：使用合适的评价指标（如 \(R^2\)、RMSE、准确率、召回率、AUC 等），并对模型参数进行合理解读（见 `docs/成员B_模型结果摘要.md`）；
- **原型系统展示**：通过Streamlit系统展示数据、预处理结果、模型输出和评价指标，支持交互式风险评估；
- **开源与协作**：通过 GitHub 公开项目仓库，提交完整的数据、代码、文档和项目小视频，展示团队协作过程。

