"""
阶段4：原型系统开发
功能：Streamlit交互式Web系统
作者：成员B
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# 页面配置
st.set_page_config(
    page_title="糖尿病再入院风险与住院天数预测系统",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    /* 主标题样式 */
    .main-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1f77b4;
        text-align: left;
        margin-bottom: 1.5rem;
        padding: 0.75rem 0;
        border-bottom: 2px solid #e0e0e0;
    }
    
    /* 卡片样式 */
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    
    /* 侧边栏样式 */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* 按钮样式 */
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #1565a0;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* 分隔线样式 */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #1f77b4, transparent);
        margin: 2rem 0;
    }
    
    /* 信息框样式 */
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    
    /* 侧边栏导航标题样式 */
    .nav-title {
        text-align: left;
        padding: 0.5rem 0;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        font-weight: 600;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 侧边栏导航 ====================
st.sidebar.markdown('<div class="nav-title">导航菜单</div>', unsafe_allow_html=True)

page = st.sidebar.radio(
    "选择页面",
    ["首页", "数据概览", "预处理展示", "模型结果", "风险预测"],
    label_visibility="collapsed"
)

# ==================== 加载数据和模型 ====================
@st.cache_data
def load_data():
    """加载数据"""
    df = pd.read_csv('data/processed/diabetic_data_cleaned.csv')
    return df

@st.cache_data
def load_featured_data():
    """加载特征工程后的数据"""
    if os.path.exists('data/processed/diabetic_data_featured.csv'):
        return pd.read_csv('data/processed/diabetic_data_featured.csv')
    return None

@st.cache_resource
def load_models():
    """加载模型"""
    models = {}
    if os.path.exists('models/regression_model.pkl'):
        with open('models/regression_model.pkl', 'rb') as f:
            models['regression'] = pickle.load(f)
    if os.path.exists('models/classification_model.pkl'):
        with open('models/classification_model.pkl', 'rb') as f:
            models['classification'] = pickle.load(f)
    return models

# ==================== 首页 ====================
if page == "首页":
    st.markdown('<div class="main-title">糖尿病再入院风险与住院天数预测系统</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("""
    <div class="info-box">
        <p>本项目利用统计建模方法，分析影响糖尿病患者<strong>30天内再入院风险</strong>和<strong>住院天数</strong>的关键因素，
        构建可解释的预测模型，为临床决策提供支持。</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 数据信息
    try:
        df = load_data()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("数据规模", f"{len(df):,}条")
        with col2:
            st.metric("变量数", len(df.columns))
        with col3:
            st.metric("再入院率", f"{df['readmitted_30d'].mean():.2%}")
        with col4:
            st.metric("平均住院天数", f"{df['time_in_hospital'].mean():.2f}天")
    except:
        st.info("数据加载中...")
    
    st.markdown("---")
    st.markdown("**模型方法：** 多元线性回归（住院天数） | Logistic回归（再入院风险）")

# ==================== 数据概览 ====================
elif page == "数据概览":
    st.markdown('<div class="main-title">数据概览与可视化</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    try:
        df = load_data()
        
        # 基本信息指标
        st.markdown("### 数据概览指标")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总记录数", f"{len(df):,}")
        with col2:
            st.metric("变量数", len(df.columns))
        with col3:
            st.metric("30天再入院率", f"{df['readmitted_30d'].mean():.2%}")
        with col4:
            st.metric("平均住院天数", f"{df['time_in_hospital'].mean():.2f}天")
        
        st.markdown("---")
        
        # 数据分布统计
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 再入院情况分布")
            readmission_counts = df['readmitted_30d'].value_counts().sort_index()
            readmission_df = pd.DataFrame({
                '类别': ['未再入院', '30天内再入院'],
                '数量': [readmission_counts.get(0, 0), readmission_counts.get(1, 0)],
                '比例': [
                    f"{readmission_counts.get(0, 0)/len(df):.2%}",
                    f"{readmission_counts.get(1, 0)/len(df):.2%}"
                ]
            })
            st.dataframe(readmission_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### 住院天数统计")
            los_stats = df['time_in_hospital'].describe()
            los_df = pd.DataFrame({
                '统计量': ['最小值', '25%分位数', '中位数', '75%分位数', '最大值', '平均值'],
                '天数': [
                    f"{los_stats['min']:.1f}",
                    f"{los_stats['25%']:.1f}",
                    f"{los_stats['50%']:.1f}",
                    f"{los_stats['75%']:.1f}",
                    f"{los_stats['max']:.1f}",
                    f"{los_stats['mean']:.1f}"
                ]
            })
            st.dataframe(los_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # 展示EDA图表
        st.markdown("### 探索性数据分析图表")
        figure_files = [
            ('01_readmission_distribution.png', '再入院情况分布'),
            ('02_length_of_stay_distribution.png', '住院天数分布'),
            ('03_demographic_distribution.png', '人口学特征分布'),
            ('04_readmission_by_groups.png', '不同分组的再入院率'),
            ('05_length_of_stay_by_groups.png', '不同分组的住院天数'),
            ('06_correlation_heatmap.png', '连续变量相关性热力图')
        ]
        
        for fig_file, fig_title in figure_files:
            fig_path = f'docs/figures/{fig_file}'
            if os.path.exists(fig_path):
                st.subheader(fig_title)
                st.image(fig_path, use_container_width=True)
        
    except Exception as e:
        st.error(f"加载数据时出错: {str(e)}")

# ==================== 预处理展示 ====================
elif page == "预处理展示":
    st.markdown('<div class="main-title">数据预处理展示</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 数据清洗流程
    st.markdown("### 数据清洗流程")
    st.markdown("""
    - **删除重复记录**：检查并删除重复的住院记录
    - **处理特殊值**：将'?'替换为NaN后统一处理
    - **创建因变量**：`readmitted_30d`（30天内再入院，0/1）、`time_in_hospital`（住院天数）
    - **处理缺失值**：关键变量无缺失值，分类变量缺失值填充为'Unknown'
    - **创建衍生变量**：`insulin_use`、`medication_changed`
    """)
    
    st.markdown("---")
    
    # 特征工程
    st.markdown("### 特征工程")
    st.markdown("""
    - **诊断编码分组（ICD-9）**：将ICD-9编码分组为糖尿病相关、心血管相关、呼吸系统相关等类别
    - **分类变量分组**：出院去向（30类→5类）、入院来源（25类→4类）
    """)
    
    st.markdown("---")
    
    # 数据质量统计
    try:
        df = load_data()
        st.markdown("### 数据质量统计")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 关键变量缺失值统计")
            key_vars = ['readmitted_30d', 'time_in_hospital', 'age', 'gender', 
                       'number_diagnoses', 'num_medications']
            missing_stats = pd.DataFrame({
                '变量': key_vars,
                '缺失值数': [df[var].isna().sum() for var in key_vars],
                '缺失率': [f"{df[var].isna().sum()/len(df):.2%}" for var in key_vars]
            })
            st.dataframe(missing_stats, use_container_width=True, hide_index=True)
            
            if missing_stats['缺失值数'].sum() == 0:
                st.success("所有关键变量均无缺失值，数据质量良好。")
        
        with col2:
            st.markdown("#### 数据基本信息")
            info_data = {
                '指标': ['总记录数', '变量数', '30天再入院数', '平均住院天数'],
                '数值': [
                    f"{len(df):,}",
                    len(df.columns),
                    f"{df['readmitted_30d'].sum():,}",
                    f"{df['time_in_hospital'].mean():.2f}天"
                ]
            }
            st.dataframe(pd.DataFrame(info_data), use_container_width=True, hide_index=True)
    
    except Exception as e:
        st.warning(f"无法加载数据统计: {str(e)}")

# ==================== 模型结果 ====================
elif page == "模型结果":
    st.markdown('<div class="main-title">模型结果展示</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    models = load_models()
    
    # 使用选项卡组织回归和分类模型
    tab_reg, tab_clf = st.tabs(["回归模型（住院天数预测）", "分类模型（再入院预测）"])
    
    with tab_reg:
        st.markdown("### 回归模型（住院天数预测）")
        
        if 'regression' in models:
            reg_model = models['regression']
            metrics_base = reg_model['metrics_base']
            metrics_enhanced = reg_model['metrics_enhanced']
            
            # 模型对比
            st.markdown("#### 模型性能对比")
            comparison_data = {
                '指标': ['R²', 'RMSE', 'MAE'],
                '基准模型': [
                    f"{metrics_base['test_r2']:.4f}",
                    f"{metrics_base['test_rmse']:.4f}",
                    f"{metrics_base['test_mae']:.4f}"
                ],
                '改进模型': [
                    f"{metrics_enhanced['test_r2']:.4f}",
                    f"{metrics_enhanced['test_rmse']:.4f}",
                    f"{metrics_enhanced['test_mae']:.4f}"
                ],
                '提升': [
                    f"+{metrics_enhanced['test_r2'] - metrics_base['test_r2']:.4f}",
                    f"-{metrics_base['test_rmse'] - metrics_enhanced['test_rmse']:.4f}",
                    f"-{metrics_base['test_mae'] - metrics_enhanced['test_mae']:.4f}"
                ]
            }
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.markdown("#### 改进模型评估指标")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("测试集 R²", f"{metrics_enhanced['test_r2']:.4f}")
            with col2:
                st.metric("测试集 RMSE", f"{metrics_enhanced['test_rmse']:.4f}天")
            with col3:
                st.metric("测试集 MAE", f"{metrics_enhanced['test_mae']:.4f}天")
            
            # 展示图表
            st.markdown("---")
            if os.path.exists('docs/figures/07_regression_coefficients.png'):
                st.markdown("#### 回归系数分析")
                st.image('docs/figures/07_regression_coefficients.png', use_container_width=True)
            
            if os.path.exists('docs/figures/08_regression_diagnostics.png'):
                st.markdown("#### 残差诊断")
                st.image('docs/figures/08_regression_diagnostics.png', use_container_width=True)
        else:
            st.warning("回归模型尚未训练，请先运行建模代码。")
    
    with tab_clf:
        st.markdown("### 分类模型（30天再入院预测）")
        
        if 'classification' in models:
            clf_model = models['classification']
            metrics_base = clf_model['metrics_base']
            metrics_enhanced = clf_model['metrics_enhanced']
            
            # 模型对比
            st.markdown("#### 模型性能对比")
            comparison_data = {
                '指标': ['准确率', '精确率', '召回率', 'F1值', 'AUC'],
                '基准模型': [
                    f"{metrics_base['test_acc']:.4f}",
                    f"{metrics_base['test_prec']:.4f}",
                    f"{metrics_base['test_rec']:.4f}",
                    f"{metrics_base['test_f1']:.4f}",
                    f"{metrics_base['test_auc']:.4f}"
                ],
                '改进模型': [
                    f"{metrics_enhanced['test_acc']:.4f}",
                    f"{metrics_enhanced['test_prec']:.4f}",
                    f"{metrics_enhanced['test_rec']:.4f}",
                    f"{metrics_enhanced['test_f1']:.4f}",
                    f"{metrics_enhanced['test_auc']:.4f}"
                ],
                '提升': [
                    f"{metrics_enhanced['test_acc'] - metrics_base['test_acc']:+.4f}",
                    f"{metrics_enhanced['test_prec'] - metrics_base['test_prec']:+.4f}",
                    f"{metrics_enhanced['test_rec'] - metrics_base['test_rec']:+.4f}",
                    f"{metrics_enhanced['test_f1'] - metrics_base['test_f1']:+.4f}",
                    f"{metrics_enhanced['test_auc'] - metrics_base['test_auc']:+.4f}"
                ]
            }
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.markdown("#### 改进模型评估指标")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("准确率", f"{metrics_enhanced['test_acc']:.4f}")
            with col2:
                st.metric("精确率", f"{metrics_enhanced['test_prec']:.4f}")
            with col3:
                st.metric("召回率", f"{metrics_enhanced['test_rec']:.4f}")
            with col4:
                st.metric("F1值", f"{metrics_enhanced['test_f1']:.4f}")
            with col5:
                st.metric("AUC", f"{metrics_enhanced['test_auc']:.4f}")
            
            # 展示图表
            st.markdown("---")
            if os.path.exists('docs/figures/09_classification_or_values.png'):
                st.markdown("#### OR值（优势比）分析")
                st.image('docs/figures/09_classification_or_values.png', use_container_width=True)
            
            if os.path.exists('docs/figures/10_classification_confusion_matrix.png'):
                st.markdown("#### 混淆矩阵")
                st.image('docs/figures/10_classification_confusion_matrix.png', use_container_width=True)
            
            if os.path.exists('docs/figures/11_classification_roc_curve.png'):
                st.markdown("#### ROC曲线")
                st.image('docs/figures/11_classification_roc_curve.png', use_container_width=True)
        else:
            st.warning("分类模型尚未训练，请先运行建模代码。")

# ==================== 风险预测 ====================
elif page == "风险预测":
    st.markdown('<div class="main-title">交互式风险预测</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        <p style='font-size: 1.1rem; margin: 0;'>
        输入患者特征信息，系统将基于训练好的统计模型预测患者的<strong>30天再入院风险</strong>和<strong>预计住院天数</strong>，
        为临床决策提供数据支持。
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    models = load_models()
    
    if 'regression' not in models or 'classification' not in models:
        st.error("模型尚未加载，请先运行建模代码生成模型文件。")
        st.stop()
    
    reg_model = models['regression']
    clf_model = models['classification']
    
    # 用户输入界面
    st.markdown("### 患者特征输入")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 基本信息")
        age = st.selectbox("年龄组", ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', 
                                      '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'])
        gender = st.selectbox("性别", ['Male', 'Female', 'Unknown/Invalid'])
        race = st.selectbox("种族", ['Caucasian', 'AfricanAmerican', 'Asian', 'Hispanic', 'Other', 'Unknown'])
        
        st.markdown("#### 诊断与治疗信息")
        number_diagnoses = st.slider("诊断数量", 1, 16, 7)
        num_medications = st.slider("用药数量", 1, 81, 15)
        insulin_use = st.selectbox("是否使用胰岛素", [0, 1], format_func=lambda x: '是' if x == 1 else '否')
        diabetesMed = st.selectbox("是否使用糖尿病药物", ['Yes', 'No'])
    
    with col2:
        st.markdown("#### 既往就医史")
        number_outpatient = st.number_input("既往门诊次数", 0, 50, 0)
        number_emergency = st.number_input("既往急诊次数", 0, 50, 0)
        number_inpatient = st.number_input("既往住院次数", 0, 20, 0)
        
        st.markdown("#### 本次住院信息")
        num_lab_procedures = st.slider("实验室检查次数", 1, 132, 40)
        num_procedures = st.slider("操作/手术次数", 0, 6, 0)
        admission_type_id = st.selectbox("入院类型", [1, 2, 3, 4, 5, 6, 7, 8])
        admission_source_id = st.selectbox("入院来源", list(range(1, 26)))
    
    # 预测按钮
    st.markdown("---")
    if st.button("开始预测", type="primary", use_container_width=True):
        try:
            # 准备输入数据
            input_data = {
                'age': [age],
                'gender': [gender],
                'race': [race],
                'number_outpatient': [number_outpatient],
                'number_emergency': [number_emergency],
                'number_inpatient': [number_inpatient],
                'number_diagnoses': [number_diagnoses],
                'num_lab_procedures': [num_lab_procedures],
                'num_procedures': [num_procedures],
                'num_medications': [num_medications],
                'admission_type_id': [admission_type_id],
                'admission_source_id': [admission_source_id],
                'insulin_use': [insulin_use],
                'diabetesMed': [diabetesMed]
            }
            
            # 添加增强特征（简化处理）
            input_data['diag_1_group'] = ['Other']  # 默认值
            input_data['diag_2_group'] = ['Other']
            input_data['diag_3_group'] = ['Other']
            input_data['discharge_disposition_group'] = ['Home']
            input_data['admission_source_group'] = ['Other']
            input_data['medication_changed'] = [0]
            
            # 转换为DataFrame
            input_df = pd.DataFrame(input_data)
            
            # 编码特征
            from sklearn.preprocessing import LabelEncoder
            
            # 回归模型预测
            reg_features = reg_model['features_enhanced']
            X_reg = input_df[reg_features].copy()
            
            # 编码
            for col in X_reg.select_dtypes(include=['object']).columns:
                if col in reg_model['encoders']:
                    le = reg_model['encoders'][col]
                    X_reg[col] = X_reg[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else 0
                    )
            
            X_reg_scaled = reg_model['scaler_enhanced'].transform(X_reg)
            predicted_los = reg_model['model_enhanced'].predict(X_reg_scaled)[0]
            
            # 分类模型预测
            clf_features = clf_model['features_enhanced']
            X_clf = input_df[clf_features].copy()
            
            # 编码
            for col in X_clf.select_dtypes(include=['object']).columns:
                if col in clf_model['encoders']:
                    le = clf_model['encoders'][col]
                    X_clf[col] = X_clf[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else 0
                    )
            
            X_clf_scaled = clf_model['scaler_enhanced'].transform(X_clf)
            readmission_proba = clf_model['model_enhanced'].predict_proba(X_clf_scaled)[0, 1]
            
            # 显示结果
            st.markdown("---")
            st.markdown("### 预测结果")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 预测住院天数")
                st.metric("", f"{predicted_los:.1f}天", 
                         help="基于多元线性回归模型预测的住院天数")
                
                # 风险等级提示
                if predicted_los < 4:
                    st.success("预计住院时间较短（<4天）")
                elif predicted_los < 7:
                    st.info("预计住院时间中等（4-7天）")
                else:
                    st.warning("预计住院时间较长（>7天）")
                
                # 可视化
                fig, ax = plt.subplots(figsize=(6, 2))
                ax.barh([0], [predicted_los], color='#1f77b4', alpha=0.7)
                ax.set_xlim(0, max(15, predicted_los * 1.2))
                ax.set_xlabel('住院天数')
                ax.set_yticks([])
                ax.axvline(x=4, color='green', linestyle='--', alpha=0.5, label='短（<4天）')
                ax.axvline(x=7, color='orange', linestyle='--', alpha=0.5, label='中（4-7天）')
                ax.legend(loc='lower right', fontsize=8)
                st.pyplot(fig)
            
            with col2:
                st.markdown("#### 30天再入院风险")
                st.metric("", f"{readmission_proba:.2%}",
                         help="基于Logistic回归模型预测的30天内再入院概率")
                
                # 风险等级提示
                if readmission_proba < 0.1:
                    st.success("再入院风险较低（<10%）")
                elif readmission_proba < 0.2:
                    st.info("再入院风险中等（10-20%）")
                else:
                    st.error("再入院风险较高（>20%），建议加强随访")
                
                # 可视化
                fig, ax = plt.subplots(figsize=(6, 2))
                colors = ['green' if readmission_proba < 0.1 else 
                         'orange' if readmission_proba < 0.2 else 'red']
                ax.barh([0], [readmission_proba], color=colors[0], alpha=0.7)
                ax.set_xlim(0, 1)
                ax.set_xlabel('再入院概率')
                ax.set_yticks([])
                ax.axvline(x=0.1, color='green', linestyle='--', alpha=0.5, label='低风险（<10%）')
                ax.axvline(x=0.2, color='orange', linestyle='--', alpha=0.5, label='中风险（10-20%）')
                ax.legend(loc='lower right', fontsize=8)
                st.pyplot(fig)
            
            st.markdown("---")
            st.markdown("### 主要风险因素")
            st.markdown("""
            - **诊断数量**：诊断数量越多，再入院风险和住院天数通常越高
            - **既往就医史**：既往住院/急诊次数多的患者，再入院风险显著更高
            - **用药情况**：使用胰岛素的患者可能需要更长的住院时间
            - **年龄**：老年患者通常住院时间更长，再入院风险更高
            """)
        
        except Exception as e:
            st.error(f"预测过程中出错: {str(e)}")

