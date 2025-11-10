# AutoML 自动化机器学习平台 - 风控建模智能体

## 📑 目录

- [项目简介](#-项目简介)
- [主要特性](#-主要特性)
- [项目结构](#-项目结构)
- [环境要求](#️-环境要求)
- [安装指南](#-安装指南)
- [配置说明](#️-配置说明)
- [快速开始](#-快速开始)
- [模块说明](#-模块说明)
- [输出说明](#-输出说明)
- [注意事项](#️-注意事项)
- [故障排查](#-故障排查)
- [版本历史](#-版本历史)

## 📋 项目简介

本项目是一个面向风险控制建模的自动化机器学习（AutoML）平台，提供从数据质量分析、数据清洗、特征筛选、模型训练到模型评估的完整建模流程。平台支持大规模数据处理、自动化特征工程、超参数优化以及全面的模型评估报告生成。

## ✨ 主要特性

- **完整建模流程**：涵盖数据质量分析、数据清洗与切分、特征筛选、模型训练、模型评估五个核心模块
- **大数据支持**：支持大文件分批读取，自动判断文件大小并选择最优读取策略
- **自动化特征工程**：支持IV值、PSI值、缺失率、相关性等多维度特征筛选
- **超参数优化**：集成Optuna进行自动化超参数寻优
- **多模型支持**：目前支持LightGBM，可扩展支持XGBoost、逻辑回归等
- **全面评估报告**：自动生成数据质量报告和模型评估报告（Excel格式）
- **模型导出**：支持PKL和PMML两种格式的模型导出
- **分数转换**：支持将预测概率转换为标准评分卡分数
- **可配置化**：通过JSON配置文件灵活控制各模块的运行参数

## 📁 项目结构

```
AutoML-codes/
│
├── config/                    # 配置文件目录
│   ├── data_analysis_config.json        # 数据质量分析配置
│   ├── data_clean_split_config.json     # 数据清洗与切分配置
│   ├── feature_select_config.json       # 特征筛选配置
│   ├── model_train_config.json          # 模型训练配置
│   └── model_eval_config.json           # 模型评估配置
│
├── data/                      # 数据文件目录
│   ├── data.csv              # 原始数据文件
│   └── feature_dict.csv      # 特征字典文件（特征中英文映射）
│
├── outputs/                   # 中间输出文件目录（会被git忽略）
│   ├── data_analysis/        # 数据质量分析输出
│   │   ├── report/           # 数据质量报告
│   │   └── outputs/          # 其他输出文件
│   ├── data_clean_split/     # 数据清洗与切分输出
│   │   ├── data/             # 清洗后的数据（train.csv, test.csv, psi.csv）
│   │   ├── report/           # 清洗报告
│   │   └── outputs/          # 其他输出文件
│   ├── feature_select/       # 特征筛选输出
│   │   └── outputs/          # 筛选后的特征列表等
│   ├── model_train/          # 模型训练输出
│   │   ├── model/            # 模型文件（.pkl, .pmml）
│   │   └── outputs/          # 特征重要性、模型参数等
│   └── model_eval/           # 模型评估输出
│       ├── data/             # 评估数据（含分数）
│       ├── report/           # 评估报告
│       └── outputs/          # 其他输出文件
│
├── results/                   # 结果展示文件目录（JSON格式）
├── logs/                      # 日志文件目录
├── startup/                   # 启动检查文件目录
│
├── data_analysis.py          # 数据质量分析主程序
├── data_clean_split.py       # 数据清洗与切分主程序
├── feature_select.py         # 特征筛选主程序
├── model_train.py            # 模型训练主程序
├── model_eval.py             # 模型评估主程序
├── README.md                 # 项目说明文档
├── requirements.txt          # Python依赖包列表
├── .gitignore                # Git忽略文件配置
│
└── utils/                     # 工具函数目录
    ├── config.py             # 内部配置工具（用于生成JSON配置）
    ├── data_utils.py         # 数据读取工具
    ├── utils_data_quality.py # 数据质量分析工具
    ├── utils_data_clear_split.py # 数据清洗与切分工具
    ├── utils_feature_select.py   # 特征筛选工具
    ├── utils_model_train.py      # 模型训练工具
    ├── utils_model_evaluation.py # 模型评估工具
    ├── utils_report.py           # 报告生成工具
    ├── retrain_code_generator.py # 重训练代码生成器
    ├── utils1.py                 # 通用工具函数
    └── pyce1.py                  # 特征工程工具
```

## 🛠️ 环境要求

### Python 版本
- Python 3.7+（推荐 Python 3.8+）

### 依赖库

项目使用 `requirements.txt` 管理所有依赖包，包含完整的环境配置。主要依赖包括：

- **数据处理**：pandas, numpy
- **机器学习**：lightgbm, scikit-learn, xgboost, catboost
- **超参数优化**：optuna, hyperopt
- **特征工程**：sklearn-pandas
- **模型导出**：sklearn2pmml, pypmml
- **数据可视化**：matplotlib, seaborn, plotly
- **Excel处理**：xlwt, xlrd, openpyxl
- **其他工具**：scipy, statsmodels, python-dateutil 等

## 📦 安装指南

### 1. 克隆项目

```bash
git clone <repository-url>
cd AutoML-codes
```

### 2. 安装依赖

使用项目提供的 `requirements.txt` 安装完整环境：

```bash
pip install -r requirements.txt
```

**注意**：`requirements.txt` 包含了完整的环境依赖，安装可能需要一些时间。如果遇到依赖冲突，建议使用虚拟环境：

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境（Windows）
venv\Scripts\activate

# 激活虚拟环境（Linux/Mac）
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 准备数据

将数据文件放置在 `data/` 目录下：
- `data.csv`：原始数据文件
- `feature_dict.csv`：特征字典文件（可选，如无只含有表头即可）

### 4. 配置项目

项目配置文件位于 `config/` 目录下，包含以下配置文件：
- `data_analysis_config.json` - 数据质量分析配置
- `data_clean_split_config.json` - 数据清洗与切分配置
- `feature_select_config.json` - 特征筛选配置
- `model_train_config.json` - 模型训练配置
- `model_eval_config.json` - 模型评估配置

**注意**：请根据实际数据路径和需求修改配置文件中的参数。

## ⚙️ 配置说明
项目使用JSON配置文件管理各模块的运行参数。配置文件位于 `config/` 目录下，每个模块对应一个配置文件。

## 🚀 快速开始

### 完整流程示例

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

2. **准备数据**
   - 将数据文件 `data.csv` 放入 `data/` 目录
   - 可选：准备特征字典文件 `feature_dict.csv`

3. **配置项目**
   - 根据实际数据路径和需求修改 `config/` 目录下的配置文件
   - 主要修改项：数据路径、输出路径、字段名、时间范围等

4. **运行完整流程**
   ```bash
   # 1. 数据质量分析
   python data_analysis.py -conf config/data_analysis_config.json
   
   # 2. 数据清洗与切分
   python data_clean_split.py -conf config/data_clean_split_config.json
   
   # 3. 特征筛选
   python feature_select.py -conf config/feature_select_config.json
   
   # 4. 模型训练
   python model_train.py -conf config/model_train_config.json
   
   # 5. 模型评估
   python model_eval.py -conf config/model_eval_config.json
   ```

5. **查看结果**
   - 数据质量报告：`outputs/data_analysis/report/data_quality_report.xls`
   - 模型文件：`outputs/model_train/model/model.pkl`
   - 模型评估报告：`outputs/model_eval/report/model_evaluation_report.xlsx`

## 📊 模块说明

### 1. 数据质量分析 (data_analysis.py)

**功能**：
- 数据基本信息统计（行数、列数、数据类型等）
- 缺失值分析（NaN和特定缺失标记）
- 数值型变量描述性统计
- 类别型变量频数统计
- 目标变量分布分析
- 渠道分布分析

**输出**：
- 数据质量报告（Excel格式）：`outputs/data_analysis/report/data_quality_report.xls`
- 结果JSON文件：`results/data_analysis.json`

### 2. 数据清洗与切分 (data_clean_split.py)

**功能**：
- 数据清洗（处理缺失值、异常值等）
- 数据切分（训练集、测试集、PSI数据集）
- 支持按时间切分或随机切分
- 目标变量统计报告

**输出**：
- 清洗后的数据：`outputs/data_clean_split/data/cleaned_data.csv`
- 训练集：`outputs/data_clean_split/data/train.csv`
- 测试集：`outputs/data_clean_split/data/test.csv`
- PSI数据集：`outputs/data_clean_split/data/psi.csv`
- 目标变量统计报告：`outputs/data_clean_split/report/target_stats_report.csv`
- 结果JSON文件：`results/data_clean_split.json`

### 3. 特征筛选 (feature_select.py)

**功能**：
- IV值计算与筛选
- PSI值计算与筛选
- 缺失率筛选
- 相关性分析
- Null Importance筛选（可选）
- 支持人工指定特征列表

**输出**：
- 筛选后的特征列表：`outputs/feature_select/outputs/use_fea.json`
- 特征分析结果：`outputs/feature_select/outputs/`
- 结果JSON文件：`results/feature_select.json`

### 4. 模型训练 (model_train.py)

**功能**：
- LightGBM模型训练
- 超参数优化（Optuna）
- 特征重要性计算
- 模型保存（PKL和PMML格式）
- 模型参数记录

**输出**：
- 模型文件：`outputs/model_train/model/model.pkl`
- PMML模型：`outputs/model_train/model/model.pmml`
- 特征重要性：`outputs/model_train/outputs/feature_imp.csv`
- 模型参数：`outputs/model_train/outputs/model_params.json`
- 最终模型特征：`outputs/model_train/outputs/final_model_fea.json`
- 结果JSON文件：`results/model_train.json`

### 5. 模型评估 (model_eval.py)

**功能**：
- 模型性能评估（AUC、KS等指标）
- 分数转换（概率转评分卡分数）
- 分群评估（按渠道等维度）
- Lift图分析
- PSI稳定性分析
- 综合评估报告生成

**输出**：
- 评估数据（含分数）：`outputs/model_eval/data/`
- 模型评估报告（Excel格式）：`outputs/model_eval/report/model_evaluation_report.xlsx`
- 结果JSON文件：`results/model_eval.json`

## 📈 输出说明

### 目录结构

- **outputs/**：中间输出文件目录，包含各模块的详细输出
- **results/**：结果展示文件目录，包含各模块的JSON格式结果摘要
- **startup/**：启动检查文件目录，用于跟踪各模块的运行状态
- **logs/**：日志文件目录（如需要）

### 主要输出文件

1. **数据质量报告**：`outputs/data_analysis/report/data_quality_report.xls`
2. **清洗后的数据**：`outputs/data_clean_split/data/`
3. **筛选后的特征列表**：`outputs/feature_select/outputs/use_fea.json`
4. **训练好的模型**：`outputs/model_train/model/model.pkl` 和 `model.pmml`
5. **模型评估报告**：`outputs/model_eval/report/model_evaluation_report.xlsx`

## ⚠️ 注意事项

1. **数据格式要求**：
   - 数据文件必须是CSV格式
   - 必须包含主键列（id_col）、时间列（date_col）、标签列（label）、渠道列（channel_col）
   - 缺失值可以使用NaN或配置中指定的缺失标记

2. **内存管理**：
   - 项目支持大文件分批读取（默认阈值500MB）
   - 对于超大文件，会自动启用分批读取模式

3. **配置一致性**：
   - 确保配置文件中的字段名与数据文件中的字段名一致
   - 时间范围配置要与数据文件中的实际时间范围匹配

4. **执行顺序**：
   - 建议按照：数据质量分析 → 数据清洗与切分 → 特征筛选 → 模型训练 → 模型评估的顺序执行
   - 每个模块的输出作为下一个模块的输入

5. **特征字典文件**：
   - `feature_dict.csv` 文件用于特征中英文映射，如无此文件，可只创建表头

6. **模型格式**：
   - PKL格式：用于Python环境下的模型加载
   - PMML格式：用于跨平台模型部署

## 🔧 故障排查

### 常见问题

1. **文件读取失败**
   - 检查数据文件路径是否正确
   - 检查数据文件是否存在
   - 检查必需的字段是否存在

2. **内存不足**
   - 减小 `chunk_size` 参数
   - 减少特征数量
   - 使用更大的内存机器

3. **模型训练失败**
   - 检查特征列表是否为空
   - 检查目标变量分布是否合理
   - 检查超参数范围是否合理

4. **评估报告生成失败**
   - 检查模型文件是否存在
   - 检查评估数据是否存在
   - 检查报告输出目录权限

## 📝 版本历史

### v1.0.0
- 初始版本发布
- 实现数据质量分析、数据清洗、特征筛选、模型训练、模型评估五个核心模块
- 支持LightGBM模型训练
- 支持超参数优化
- 支持模型评估报告生成

## 📄 许可证

[待添加许可证信息]

## 👥 贡献者

[待添加贡献者信息]

## 📮 联系方式

[待添加联系方式]

## 🙏 致谢

感谢所有为这个项目做出贡献的开发者和用户。

---

**注意**：本项目专为风险控制建模场景设计，使用时请根据实际业务需求调整配置参数。建议在生产环境使用前进行充分的测试和验证。
