---
title: "糖尿病风险预测模型构建与数据分析"
collection: portfolio
type: "Data Analysis"
permalink: /portfolio/diabetes-risk-prediction
date: 2026-01-18
excerpt: "基于10万条临床记录的糖尿病风险预测模型构建与分析"
header:
  teaser: /images/portfolio/diabetes-risk-prediction/diabetes_visualization.png
tags:
  - 糖尿病预测
  - 机器学习
  - 数据分析
  - 医疗健康
techstack:
  - name: Python
  - name: Scikit-learn
  - name: XGBoost
  - name: Pandas
  - name: Matplotlib
---

## 项目背景

糖尿病是全球范围内严重威胁人类健康的慢性疾病，早期筛查和风险预测对于疾病防控至关重要。本项目基于10万条临床记录，通过数据分析和机器学习方法，构建糖尿病风险预测模型，识别关键影响因素，为临床决策提供支持。

## 数据概览

- **数据集规模**：100,000条临床记录，9个特征变量
- **核心特征**：年龄、性别、BMI、糖化血红蛋白（HbA1c）、血糖水平、高血压病史、心脏病史、吸烟史
- **目标变量**：糖尿病诊断结果（0=非糖尿病，1=糖尿病）

## 核心实现

### 数据预处理

```python
# 处理重复值
df = df.drop_duplicates()

# 异常值处理（以BMI为例）
Q1 = df['bmi'].quantile(0.25)
Q3 = df['bmi'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 保留所有数据用于后续分析
df_clean = df.copy()
```

### 统计分析

```python
# 糖尿病患者 vs 非糖尿病患者基线特征对比
diabetic = df_clean[df_clean['diabetes'] == 1]
non_diabetic = df_clean[df_clean['diabetes'] == 0]

# 相关性分析
correlation_matrix = df_clean[numeric_cols].corr()
```

### 模型构建

```python
# 编码分类变量
for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 建立多个模型
models = {
    '逻辑回归': LogisticRegression(random_state=42, max_iter=1000),
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
}

# 训练模型
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
```

## 分析结果

### 基线特征对比

| 特征 | 糖尿病患者 | 非糖尿病患者 |
| :--- | :--- | :--- |
| 年龄 | 60.93 ± 14.55 | 39.94 ± 22.23 |
| BMI | 32.00 ± 7.56 | 26.87 ± 6.51 |
| HbA1c | 6.93 ± 1.08 | 5.40 ± 0.97 |
| 血糖水平 | 194.03 ± 58.63 | 132.82 ± 34.24 |
| 高血压比例 | 24.6% | 6.1% |
| 心脏病比例 | 14.9% | 3.0% |

### 相关性分析

![相关性矩阵](/images/portfolio/diabetes-risk-prediction/correlation_heatmap.png)

各变量与糖尿病的相关性排序：
1. 血糖水平: 0.424
2. HbA1c水平: 0.406
3. 年龄: 0.265
4. BMI: 0.215
5. 高血压: 0.196
6. 心脏病: 0.171

### 模型性能对比

| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 逻辑回归 | 0.9594 | 0.8670 | 0.6380 | 0.7351 | 0.9594 |
| 随机森林 | 0.9695 | 0.9490 | 0.6910 | 0.7997 | 0.9582 |
| XGBoost | 0.9706 | 0.9601 | 0.6952 | 0.8064 | 0.9762 |

### ROC曲线对比

![ROC曲线](/images/portfolio/diabetes-risk-prediction/roc_curve.png)

## 结论

本研究通过对10万条临床记录的深入分析，得出以下主要结论：

1. **关键影响因素**：血糖水平、HbA1c、年龄和BMI是糖尿病的主要预测因子。
2. **模型性能**：XGBoost模型表现最优，AUC达到0.9762，具有良好的临床应用潜力。
3. **临床意义**：该模型可用于糖尿病早期筛查，帮助医生识别高风险人群，制定个性化预防方案。