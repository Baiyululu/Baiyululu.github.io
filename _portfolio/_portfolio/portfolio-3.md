---
title: "糖尿病预测分析项目"
collection: portfolio
type: "Machine Learning"
permalink: /portfolio/diabetes-prediction
date: 2026-01-14
excerpt: "基于机器学习的糖尿病风险评估与预测"
header:
  teaser: /images/portfolio/diabetes-prediction/diabetes-header.jpg
tags:
  - 机器学习
  - 数据分析
  - 糖尿病预测
  - 医疗健康
tech_stack:
  - name: Python
  - name: Scikit-learn
  - name: XGBoost
  - name: Pandas
  - name: Matplotlib
---

## 项目背景

糖尿病是一种常见的慢性疾病，早期诊断和干预对于控制病情至关重要。本项目基于公开的糖尿病预测数据集，通过数据分析和机器学习模型构建，实现糖尿病风险的准确预测。项目涵盖数据预处理、统计分析、模型构建与评估等完整流程，最终构建了高精度的糖尿病预测模型。

## 数据准备

### 数据集介绍

- **数据集规模**：100,000 行 × 9 列
- **特征变量**：性别、年龄、高血压、心脏病、吸烟史、BMI、HbA1c水平、血糖水平
- **目标变量**：糖尿病（0/1）

### 数据预处理

```python
# 缺失值检查
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100

# 重复值处理
duplicates = df.duplicated().sum()
df = df.drop_duplicates()

# 异常值处理（以BMI为例）
Q1 = df['bmi'].quantile(0.25)
Q3 = df['bmi'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_clean = df.copy()  # 保留所有数据
```

## 统计分析

### 描述性统计

- **糖尿病患者比例**：8.82%（8,482人）
- **非糖尿病患者比例**：91.18%（87,664人）

### 基线特征对比

| 特征 | 糖尿病患者 | 非糖尿病患者 |
|------|------------|--------------|
| 年龄 | 60.93 ± 14.55 | 39.94 ± 22.23 |
| BMI | 32.00 ± 7.56 | 26.87 ± 6.51 |
| HbA1c水平 | 6.93 ± 1.08 | 5.40 ± 0.97 |
| 血糖水平 | 194.03 ± 58.63 | 132.82 ± 34.24 |
| 高血压比例 | 24.6% | 6.1% |
| 心脏病比例 | 14.9% | 3.0% |

### 统计检验

```python
# 独立样本t检验（年龄）
t_stat, p_value = stats.ttest_ind(diabetic['age'], non_diabetic['age'])

# BMI的t检验
t_stat_bmi, p_value_bmi = stats.ttest_ind(diabetic['bmi'], non_diabetic['bmi'])

# 卡方检验（高血压）
contingency_table = pd.crosstab(df_clean['diabetes'], df_clean['hypertension'])
chi2, p_chi, dof, expected = chi2_contingency(contingency_table)
```

### 相关性分析

各变量与糖尿病的相关性：
- 血糖水平: 0.424
- HbA1c水平: 0.406
- 年龄: 0.265
- BMI: 0.215
- 高血压: 0.196
- 心脏病: 0.171

## 模型构建

### 数据准备

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
```

### 模型训练

```python
# 建立多个模型
models = {
    '逻辑回归': LogisticRegression(random_state=42, max_iter=1000),
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
}

# 训练模型
trained_models = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
```

## 模型评估

### 性能对比

| 模型 | 准确率 | 精确率 | 召回率 | F1分数 | AUC |
|------|--------|--------|--------|--------|-----|
| 逻辑回归 | 0.9594 | 0.8670 | 0.6380 | 0.7351 | 0.9594 |
| 随机森林 | 0.9695 | 0.9490 | 0.6910 | 0.7997 | 0.9582 |
| XGBoost | 0.9706 | 0.9601 | 0.6952 | 0.8064 | 0.9762 |

### 分类报告

```python
print(classification_report(y_test, y_pred, target_names=['非糖尿病', '糖尿病']))
```

## 可视化分析

### 特征重要性

```python
# 随机森林特征重要性
importances = trained_models['随机森林'].feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'特征': feature_names, '重要性': importances})
feature_importance_df = feature_importance_df.sort_values('重要性', ascending=False)

# 绘制特征重要性图
plt.figure(figsize=(10, 6))
sns.barplot(x='重要性', y='特征', data=feature_importance_df)
plt.title('随机森林特征重要性')
```

![特征重要性](/images/portfolio/diabetes-prediction/feature-importance.png)

**分析结论**：血糖水平和HbA1c水平是糖尿病预测最重要的两个特征，年龄和BMI也具有一定的预测价值。

### ROC曲线

```python
# 绘制ROC曲线
plt.figure(figsize=(10, 8))
for name, model in trained_models.items():
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('ROC曲线')
plt.legend(loc='lower right')
```

![ROC曲线](/images/portfolio/diabetes-prediction/roc-curve.png)

**分析结论**：XGBoost模型表现最佳，AUC达到0.9762，具有优秀的分类性能。

## 结论

1. **模型性能**：XGBoost模型表现最佳，准确率达97.06%，AUC为0.9762
2. **关键特征**：血糖水平、HbA1c水平、年龄是糖尿病预测的最重要特征
3. **实际应用**：该模型可用于临床辅助诊断，帮助医生早期识别糖尿病高风险人群

## 图片资源清单

请在您的网站项目中创建文件夹：`images/portfolio/diabetes-prediction/`

请回到您的JupyterNotebook，手动保存以下位置的图片，并重命名为指定文件名，上传到该文件夹：

| 原图位置(Notebook章节/代码块)|建议文件名|用途| 
| :--- | :--- | :--- |
| 项目封面图|`diabetes-header.jpg` | 页面头图 |
| 3.3 节 相关性分析|`correlation-matrix.png` | 正文插图 |
| 5.2 节 特征重要性|`feature-importance.png` | 正文插图 |
| 5.2 节 ROC曲线|`roc-curve.png` | 正文插图 |
| 5.2 节 混淆矩阵|`confusion-matrix.png` | 正文插图 |

## 行动指引

1. **创建文件夹**：在您的网站项目中创建`images/portfolio/diabetes-prediction/`文件夹
2. **保存图片**：从JupyterNotebook中保存上述图片，并重命名为指定文件名
3. **上传文件**：将生成的`.md`文件上传到`_portfolio/`文件夹
4. **验证效果**：启动本地服务器，访问`http://localhost:4000/portfolio/diabetes-prediction`查看效果
5. **部署上线**：将所有文件推送到GitHub Pages或其他静态网站托管平台
