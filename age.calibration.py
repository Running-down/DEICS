import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import pandas as pd

# 使用rpy2加载R中的模型文件
r = robjects.r

# 替换为你的XGBoost RDS文件路径
rds_file_path_xgb = r"D:\AOngoingWork\DICS\codeAdata\calibration\model_xgbtree.rds"

# 加载XGBoost模型
xgb_model = r['readRDS'](rds_file_path_xgb)

# 激活pandas2ri
pandas2ri.activate()

# 读取数据
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

# 假设数据中的标签列名为 'infection'
X_train = data_train.drop('infection', axis=1)
y_train = data_train['infection']
X_test = data_test.drop('infection', axis=1)
y_test = data_test['infection']

# 年龄阈值
age_threshold = 65

# 按年龄分组
grouped_data_train = data_train.copy()
grouped_data_test = data_test.copy()

# 添加年龄分组标志
grouped_data_train['age_group'] = grouped_data_train['age'] > age_threshold
grouped_data_test['age_group'] = grouped_data_test['age'] > age_threshold

# 定义颜色和标签
age_groups = [True, False]  # True表示 > 65, False表示 <= 65
age_labels = {True: "Age > 65", False: "Age ≤ 65"}
colors = {True: "blue", False: "green"}

# 绘制训练集校准曲线
plt.figure(figsize=(8, 8))

for age_group in age_groups:
    train_subset = grouped_data_train[grouped_data_train['age_group'] == age_group]
    if train_subset.empty:
        print(f"Skipping age_group={age_group} (no data in training set).")
        continue

    # 分离特征和标签
    X_train_subset = train_subset.drop(['infection', 'age_group'], axis=1)
    y_train_subset = train_subset['infection']

    # 获取训练集预测概率
    prob_xgb_train = r.predict(xgb_model, X_train_subset, type='prob')
    prob_xgb_train_df = pandas2ri.rpy2py_dataframe(prob_xgb_train)
    prob_xgb_train_yes = prob_xgb_train_df['yes']

    # 计算校准曲线
    prob_pred_xgb_train, prob_true_xgb_train = calibration_curve(y_train_subset, prob_xgb_train_yes, n_bins=2)

    # 绘制校准曲线
    plt.plot(prob_pred_xgb_train, prob_true_xgb_train, marker='o', label=age_labels[age_group], linestyle='--', color=colors[age_group])

plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly Calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve (Train)')
plt.legend()
plt.savefig('calibration_curve_train_combined.png')
plt.show()

# 绘制测试集校准曲线
plt.figure(figsize=(8, 8))

for age_group in age_groups:
    test_subset = grouped_data_test[grouped_data_test['age_group'] == age_group]
    if test_subset.empty:
        print(f"Skipping age_group={age_group} (no data in testing set).")
        continue

    # 分离特征和标签
    X_test_subset = test_subset.drop(['infection', 'age_group'], axis=1)
    y_test_subset = test_subset['infection']

    # 获取测试集预测概率
    prob_xgb_test = r.predict(xgb_model, X_test_subset, type='prob')
    prob_xgb_test_df = pandas2ri.rpy2py_dataframe(prob_xgb_test)
    prob_xgb_test_yes = prob_xgb_test_df['yes']

    # 计算校准曲线
    prob_pred_xgb_test, prob_true_xgb_test = calibration_curve(y_test_subset, prob_xgb_test_yes, n_bins=2)

    # 绘制校准曲线
    plt.plot(prob_pred_xgb_test, prob_true_xgb_test, marker='o', label=age_labels[age_group], linestyle='--', color=colors[age_group])

plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly Calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve (Test)')
plt.legend()
plt.savefig('calibration_curve_test_combined.png')
plt.show()
