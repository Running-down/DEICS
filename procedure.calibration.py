import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 使用rpy2加载R中的模型文件
r = robjects.r

# 替换为你的XGBoost RDS文件路径
rds_file_path_xgb = r"D:\\AOngoingWork\\DICS\\codeAdata\\calibration\\model_xgbtree.rds"

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

# 按Procedure_name分组
procedure_train_groups = data_train['Procedure_name'].unique()  # 获取所有唯一的Procedure_name
procedure_test_groups = data_test['Procedure_name'].unique()  # 获取所有唯一的Procedure_name

# 按照 Precedure_name 分组
procedure_names = ['1', '2', '3', '4', '5', '6']  # 对应 Procedure_name 的取值（确保字符串类型）

# 定义颜色和标签
procedure_labels = {
    '1': "Isolated CABG",
    '2': "AVRorMVRorTVR",
    '3': "AVR+MVR",
    '4': "Valve+CABG surgery",
    '5': "Thoracic surgery",
    '6': "Others"
}

colors = {
    '1': "blue",
    '2': "green",
    '3': "red",
    '4': "purple",
    '5': "orange",
    '6': "cyan"
}

# 绘制训练集校准曲线
plt.figure(figsize=(8, 8))

# 循环遍历每个Procedure_name，绘制训练集校准曲线
for procedure in procedure_train_groups:
    # 按照Procedure_name分组数据
    train_subset = data_train[data_train['Procedure_name'] == procedure]
    if train_subset.empty:
        print(f"Skipping procedure={procedure} (no data in training set).")
        continue

    # 分离特征和标签，并保留Procedure_name列
    X_train_subset = train_subset.drop(['infection'], axis=1)
    y_train_subset = train_subset['infection']

    # 获取训练集预测概率
    X_train_r = pandas2ri.py2rpy(X_train_subset)
    prob_xgb_train = r.predict(xgb_model, X_train_r, type='prob')

    # 转换预测结果为pandas DataFrame
    prob_xgb_train_df = pandas2ri.rpy2py_dataframe(prob_xgb_train)

    # 假设概率列名为 'yes'
    if 'yes' in prob_xgb_train_df.columns:
        prob_xgb_train_yes = prob_xgb_train_df['yes']
    else:
        print(f"Warning: Column 'yes' not found in the prediction result for procedure {procedure}.")
        continue

    # 计算校准曲线
    prob_pred_xgb_train, prob_true_xgb_train = calibration_curve(y_train_subset, prob_xgb_train_yes, n_bins=2)

    # 绘制训练集校准曲线
    plt.plot(prob_pred_xgb_train, prob_true_xgb_train, marker='o', label=f'Train - {procedure_labels[str(procedure)]}', linestyle='--', color=colors[str(procedure)])

# 绘制完毕后，设置图形内容
plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly Calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve (Train)')
plt.legend()
plt.savefig('calibration_curve_train_procedure.png')
plt.show()

# 绘制验证集校准曲线
plt.figure(figsize=(8, 8))

# 循环遍历每个Procedure_name，绘制验证集校准曲线
for procedure in procedure_test_groups:
    # 按照Procedure_name分组数据
    test_subset = data_test[data_test['Procedure_name'] == procedure]
    if test_subset.empty:
        print(f"Skipping procedure={procedure} (no data in test set).")
        continue

    # 分离特征和标签，并保留Procedure_name列
    X_test_subset = test_subset.drop(['infection'], axis=1)
    y_test_subset = test_subset['infection']

    # 获取验证集预测概率
    X_test_r = pandas2ri.py2rpy(X_test_subset)
    prob_xgb_test = r.predict(xgb_model, X_test_r, type='prob')

    # 转换预测结果为pandas DataFrame
    prob_xgb_test_df = pandas2ri.rpy2py_dataframe(prob_xgb_test)

    # 假设概率列名为 'yes'
    if 'yes' in prob_xgb_test_df.columns:
        prob_xgb_test_yes = prob_xgb_test_df['yes']
    else:
        print(f"Warning: Column 'yes' not found in the prediction result for procedure {procedure}.")
        continue

    # 计算校准曲线
    prob_pred_xgb_test, prob_true_xgb_test = calibration_curve(y_test_subset, prob_xgb_test_yes, n_bins=2)

   # 绘制验证集校准曲线
    plt.plot(prob_pred_xgb_test, prob_true_xgb_test, marker='o', label=f'Test - {procedure_labels[str(procedure)]}', linestyle='-', color=colors[str(procedure)])

plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly Calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve (Test)')
plt.legend()
plt.savefig('calibration_curve_test_procedure.png')
plt.show()
