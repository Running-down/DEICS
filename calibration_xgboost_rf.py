import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
import pandas as pd

# 使用rpy2中的robjects加载R中的模型文件
r = robjects.r

# 替换成你的RDS文件路径
rds_file_path_rf = r"D:\AOngoingWork\DICS\codeAdata\calibration\model_rf.rds"
rds_file_path_xgb = r"D:\AOngoingWork\DICS\codeAdata\calibration\model_xgbtree.rds"

# 使用rpy2的readRDS()函数加载RDS文件
rf_model = robjects.r['readRDS'](rds_file_path_rf)
xgb_model = robjects.r['readRDS'](rds_file_path_xgb)

# 将R中的模型转换为Python对象
pandas2ri.activate()
rf_model_py = pandas2ri.rpy2py(rf_model)
xgb_model_py = pandas2ri.rpy2py(xgb_model)

# 读取数据
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

# 假设数据中的标签列名为 'infection'，你需要根据实际情况替换为正确的列名
X_train = data_train.drop('infection', axis=1)
y_train = data_train['infection']
X_test = data_test.drop('infection', axis=1)
y_test = data_test['infection']

print("Training Data - Class Distribution:")
print(y_train.value_counts())

print("\nTesting Data - Class Distribution:")
print(y_test.value_counts())

print("Training Data - Data Types:")
print(X_train.dtypes)

print("\nTesting Data - Data Types:")
print(X_test.dtypes)

## 获取训练集的预测概率
prob_rf_train = r.predict(rf_model, X_train, type='prob')
prob_rf_train_df = pandas2ri.rpy2py_dataframe(prob_rf_train)
prob_rf_train_yes = prob_rf_train_df['yes']

prob_xgb_train = r.predict(xgb_model, X_train, type='prob')
prob_xgb_train_df = pandas2ri.rpy2py_dataframe(prob_xgb_train)
prob_xgb_train_yes = prob_xgb_train_df['yes']

# Calculate calibration curve for training set
prob_pred_rf_train, prob_true_rf_train = calibration_curve(y_train, prob_rf_train_yes, n_bins=3)
prob_pred_xgb_train, prob_true_xgb_train = calibration_curve(y_train, prob_xgb_train_yes, n_bins=3)

## 获取测试集的预测概率
prob_rf_test = r.predict(rf_model, X_test, type='prob')
prob_rf_test_df = pandas2ri.rpy2py_dataframe(prob_rf_test)
prob_rf_test_yes = prob_rf_test_df['yes']

prob_xgb_test = r.predict(xgb_model, X_test, type='prob')
prob_xgb_test_df = pandas2ri.rpy2py_dataframe(prob_xgb_test)
prob_xgb_test_yes = prob_xgb_test_df['yes']

# Calculate calibration curve for testing set
prob_pred_rf_test, prob_true_rf_test = calibration_curve(y_test, prob_rf_test_yes, n_bins=3)
prob_pred_xgb_test, prob_true_xgb_test = calibration_curve(y_test, prob_xgb_test_yes, n_bins=3)

# Plot calibration curve for training set
plt.figure(figsize=(8, 8))
plt.plot(prob_pred_rf_train, prob_true_rf_train, marker='o', label='Random Forest (Train)', linestyle='--', color='blue')
plt.plot(prob_pred_xgb_train, prob_true_xgb_train, marker='o', label='XGBoost (Train)', linestyle='--', color='green')
plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly Calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve (Train)')
plt.legend()

# Save the figure
plt.savefig('calibration_curve_train.png')  # Change the filename as needed

plt.show()

# Plot calibration curve for testing set
plt.figure(figsize=(8, 8))
plt.plot(prob_pred_rf_test, prob_true_rf_test, marker='o', label='Random Forest (Test)', linestyle='--', color='blue')
plt.plot(prob_pred_xgb_test, prob_true_xgb_test, marker='o', label='XGBoost (Test)', linestyle='--', color='green')
plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfectly Calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve (Test)')
plt.legend()

# Save the figure
plt.savefig('calibration_curve_test.png')  # Change the filename as needed

plt.show()




