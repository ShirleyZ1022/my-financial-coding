import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# 存储日期和价格的列表
dates = []
prices = []


def get_data(file_path):
    global dates, prices
    dates = []
    prices = []
    with open(file_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # 跳过表头行
        for row in csv_reader:
            # 处理日期：假设格式为“月/日/年”，提取“日”（索引为1）转为整数
            date_parts = row[0].split('/')
            if len(date_parts) == 3:
                dates.append(int(date_parts[1]))
            # 处理收盘价：去除$后转为浮点数
            price_str = row[1].replace('$', '')
            prices.append(float(price_str))

# 预测函数


def predict_prices(dates_list, prices_list, x):
    # 将日期数据 reshape 为 sklearn 要求的格式
    dates_reshaped = np.reshape(dates_list, (len(dates_list), 1))

    # 优化 RBF 模型参数：增大 C 让模型更拟合，调整 gamma 捕捉局部波动
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf.fit(dates_reshaped, prices_list)

    # 线性模型
    svr_linear = SVR(kernel='linear', C=1e3)
    svr_linear.fit(dates_reshaped, prices_list)

    # 多项式模型
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_poly.fit(dates_reshaped, prices_list)

    return svr_rbf.predict(np.array([[x]])), svr_linear.predict(np.array([[x]])), svr_poly.predict(np.array([[x]]))


# 读取数据（替换为你的 CSV 文件路径）
file_path = r'C:\Users\DELL\Desktop\HistoricalData_1758960835442.csv'
get_data(file_path)

# 生成更密集的日期点用于预测，让曲线更平滑
dates_pred = np.linspace(min(dates), max(dates), 100).reshape(-1, 1)

# 训练模型（用原始日期数据）
dates_reshaped = np.reshape(dates, (len(dates), 1))
svr_rbf = SVR(kernel='rbf', C=1000, gamma=0.1)
svr_rbf.fit(dates_reshaped, prices)

svr_linear = SVR(kernel='linear', C=1000)
svr_linear.fit(dates_reshaped, prices)

svr_poly = SVR(kernel='poly', C=1000, degree=3)
svr_poly.fit(dates_reshaped, prices)

# 用密集日期点预测
price_pred_rbf = svr_rbf.predict(dates_pred)
price_pred_linear = svr_linear.predict(dates_pred)
price_pred_poly = svr_poly.predict(dates_pred)

# 绘图
plt.scatter(dates, prices, color='black', label='Data')
plt.plot(dates_pred, price_pred_rbf, color='red', label='RBF model')
plt.plot(dates_pred, price_pred_linear, color='green', label='Linear model')
plt.plot(dates_pred, price_pred_poly, color='blue', label='Polynomial model')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Support Vector Regression for Stock Price')
plt.legend()
plt.show()

# 示例：预测某个日期的价格（这里以 dates 中的最大值 + 1 为例）
pred_date = max(dates) + 1
rbf_pred, linear_pred, poly_pred = predict_prices(dates, prices, pred_date)
print(f"对于日期 {pred_date}：")
print(f"RBF 模型预测价格：{rbf_pred[0]}")
print(f"线性模型预测价格：{linear_pred[0]}")
print(f"多项式模型预测价格：{poly_pred[0]}")
