import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# 加载训练数据
data = pd.read_csv('training_data.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 训练线性模型
model = LinearRegression()
model.fit(X, y)

# 保存模型文件（可选）
joblib.dump(model, 'linear_model.pkl')

# 保存模型系数到文本文件
with open('linear_model.txt', 'w') as f:
    f.write(f'Coefficients: {model.coef_}\nIntercept: {model.intercept_}\n')