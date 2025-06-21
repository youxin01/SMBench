import pandas as pd
import numpy as np
from typing import Callable
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, RANSACRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, classification_report
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
from scipy.interpolate import UnivariateSpline
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pmdarima import auto_arima
import statsmodels.api as sm
from sklearn.svm import SVC

def get_header():
    header ="""import pandas as pd
import numpy as np
from typing import Callable
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, RANSACRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, classification_report
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
from scipy.interpolate import UnivariateSpline
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pmdarima import auto_arima
import statsmodels.api as sm
from sklearn.svm import SVC"""
    return header

# 数据集划分
def split_data(df: pd.DataFrame, target_column: str, test_size=0.2, random_state=None):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# 训练XGBoost模型预测连续值
def train_xgboost_regressor(
    data: pd.DataFrame,
    target: str,
    new_data: pd.DataFrame,
    test_size: float = 0.2,
    params: dict = None,
    num_boost_round: int = 100,
    with_label: bool = False
):
    X_train, X_test, y_train, y_test = split_data(data, target,test_size)
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'eta': 0.3,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    model = xgb.train(params, dtrain, num_boost_round=num_boost_round)

    y_pred = model.predict(dtest)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"RMSE: {rmse}")

    # 预测新数据
    dnew = xgb.DMatrix(new_data)
    prediction = model.predict(dnew)
    
    if with_label:
        new_data1 = new_data.copy()
        new_data1['prediction'] = prediction
        return new_data1
    else:
        return prediction

# 训练XGBoost模型预测分类 return array([])
def train_xgboost_classifier(
    data: pd.DataFrame,
    target: str,
    new_data: pd.DataFrame,
    test_size: float = 0.2,
    params: dict = None,
    num_boost_round: int = 100,
    with_label: bool = False
):
    X_train, X_test, y_train, y_test = split_data(data, target,test_size)
    
    if params is None:
        params = {
            'objective': 'multi:softprob',  # 多分类概率输出
            'num_class': len(np.unique(y_train)),  # 类别数量
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': ['mlogloss', 'merror']  # 多分类的评估指标
        }
    if params.get('num_class') is None:
        params['num_class'] = len(np.unique(y_train))

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    model = xgb.train(params, dtrain, num_boost_round=num_boost_round)

    y_pred = model.predict(dtest)
    # 根据 y_pred 的维度选择处理方式
    if len(y_pred.shape) == 1:  # 一维数组，二分类问题
        predictions = (y_pred > 0.5).astype(int)
    else:  # 二维数组，多分类问题
        predictions = np.argmax(y_pred, axis=1)

    # 计算准确率
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")

    # 预测新数据
    dnew = xgb.DMatrix(new_data)
    probs = model.predict(dnew)

    # 判断是二分类还是多分类
    if len(probs.shape) == 1:  # 二分类问题
        predictions = (probs > 0.5).astype(int)
    else:  # 多分类问题
        predictions = np.argmax(probs, axis=1)

    # 根据是否有标签返回不同的结果
    if with_label:
        new_data1 = new_data.copy()
        new_data1['predicted_class'] = predictions
        return new_data1
    else:
        return predictions


# LightGBM模型,回归模型
def train_lightgbm_regressor(
    data: pd.DataFrame,
    target: str,
    new_data: pd.DataFrame,
    test_size: float = 0.2,
    params: dict = None,
    num_boost_round: int = 100,
    with_label: bool = False
):
    X_train, X_test, y_train, y_test = split_data(data, target,test_size)
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.3,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)

    model = lgb.train(params, dtrain, num_boost_round=num_boost_round, valid_sets=[dtrain, dtest])

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"RMSE: {rmse}")

    # 预测新数据
    prediction = model.predict(new_data, num_iteration=model.best_iteration)
    
    if with_label:
        new_data1 = new_data.copy()
        new_data1['prediction'] = prediction
        return new_data1
    else:
        return prediction
    
# LightGBM模型,分类模型
def train_lightgbm_classifier(
    data: pd.DataFrame,
    target: str,
    new_data: pd.DataFrame,
    test_size: float = 0.2,
    params: dict = None,
    num_boost_round: int = 100,
    with_label: bool = False
):
    X_train, X_test, y_train, y_test = split_data(data, target,test_size)
    
    if params is None:
        params = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'num_class': len(np.unique(y_train)),
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)

    model = lgb.train(params, dtrain, num_boost_round=num_boost_round, valid_sets=[dtrain, dtest])

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    if len(y_pred.shape) == 1:  # 二分类问题
        predictions = (y_pred > 0.5).astype(int)
    else:  # 多分类问题
        predictions = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")

    # 预测新数据
    probs = model.predict(new_data, num_iteration=model.best_iteration)
    # 判断是二分类还是多分类
    if len(probs.shape) == 1:  # 二分类问题
        predictions = (probs > 0.5).astype(int)
    else:  # 多分类问题
        predictions = np.argmax(probs, axis=1)
    
    if with_label:
        new_data1 = new_data.copy()
        new_data1['predicted_class'] = predictions
        return new_data1
    else:
        return predictions
    
#K-nearst算法，分类模型
def train_knn_classifier(
    data: pd.DataFrame,
    target: str,
    new_data: pd.DataFrame,
    test_size: float = 0.2,
    n_neighbors: int = 3,
    with_label: bool = False
):
    X_train, X_test, y_train, y_test = split_data(data, target,test_size)
    
    # 确保输入数据是 C 连续的 NumPy 数组
    X_train = np.ascontiguousarray(X_train)
    X_test = np.ascontiguousarray(X_test)
    new_data = np.ascontiguousarray(new_data)
    
    # 训练 KNN 模型
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    
    # 在测试集上评估模型
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")
    
    # 预测新数据
    predictions = model.predict(new_data)
    
    if with_label:
        # 将 new_data 转换为 DataFrame
        new_data1 = pd.DataFrame(new_data, columns=[f'feature{i+1}' for i in range(new_data.shape[1])])
        # 添加预测标签列
        new_data1['predicted_class'] = predictions
        return new_data1
    else:
        return predictions

 

# 简单线性拟合
def train_simple_regressor(
    data: pd.DataFrame,
    target: str,
    new_data: pd.DataFrame = None,
    method: str = 'linear',
    test_size: float = 0.2,
    degree: int = 2,
    s: float = None,
    with_label: bool = False
):
    """
    统一拟合函数（支持线性、多项式、样条插值）
    
    参数：
        data : pd.DataFrame, 训练数据（包含特征和target列）
        target : str, 目标列名
        new_data : pd.DataFrame, 待预测的新数据（可选）
        method : str, 拟合方法 ('linear', 'poly', 'spline')
        test_size : float, 测试集比例（0-1）
        degree : int, 多项式阶数（仅对method='poly'有效）
        s : float, 样条平滑参数（仅对method='spline'有效）
        with_label : bool, 是否将预测结果与输入数据合并返回
    
    返回：
        如果 with_label=True: 返回带预测结果的DataFrame
        否则: 返回预测结果数组/DataFrame
    """
    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = split_data(data,target,test_size)
    
    # 选择拟合方法
    if method == 'linear':
        model = LinearRegression()
        model.fit(X_train, y_train)
        predict_func = lambda x: model.predict(np.ascontiguousarray(x))
        
    elif method == 'poly':
        model = make_pipeline(
            PolynomialFeatures(degree=degree),
            LinearRegression()
        )
        model.fit(X_train, y_train)
        predict_func = lambda x: model.predict(np.ascontiguousarray(x))
        
    elif method == 'spline':
        if X_train.shape[1] != 1:
            raise ValueError("样条插值仅支持单特征输入！")
        s = s if s is not None else len(y_train) * np.std(y_train) * 0.1
        model = UnivariateSpline(X_train.flatten(), y_train, s=s)
        predict_func = lambda x: model(np.ascontiguousarray(x).flatten())
        
    else:
        raise ValueError("method必须是 'linear', 'poly', 或 'spline'")
    
    # 评估模型
    y_pred = predict_func(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"评估结果 - MSE: {mse:.4f}, R²: {r2:.4f}")
    
    predictions = predict_func(new_data)
    
    if with_label:
        new_data1 = new_data.copy()
        new_data1['prediction'] = predictions
        return new_data1
    else:
        return predictions

def svm_classify(data, independent, dependent, **kwargs):
    if isinstance(independent, str):
        X = data[[independent]]
    else:
        X = data[independent]
    y = data[dependent]

    model = SVC(**kwargs)
    model.fit(X, y)
    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)

    return {"model": model, "y_pred": y_pred, "accuracy": accuracy, "report": report}

def logistic_classify(data, independent, dependent, **kwargs):
    if isinstance(independent, str):
        X = data[[independent]]
    else:
        X = data[independent]
    y = data[dependent]

    model = LogisticRegression(**kwargs)
    model.fit(X, y)
    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)

    return {"model": model, "y_pred": y_pred, "accuracy": accuracy, "report": report}

def knn_classify(data, independent, dependent, **kwargs):
    if isinstance(independent, str):
        X = data[[independent]]
    else:
        X = data[independent]
    y = data[dependent]

    model = KNeighborsClassifier(**kwargs)
    model.fit(X, y)
    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)

    return {"model": model, "y_pred": y_pred, "accuracy": accuracy, "report": report}

def linear_regression(data, independent, dependent):
    """
    线性回归（简单或多元）
    返回包含模型、统计摘要、R方等信息的字典
    """
    df = data.copy()
    X = sm.add_constant(df[independent])  # 自动处理Series和DataFrame
    y = df[dependent]
    model = sm.OLS(y, X, missing='drop').fit()
    
    return {
        "model": model,
        "summary": model.summary().as_text(),
        "r2": model.rsquared,
        "adj_r2": model.rsquared_adj,
        "pvalues": model.pvalues.to_dict(),
        "residuals": model.resid
    }

def polynomial_regression(data, independent, dependent, degree=2):
    """
    多项式回归（单变量）
    返回包含多项式模型、特征转换器、预测值等信息的字典
    """
    df = data.copy()
    # 确保使用第一个特征如果是列表
    indep_var = independent[0] if isinstance(independent, list) else independent
    X_raw = df[[indep_var]]
    y = df[dependent].values
    
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_raw)
    
    lr = LinearRegression().fit(X_poly, y)
    y_pred = lr.predict(X_poly)
    
    return {
        "model": lr,
        "poly_features": poly,
        "degree": degree,
        "r2": r2_score(y, y_pred),
        "y_pred": y_pred
    }

def exponential_regression(data, independent, dependent):
    """
    指数回归（单变量）
    返回包含对数空间模型、系数、预测值等信息的字典
    """
    df = data.copy()
    # 确保使用第一个特征如果是列表
    indep_var = independent[0] if isinstance(independent, list) else independent
    df_temp = df[(df[dependent] > 0)].dropna()
    
    if df_temp.empty:
        raise ValueError("No valid data after filtering positive dependent values")
    
    X = sm.add_constant(df_temp[indep_var])
    y_log = np.log(df_temp[dependent])
    
    model = sm.OLS(y_log, X, missing='drop').fit()
    a = np.exp(model.params[0])
    b = model.params[1]
    
    return {
        "model": model,
        "a": a,
        "b": b,
        "r2": model.rsquared,
        "y_pred": a * np.exp(b * df_temp[indep_var])
    }

def power_regression(data, independent, dependent):
    """
    幂律回归（单变量）
    返回包含对数空间模型、系数、预测值等信息的字典
    """
    df = data.copy()
    # 确保使用第一个特征如果是列表
    indep_var = independent[0] if isinstance(independent, list) else independent
    
    df_temp = df[(df[independent] > 0) & 
                 (df[dependent] > 0)].dropna()
    
    if df_temp.empty:
        raise ValueError("No valid data after filtering positive values")
    
    X = sm.add_constant(np.log(df_temp[indep_var]))
    y_log = np.log(df_temp[dependent])
    
    model = sm.OLS(y_log, X, missing='drop').fit()
    a = np.exp(model.params[0])
    b = model.params[1]
    
    y_pred = a * np.power(df_temp[indep_var], b)
    
    return {
        "model": model,
        "a": a,
        "b": b,
        "r2": r2_score(df_temp[dependent], y_pred),
        "y_pred": y_pred
    }

def ransac_regression(data, independent, dependent):
    """
    RANSAC鲁棒回归（单变量）
    返回包含模型、内点掩码、预测值等信息的字典
    """
    df = data.copy()
    # 确保使用第一个特征如果是列表
    indep_var = independent[0] if isinstance(independent, list) else independent
    
    X = df[[indep_var]].values
    y = df[dependent].values
    
    ransac = RANSACRegressor().fit(X, y)
    y_pred = ransac.predict(X)
    
    return {
        "model": ransac,
        "inlier_mask": ransac.inlier_mask_,
        "r2": r2_score(y[ransac.inlier_mask_], y_pred[ransac.inlier_mask_]),
        "y_pred": y_pred
    }

# LSTM模型，返回new_row长度的预测值
def train_lstm_model(
    train_data: pd.DataFrame,
    new_row: int,
    target_col: str,
    seq_length: int,
    split: float = 0.2,
    num_epochs: int = 10,
    input_size: int = 1,
    hidden_size: int = 64,
    output_size: int = 1,
    num_layers: int = 3,
    learning_rate: float = 0.001,
    criterion_fun: Callable = nn.MSELoss(),
) -> np.ndarray:
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 数据预处理
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(train_data[target_col].values.reshape(-1, 1))  # 归一化到 [0, 1]

    # 创建时间序列数据集
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:i + seq_length]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    X, y = create_sequences(data_normalized, seq_length)

    # 划分训练集和测试集
    test_size = int(len(X) * split)  # 20% 作为测试集
    train_size= len(X) - test_size
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 转换为 PyTorch 张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # 创建 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 定义 LSTM 模型
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.lstm(x)  # LSTM 输出
            out = self.fc(out[:, -1, :])  #
            return out
        
    def train_lstm_model(model, dataloader, criterion, optimizer, num_epochs=num_epochs, device=device):
        model.to(device)
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    def test_lstm_model(model, dataloader, criterion, device=device):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Test Loss: {avg_loss:.4f}")
        return avg_loss
    
    model = LSTMModel(input_size, hidden_size, output_size, num_layers)
    criterion = criterion_fun
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    train_lstm_model(model, train_dataloader, criterion, optimizer, num_epochs, device)

    # 测试模型
    test_loss = test_lstm_model(model, test_dataloader, criterion, device)

    # 往后预测new_row长度
    def predict_future(model, last_sequence, new_row, device='cuda'):
        model.eval()
        predictions = []
        current_sequence = last_sequence.clone().detach().to(device)  # 将最后一个序列移动到设备上

        with torch.no_grad():
            for _ in range(new_row):
                # 预测下一个时间步的值
                output = model(current_sequence.unsqueeze(0))  # 增加 batch 维度
                predictions.append(output.item())

                # 更新当前序列，去掉第一个时间步，添加预测值
                current_sequence = torch.cat((current_sequence[1:], output), dim=0)

        return predictions

    # 获取最后一个序列作为预测的起点
    last_sequence = X_test[-1].to(device)

    # 使用模型进行预测
    predictions = predict_future(model, last_sequence, new_row, device)

    # 将预测结果反归一化
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    return predictions

# ARIMA模型
def train_arima_model(
    train_data: pd.DataFrame,
    target_col: str,
    new_row: int,
    split: float = 0.2,
    seasonal: bool = False,
    seasonal_period: int = None,
) -> np.ndarray:
    """
    output: np.ndarray,shape(new_row,1)
    """
    # 参数检查
    if seasonal and seasonal_period is None:
        raise ValueError("seasonal_period must be provided when seasonal=True")
    # 数据预处理
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(train_data[target_col].values.reshape(-1, 1))  # 归一化到 [0, 1]
    data_series = pd.Series(data_normalized.flatten())  # 转换为 Pandas Series

    # 划分训练集和测试集
    train_size = int(len(data_series) * (1 - split))  # 80% 作为训练集
    train_data_series, test_data_series = data_series[:train_size], data_series[train_size:]

    # 使用 auto_arima 自动选择参数并拟合模型
    if seasonal:
        model = auto_arima(train_data_series, seasonal=seasonal, m=seasonal_period, trace=True)
    else:
        model = auto_arima(train_data_series, seasonal=seasonal, trace=True)
    print(model.summary())  # 打印模型摘要

    # 在测试集上评估模型
    test_predictions = model.predict(n_periods=len(test_data_series))
    test_mse = mean_squared_error(test_data_series, test_predictions)
    print(f"Test MSE: {test_mse:.4f}")

    # 预测未来 new_row 个时间步
    future_predictions = model.predict(n_periods=new_row)

    # 将预测结果反归一化
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions

