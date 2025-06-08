ml="""
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
"""



