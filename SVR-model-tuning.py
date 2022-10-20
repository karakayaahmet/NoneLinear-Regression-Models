from cv2 import mean
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVR

from warnings import filterwarnings
filterwarnings("ignore")

df = pd.read_csv("Hitters.csv")
df = df.dropna()
dms = pd.get_dummies(df[["League", "Division", "NewLeague"]])
y = df["Salary"]
X_ = df.drop(["League", "Division", "Salary", "NewLeague"], axis=1).astype('float64')
X = pd.concat([X_, dms[["League_N", "Division_W", "NewLeague_N"]]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)

svr_model = SVR(kernel="linear")
svr_model = svr_model.fit(X_train, y_train)

#************************************************************
# model tuning
svr_model = SVR(kernel = "linear").fit(X_train, y_train)
svr_params = {"C":[0.1, 0.5, 1, 3]}
svr_model = GridSearchCV(svr_model, svr_params, cv=5).fit(X_train, y_train)

print(svr_model.best_params_)

#final
svr_tuned = SVR(kernel="linear", C=0.5).fit(X_train, y_train)
y_pred = svr_tuned.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, y_pred)))
