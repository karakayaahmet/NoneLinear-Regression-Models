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

#ilk olarak standartlaştırma yapılmakta.
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
mlp_model = MLPRegressor().fit(X_train_scaled, y_train)

y_pred = mlp_model.predict(X_test_scaled)
print(np.sqrt(mean_squared_error(y_test, y_pred)))

#***********************************************************

mlp_params = {"alpha":[0.1, 0.01, 0.001, 0.0001],
                "hidden_layer_sizes": [(10,20), (5,5),(100,100)]}

mlp_cv_model = GridSearchCV(mlp_model, mlp_params, cv=10, verbose = 2, n_jobs = -1).fit(X_train, y_train)
print(mlp_cv_model)
print(mlp_cv_model.best_params_)

#final
mlp_tuned = MLPRegressor(alpha=0.1, hidden_layer_sizes=(100,100)).fit(X_train_scaled, y_train)
y_pred = mlp_tuned.predict(X_test_scaled)
print(np.sqrt(mean_squared_error(y_test, y_pred )))
