import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv("indian_urban_house_price_practice_100k.csv")
df.shape
df.columns
df.info()
df.head()
df.tail()
df.describe()


df.isnull().sum()
df['Power_Backup'] = df['Power_Backup'].fillna(df['Power_Backup'].mode()[0])
df.duplicated().sum()



x=df.drop(columns=["Price_INR_Lakhs","Property_ID"])
y=df["Price_INR_Lakhs"]

x.columns

x_num=x.select_dtypes("number")

x_num.dtypes

x_cat=x.select_dtypes("object")

for i in x_cat:
    print(i)
    print("*" *50)
    print(x_cat[i].value_counts())


from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

oe=OrdinalEncoder()
x[["Locality_Type","Furnishing_Status"]]=oe.fit_transform(x[["Locality_Type","Furnishing_Status"]])

x_one=x[["City","Property_Type","Water_Supply","Power_Backup","Facing","Ownership_Type","Construction_Status","Transaction_Type"]]


oh=OneHotEncoder(sparse_output=False)
encoder=oh.fit_transform(x_one)
encoded=pd.DataFrame(encoder,columns=oh.get_feature_names_out())
x=x.drop(["City","Property_Type","Water_Supply","Power_Backup","Facing","Ownership_Type","Construction_Status","Transaction_Type"],axis=1)
x=pd.concat([x,encoded],axis=1)

x.shape

#OUTLIYEAR DETECTION

for col in x_num:
    Q1=x[col].quantile(0.25)
    Q3=x[col].quantile(0.75)
    IQR=Q3-Q1
    lower=Q1-1.5*IQR
    upper=Q3+1.5*IQR
    outliyers=((x[col]<lower)|(x[col]>upper)).sum()
    print(f"{col}:{outliyers} outliyers")

ol=["Distance_to_City_Center_km", "Hospital_Distance_km","School_Distance_km", "Shopping_Mall_Distance_km","Noise_Level_dB"]

x[ol]=np.log1p(x[ol])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.3)

#basemodels

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)


from sklearn.feature_selection import SelectKBest, f_regression
fs = SelectKBest(score_func=f_regression, k=20)

x_train_fs = fs.fit_transform(x_train_scaled, y_train)

x_test_fs = fs.transform(x_test_scaled)


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train_scaled,y_train)
y_lrpredict=lr.predict(x_test_scaled)

from sklearn.metrics import r2_score,root_mean_squared_error,mean_squared_error
lr_r2=r2_score(y_test,y_lrpredict)
lr_rmse=root_mean_squared_error(y_test,y_lrpredict)

from sklearn.linear_model import Lasso
ls=Lasso()
ls.fit(x_train_scaled,y_train)
y_lspredict=ls.predict(x_test_scaled)

ls_r2=r2_score(y_test,y_lspredict)
ls_rmse=root_mean_squared_error(y_test,y_lspredict)

from sklearn.linear_model import Ridge
rd=Ridge()
rd.fit(x_train_scaled,y_train)
y_rdpredict=rd.predict(x_test_scaled)

rd_r2=r2_score(y_test,y_rdpredict)
rd_rmse=root_mean_squared_error(y_test,y_rdpredict)


from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(n_neighbors=5)
knn.fit(x_train_fs,y_train)
y_knnpredict=knn.predict(x_test_fs)

knn_r2 = r2_score(y_test, y_knnpredict)
knn_rmse=root_mean_squared_error(y_test,y_knnpredict)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor(random_state=42)
dt.fit(x_train,y_train)
y_dtpredict=dt.predict(x_test)

dt_r2 = r2_score(y_test, y_dtpredict)
dt_rmse=root_mean_squared_error(y_test,y_dtpredict)

base_model_results = { "Linear Regression": [lr_r2, lr_rmse], "Lasso Regression": [ls_r2, ls_rmse], "Ridge Regression": [rd_r2, rd_rmse], "KNN Regressor": [knn_r2, knn_rmse], "Decision Tree Regressor": [dt_r2, dt_rmse] }

base_model_results


import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


from sklearn.metrics import r2_score,root_mean_squared_error

#load dataset

df = pd.read_csv("indian_urban_house_price_practice_100k.csv")

#separate x and y

X = df.drop(
    columns=[
        "Property_ID",
        "Price_INR_Lakhs",
        "Price_Per_SqFt_INR"])
y = df["Price_INR_Lakhs"]

#separete numeric and categorical columns

numeric_cols = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_cols = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

#spilt data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#numeric pipeline

numeric_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

#categorical pipeline

categorical_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


#preprocessor

preprocessor = ColumnTransformer([
    ("num", numeric_preprocessor, numeric_cols),
    ("cat", categorical_preprocessor, categorical_cols)
])


# Decision Tree Regressor Pipeline

dt_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("dtmodel", DecisionTreeRegressor(random_state=42))
])
dt_pipeline.fit(X_train, y_train)
y_pred_dt=dt_pipeline.predict(X_test)

r2_dt = r2_score(y_test, y_pred_dt)
rmse_dt = root_mean_squared_error(y_test, y_pred_dt)


joblib.dump(dt_pipeline, "dt_pipeline.pkl")
print("model saved successfully")
