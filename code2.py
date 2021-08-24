#%%
import pandas as pd
# %%
df=pd.read_csv("gapminder.csv")
# %%
df.head()
# %%
y=df["fertility"]
# %%
x=df.drop("fertility",axis=1)
# %%
x.isna().sum()
# %%
y.isna().sum()
# %%
x.dtypes
# %%
x=pd.get_dummies(x,columns=["Region"])
# %%
type(x)
# %%
x.shape
# %%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)

# %%
x.describe()
# %%
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
# %%
from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor(min_samples_leaf=5)
model.fit(x_train,y_train)
model.score(x_test,y_test)
# %%
model.score(x_train,y_train)
# %%
