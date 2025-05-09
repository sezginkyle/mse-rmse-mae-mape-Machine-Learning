import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
df=pd.read_csv("medical.zip")
df.head(3)
df=pd.get_dummies(df,columns=["sex","smoker","region"],drop_first=1)
df.head(3)
df=df.astype(float)
df.info()
y=df[["charges"]]
x=df.drop("charges",axis=1)
l=LinearRegression()
model=l.fit(x,y)
model.score(x,y)
model.predict([[19,33,1,0,0,1,0,1]])
df_charges=pd.DataFrame()
df_charges["charges"]=y
df_charges.head(3)
df_tahmin=model.predict(x)
df_charges["Tahmin"]=df_tahmin
df_charges["Fark"]=df_charges["charges"]-df_charges["Tahmin"]
df_charges["Hatanın Karesi"]=df_charges["Fark"]*df_charges["Fark"]
for i in range(len(df_charges)):
    if df_charges["Fark"][i] < 0:
        df_charges.at[i, "Mutlak Fark"] = -df_charges["Fark"][i]
    else:
        df_charges.at[i, "Mutlak Fark"] = df_charges["Fark"][i]
df_charges["Yuzdelik Hata"]=df_charges["Mutlak Fark"]/df["charges"]
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
df_charges.mean()
mean_squared_error(charges,Tahmin)
charges=df_charges["charges"]
Tahmin=df_charges["Tahmin"]
mean_absolute_error(charges,Tahmin)
mean_absolute_percentage_error(charges,Tahmin)
