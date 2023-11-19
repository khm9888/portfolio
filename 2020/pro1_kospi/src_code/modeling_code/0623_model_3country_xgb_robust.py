
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense,Input
from sklearn.metrics import r2_score
import numpy as np
from sklearn.model_selection import train_test_split as tts
import seaborn as sns
# from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor



price_df=pd.read_csv("project\pro1/test.csv",index_col=0,header=0)

# print(type(price_df))

# print("price_df")
# print(price_df)

# print("price_df.describe()")
# print(price_df.describe())

# print("price_df.info()")
# price_df.info()

# x_df=price_df.loc[price_df.index[1]:,["s&p 500","nikkei 225","shanghai"]]#가장 최근값 없앰
# y_df=price_df.loc[:price_df.index[-2],["kospi200"]]#가장 옛날값 없앰


price_array = price_df.values

def split(datasets,timesteps):
    x_values=list()
    y_values=list()
    for i in range(len(datasets)-timesteps):#10-6
        x=datasets[i:i+timesteps,1:]#0
        y=datasets[i+timesteps,0]
        x_values.append(x)
        y_values.append(y)
    return np.array(x_values),np.array(y_values)

x,y=split(price_array,6)
x,x_pre,y,y_predict = x[:-1],x[-1],y[:-1],y[-1]
# print("x.shape")
x = x.reshape(-1,x.shape[1]*x.shape[2])
# print(x.shape)
# print("y.shape")
# print(y.shape)

x_pre = x_pre.reshape(-1,x_pre.shape[0]*x_pre.shape[1])

# print(x_pre)
# print(x_pre.shape)

x_train,x_test,y_train,y_test = tts(x,y,train_size=0.7)

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
x_pre=scaler.transform(x_pre)

# print(x_pre)

#모델


model = XGBRegressor(n_estimators=2000)


#트레이닝


model.fit(x_train,y_train,eval_metric="rmse",eval_set=((x_train,y_train),(x_test,y_test)))

#테스트

y_pre=model.predict(x_test)
y_pre = y_pre.reshape(-1,)

print(__file__)

print(f"r2:{r2_score(y_test,y_pre)}")
# print(f"x_test.shape:{x_test.shape}")
# print(f"y_pre.shape:{y_pre.shape}")

print(f"y_test:{y_test[0:20]}")
print(f"y_pre:{y_pre[0:20]}")

y_values = pd.DataFrame(dict(y_test=y_test[0:20],y_pre=y_pre[0:20]),index=range(1,21))

# print(y_values)

plt.figure(figsize=(6,6))

sns.lineplot(x=y_values.index,y="y_test", data=y_values , label ="y_test" )
sns.lineplot(x=y_values.index,y="y_pre", data=y_values ,label="y_pre")

plt.xlabel("count")
plt.ylabel("y=kospi")


plt.title("y_test & y_pre")
plt.legend(loc=0)
plt.show()

y_pre_tomorrow = model.predict(x_pre)
y_pre_tomorrow = y_pre_tomorrow.reshape(-1,)

print("tomorrow`s kospi 200 :",y_pre_tomorrow)
print("tomorrow`s truly kospi 200 :",y_predict)




# # xgbregressor

# r2:0.941773306488678
# y_test:[276.64 283.93 275.1  281.28 326.99 247.62 283.7  316.27 288.57 315.37
#  310.42 276.19 275.57 333.4  300.81 279.43 293.11 279.31 325.79 288.62]
# y_pre:[292.2866  282.3562  280.73795 283.4182  329.49478 261.39832 296.89334
#  321.96588 293.97784 319.02112 313.59146 276.7915  286.8655  334.5672
#  294.1892  279.89813 283.49585 285.53305 324.87302 284.93253]
# tomorrow`s kospi 200 : [280.02954]
# tomorrow`s truly kospi 200 : 282.35