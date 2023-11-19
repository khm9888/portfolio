import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense,Input
from sklearn.metrics import r2_score
import numpy as np
from sklearn.model_selection import train_test_split as tts
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor



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


model = GradientBoostingRegressor()


#트레이닝


model.fit(x_train,y_train)

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
# r2:0.9375571510881826
# y_test:[296.83 281.28 293.11 316.2  300.07 277.09 316.4  313.82 318.51 279.85
#  293.7  284.37 289.42 220.34 295.2  317.72 273.88 333.57 313.22 288.57]
# y_pre:[295.55090438 280.56559431 283.25023137 316.83249036 295.02042445
#  277.90059628 319.79029264 313.52306508 317.99191488 286.95207712
#  294.84244462 281.21540835 293.8936808  206.49156716 282.21564584
#  316.46760752 270.93687078 333.14630397 315.24896179 291.15547214]
# tomorrow`s kospi 200 : [279.18875607]
# tomorrow`s truly kospi 200 : 280.46