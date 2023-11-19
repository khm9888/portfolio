import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense,Input
from sklearn.metrics import r2_score
import numpy as np
from sklearn.model_selection import train_test_split as tts
import seaborn as sns


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
        x=datasets[i:i+timesteps,2:]#0
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

from sklearn.preprocessing import MinMaxScaler,RobustScaler

scaler = RobustScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
x_pre=scaler.transform(x_pre)

# print(x_pre)

#모델

input1=Input(shape=(x.shape[1],))
dense=Dense(3000,activation="relu")(input1)
dense=Dense(1,activation="relu")(dense)#회귀

model = Model(inputs=input1,outputs=dense)

model.summary()

#트레이닝

model.compile(loss="mse", optimizer="adam")
model.fit(x_train,y_train,batch_size=10,epochs=300,validation_split=0.3,verbose=2)

#테스트

loss= model.evaluate(x_test,y_test,batch_size=100)

y_pre=model.predict(x_test)
y_pre = y_pre.reshape(-1,)

print(__file__)
print(f"loss:{loss}")
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
'''
loss:65.35762941120753
r2:0.8941167093140132
y_test:[272.37 276.9  263.93 333.62 337.14 296.95 313.82 254.85 316.83 330.68
 261.57 272.08 271.57 268.27 302.78 295.49 293.64 288.62 324.58 279.4 ]
y_pre:[277.0359  284.15567 269.65958 329.12198 332.21625 291.3266  310.2673
 260.02765 304.6729  326.15533 243.99188 269.29858 276.4253  270.37827
 300.65152 291.69168 293.15652 287.18945 321.83624 287.59952]
tomorrow`s kospi 200 : [290.0844]
tomorrow`s truly kospi 200 : 280.46
'''
'''
loss:64.36044299016234
r2:0.8963805492909578
y_test:[308.39 278.03 269.93 287.89 270.67 265.99 256.28 261.37 326.6  284.38
 260.19 335.49 326.12 272.11 278.63 282.59 290.11 314.97 286.38 251.88]
y_pre:[305.70554 282.95218 267.06866 278.36884 270.94962 271.3525  264.36636
 270.4281  319.97647 286.45862 264.4598  337.82632 323.73544 272.49686
 286.42847 275.61917 283.17108 315.3741  296.10767 247.30179]
tomorrow`s kospi 200 : [290.2241]
tomorrow`s truly kospi 200 : 280.46

loss:70.08018848023129
r2:0.8740303635642226
y_test:[267.75 270.45 274.84 262.47 271.88 300.07 271.35 317.72 254.31 327.61
 318.01 268.16 267.99 309.71 294.85 298.8  276.44 272.37 272.11 240.65]
y_pre:[278.14163 272.01648 283.81732 264.78275 270.75787 290.8852  278.48355
 316.55676 260.18018 334.3228  326.9092  265.35336 261.7739  294.40967
 286.90692 289.6222  282.54062 280.1979  277.96378 247.53119]
tomorrow`s kospi 200 : [283.7709]
tomorrow`s truly kospi 200 : 280.46
'''