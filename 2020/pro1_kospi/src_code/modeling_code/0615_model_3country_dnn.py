import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense,Input
from sklearn.metrics import r2_score
import numpy as np
from sklearn.model_selection import train_test_split as tts

import seaborn as sns

price_df=pd.read_csv("project\pro1/test.csv",index_col=0,header=0)

print(type(price_df))

print("price_df")
print(price_df)

print("price_df.describe()")
print(price_df.describe())

print("price_df.info()")
price_df.info()

# x_df=price_df.loc[price_df.index[1]:,["s&p 500","nikkei 225","shanghai"]]#가장 최근값 없앰
# y_df=price_df.loc[:price_df.index[-2],["kospi200"]]#가장 옛날값 없앰


price_array = price_df.values

def split(datasets,timesteps):
    x_values=list()
    y_values=list()
    for i in range(len(datasets)-timesteps):
        x=datasets[i:i+timesteps,1:]
        y=datasets[i+timesteps,0]
        x_values.append(x)
        y_values.append(y)
    return np.array(x_values),np.array(y_values)

x,y=split(price_array,6)
print("x.shape")
x = x.reshape(-1,x.shape[1]*x.shape[2])
print(x.shape)
print("y.shape")
print(y.shape)

print(x[0])

x_train,x_test,y_train,y_test = tts(x,y,train_size=0.7)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


#모델

input1=Input(shape=(x.shape[1],))
dense=Dense(3000,activation="relu")(input1)
dense=Dense(1,activation="relu")(dense)#회귀

model = Model(inputs=input1,outputs=dense)

model.summary()

#트레이닝

model.compile(loss="mse", optimizer="adam")
model.fit(x_train,y_train,batch_size=10,epochs=350,validation_split=0.3)

#테스트

loss= model.evaluate(x_test,y_test,batch_size=100)

y_pre=model.predict(x_test)
y_pre = y_pre.reshape(-1,)
r2_score(y_test,y_pre)
print(__file__)
print(f"loss:{loss}")
print(f"r2:{r2_score(y_test,y_pre)}")
# print(f"x_test.shape:{x_test.shape}")
# print(f"y_pre.shape:{y_pre.shape}")

print(f"y_test:{y_test[0:20]}")
print(f"y_pre:{y_pre[0:20]}")

plt.figure(figsize=(6,6))

plt.plot(y_test[0:20],label="y_test")
plt.plot(y_pre[0:20],label="y_pre")

plt.legend(loc=0)
plt.show()

'''
loss:86.02171984396345
r2:0.871698327197065
y_test:[257.01 281.67 266.65 302.25 296.95 271.36 290.68 333.62 313.45 279.93
 300.81 304.58 272.1  281.31 262.95 310.92 297.78 266.12 310.87 252.4 ]
y_pre:[246.97429 267.4378  275.48074 295.87573 295.07504 274.92264 296.22302
 339.94867 302.35666 288.43866 289.1289  310.83475 270.82376 297.16757
 273.474   313.964   296.21616 273.7647  314.17484 256.997  ]

'''

'''
loss:84.86799684118051
r2:0.8585464057383454
y_test:[277.97 277.78 267.95 298.33 254.86 313.82 271.88 284.97 298.8  260.19
 326.23 277.55 294.96 303.69 304.58 267.4  270.84 276.01 314.29 320.8 ]
y_pre:[277.24225 284.70264 276.95923 285.29266 260.0305  304.82687 268.0388
 268.02432 288.09305 273.70297 328.96924 286.35703 285.66464 298.58823
 308.85065 268.8343  268.30316 269.05618 316.67654 306.5962 ]
'''

'''
loss:125.20787415217832
r2:0.8137544747431137
y_test:[314.97 299.74 252.83 291.08 323.59 316.25 288.4  274.42 271.36 333.62
 295.2  273.79 279.4  316.2  287.89 290.11 318.31 288.62 283.   293.35]
y_pre:[300.66785 291.66647 257.68872 280.04028 322.3942  305.92905 260.82526
 272.75934 262.5493  333.1319  293.49164 278.00043 280.46445 304.11362
 268.67636 283.9939  308.8671  280.00406 259.38913 290.70712]
'''