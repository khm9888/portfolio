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
    for i in range(len(datasets)-timesteps):#10-6
        x=datasets[i:i+timesteps,1:]#0
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

loss:125.48260927460885
r2:0.8208361706437616
y_test:[277.48 317.72 316.81 294.85 271.23 283.65 286.62 262.47 312.42 314.
 229.34 295.9  275.92 295.24 286.15 291.32 279.37 282.51 288.57 286.71]
y_pre:[287.09933 299.59863 305.5123  283.42236 262.72394 289.32004 266.48325
 272.53226 311.6162  301.29514 240.73581 284.88202 271.2196  281.57828
 292.63898 284.616   292.52936 273.29358 281.1274  299.79315]

'''

'''
loss:140.2560934264803
r2:0.7553601950513429
y_test:[318.89 297.22 270.09 265.73 265.18 254.85 338.05 286.62 288.61 275.48
 303.69 275.15 334.36 296.65 271.36 274.84 281.31 275.53 292.46 287.29]
y_pre:[322.98508 287.61362 273.00467 279.22952 274.11206 263.2568  350.27676
 266.2851  280.47656 275.08936 287.0352  304.639   331.32056 297.86145
 282.45483 287.96164 309.0614  279.86893 284.08176 296.82465]
'''
 
'''
 loss:131.32208264460328
r2:0.8060840730120318
y_test:[300.93 327.01 271.36 276.78 281.38 247.62 294.28 252.   226.35 320.12
 300.51 293.55 321.79 337.4  330.25 328.4  266.65 293.7  301.53 333.4 ]
y_pre:[300.19547 338.28455 291.18146 294.68347 270.49814 265.25208 288.94836
 264.42322 258.6808  298.93723 303.19043 285.9956  313.7672  330.15338
 325.19626 323.21982 271.05948 284.64447 295.88608 343.6884 ]
 
 
 
 
'''