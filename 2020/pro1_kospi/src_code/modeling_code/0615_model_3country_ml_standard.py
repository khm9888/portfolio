import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense,Input
from sklearn.metrics import r2_score
import numpy as np
from sklearn.model_selection import train_test_split as tts
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense,Input
from sklearn.metrics import r2_score
import numpy as np
from sklearn.model_selection import train_test_split as tts
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import all_estimators




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

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
x_pre=scaler.transform(x_pre)

# print(x_pre)

all_Algorithms = all_estimators(type_filter='regressor')

good=[]
bad=[]
most=""
value=0
for name,algorithms in all_Algorithms:
    try:
        model = algorithms()
        model.fit(x_train,y_train)
        y_pre=model.predict(x_test)
        if model.score(x_test,y_test)>=0.8:
            good.append(f"{name}의 정답률 = {model.score(x_test,y_test)}")
        else:
            bad.append(f"{name}의 정답률 = {model.score(x_test,y_test)}")
        if value<model.score(x_test,y_test):
            most=name
            value=model.score(x_test,y_test)
        y_pre_tomorrow = model.predict(x_pre)
        y_pre_tomorrow = y_pre_tomorrow.reshape(-1,)
        if model.score(x_test,y_test)>=0.5:
            print(name)
            print("tomorrow`s kospi 200 :",y_pre_tomorrow)
            print("tomorrow`s truly kospi 200 :",y_predict)
    except:
        pass


print("------------most-----------")
print(most)
print(value)
print("------------good-----------")
for al in good:
    print(al)
    
print("------------bad-----------")
for al in bad:
    print(al)

