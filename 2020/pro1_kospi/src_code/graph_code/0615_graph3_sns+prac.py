import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime,time


price_df=pd.read_csv("project/pro1/test.csv",index_col=0,header=0)

price_df = price_df.iloc[:-1]

print(type(price_df))

print("price_df")
print(price_df)
print("price_df.describe()")
print(price_df.describe())
print("price_df.info()")
price_df.info()
print("price_df.shape")
print(price_df.shape)

x_df=price_df.loc[:,["snp500","nikkei225","shanghai"]]
y_df=price_df.loc[:,["kospi200"]]

price_df["month"] =[i[-5:-3] for i in list(price_df.index)]
price_df["year"] =[i[:4] for i in list(price_df.index)]

# y_df["month"] =[i[-5:-3] for i in list(y_df.index)]
# y_df["year"] =[i[:4] for i in list(y_df.index)]

# months=y_df.groupby("month")["kospi200"].count()
# years=y_df.groupby("year")["kospi200"].count()
# print(months)
# print(years)

x_df.info()

# y_df= y_df.reshape(-1,1)
# print(type(x_df))

# plt.figure(figsize=(15,8))
from sklearn.preprocessing import RobustScaler

# scaler = MinMaxScaler()
# scaler2 = MinMaxScaler()

scaler = RobustScaler()


price_df.loc[:,["snp500","nikkei225","shanghai"]]=scaler.fit_transform(price_df.loc[:,["snp500","nikkei225","shanghai"]])*100
price_df.loc[:,["kospi200"]]=scaler.fit_transform(price_df.loc[:,["kospi200"]])*100


sns.set(style="darkgrid")#style must be one of white, dark, whitegrid, darkgrid, ticks

#x - snp500 y- kospi200 
sns.relplot(x="snp500",y="kospi200",data=price_df,hue="month",style ="year", palette="mako")#hue = 색상 , style - maeker style
plt.show()

# #x - snp500 y- kospi200  
sns.relplot(x="nikkei225",y="kospi200",data=price_df,hue="month",size ="year", palette="icefire")#hue = 색상 , style - maeker style, size = 크기
plt.show()

sns.relplot(x="shanghai",y="kospi200",data=price_df,hue="month",style ="year",palette="mako")#hue = 색상 , style - maeker style
plt.show()#선형

plt.show()

# sns.relplot(x="shanghai",y="kospi200",data=price_df,hue="month",style ="year")#hue = 색상 , style - maeker style
# plt.show()


# #x - snp500 y- kospi200 
# sns.relplot(x="snp500",y="kospi200",data=price_df) 
# plt.show()


tips = sns.load_dataset("tips")

# print(tips)
# sns.catplot(x="day",y="total_bill",data=tips)#stripplot
# plt.show()


# sns.catplot(x="day",y="total_bill",data=tips,kind = "swarm")#swarm
# plt.show()



# sns.catplot(x="day",y="total_bill",data=tips,kind ="box")#box
# plt.show()


# sns.catplot(x="day",y="total_bill",data=tips,kind ="violin",split=True,hue="sex")#violin
# plt.show()

# sns.catplot(x="day",y="total_bill",data=tips,kind ="boxen")#boxen
# plt.show()

# sns.catplot(x="day",y="total_bill",data=tips,kind ="point")
# sns.catplot(x="day",y="total_bill",data=tips,kind ="bar")s
# # sns.catplot(x="day",y="total_bill",data=tips,kind ="count")
# plt.show()

# titanic = sns.load_dataset("titanic")
# sns.catplot(x="sex",y="survived",hue="class",kind="bar",data =titanic)
# plt.show()


# df = pd.DataFrame(np.random.randn(500,2),columns=["0","1"])
# sns.relplot(x = "0",y="1",data=df)
# plt.show()
# sns.jointplot()

# df = pd.DataFrame(np.random.randn(500,2),columns=["0","1"])
# sns.jointplot(x = "0",y="1",data=df,kind ="hex")
# plt.show()

# df = pd.DataFrame(np.random.randn(500,2),columns=["0","1"])
# sns.jointplot(x = "0",y="1",data=df,kind ="kde")
# plt.show()
# # 