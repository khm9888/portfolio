import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import datetime,time
import math


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

price_df["month"] =[i[-5:-3] for i in list(price_df.index)]
price_df["year"] =[i[:4] for i in list(price_df.index)]

price_df["quarter"] =[ str(math.ceil(int(i[-5:-3])/3))+"q" for i in list(price_df.index) if i]


from sklearn.preprocessing import StandardScaler

# scaler = MinMaxScaler()
# scaler2 = MinMaxScaler()

scaler = StandardScaler()
scaler2 = StandardScaler()

# mapping

price_df.loc[:,["snp500","nikkei225","shanghai"]]=scaler.fit_transform(price_df.loc[:,["snp500","nikkei225","shanghai"]])*100
price_df.loc[:,["kospi200"]]=scaler.fit_transform(price_df.loc[:,["kospi200"]])*100

pivot = pd.pivot_table(price_df,index=["year","quarter"])

print(pivot)

sns.heatmap(pivot, annot=True, fmt='f')
plt.show()

# sns.set(style="darkgrid")#style must be one of white, dark, whitegrid, darkgrid, ticks

# #x - snp500 y- kospi200 
# sns.relplot(x="snp500",y="kospi200",data=price_df,hue="month",style ="year", palette="Blues")#hue = 색상 , style - maeker style


# # #x - snp500 y- kospi200  
# sns.relplot(x="nikkei225",y="kospi200",data=price_df,hue="month",size ="year", palette="terrain")#hue = 색상 , style - maeker style, size = 크기
# # plt.show()

# sns.relplot(x="shanghai",y="kospi200",data=price_df,hue="month",style ="year")#hue = 색상 , style - maeker style
# # plt.show()#선형

# plt.show()