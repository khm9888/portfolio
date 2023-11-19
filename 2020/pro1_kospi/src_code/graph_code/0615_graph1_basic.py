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



price_df["month"] =[i[-5:-3] for i in list(price_df.index)]
price_df["year"] =[i[:4] for i in list(price_df.index)]

# months=y_df.groupby("month")["kospi200"].count()
# years=y_df.groupby("year")["kospi200"].count()
# print(months)
# print(years)

# x_df.info()

# y_df= y_df.reshape(-1,1)
# print(type(x_df))

# plt.figure(figsize=(15,8))
from sklearn.preprocessing import MinMaxScaler,StandardScaler

scaler = StandardScaler()


price_df.loc[:,["snp500","nikkei225","shanghai"]]=scaler.fit_transform(price_df.loc[:,["snp500","nikkei225","shanghai"]])*100
price_df.loc[:,["kospi200"]]=scaler.fit_transform(price_df.loc[:,["kospi200"]])*100


###################################################


plt.figure(figsize=(15,8))

plt.plot(price_df["kospi200"],label="kospi200")
plt.plot(price_df["snp500"],label="snp500")
plt.plot(price_df["nikkei225"],label="nikkei225")
plt.plot(price_df["shanghai"],label="shanghai")

plt.xlabel("date")
plt.title("kospi & 3country")
labels=[price_df.index[(i*614)//6] for i in range(7)]
plt.xticks(range(0,700,100),labels)
plt.legend(loc=0)
plt.show()

#####################################################

plt.figure(figsize=(15,8))

labels=[price_df.index[(i*614)//6] for i in range(7)]
plt.xticks(range(0,700,100),labels)
plt.grid(True, color='0.7', linestyle=':', linewidth=1)


plt.plot(price_df["kospi200"],label="kospi200")
plt.plot(price_df["snp500"],label="snp500")

plt.legend(loc=0)
plt.show()

#####################################################

plt.figure(figsize=(15,8))

labels=[price_df.index[(i*614)//6] for i in range(7)]
plt.xticks(range(0,700,100),labels)
plt.grid(True, color='0.7', linestyle=':', linewidth=1)


plt.plot(price_df["kospi200"],label="kospi200")
plt.plot(price_df["nikkei225"],label="nikkei225")

plt.legend(loc=0)
plt.show()

#####################################################

plt.figure(figsize=(15,8))

labels=[price_df.index[(i*614)//6] for i in range(7)]
plt.xticks(range(0,700,100),labels)
plt.grid(True, color='0.7', linestyle=':', linewidth=1)


plt.plot(price_df["kospi200"],label="kospi200")
plt.plot(price_df["shanghai"],label="shanghai")

plt.legend(loc=0)
plt.show()

#####################################################


plt.figure(figsize=(6,6))
plt.scatter(price_df["snp500"],price_df["kospi200"],marker=".")
plt.grid(True, color='0.7', linestyle=':', linewidth=1)
plt.xlabel("snp500")
plt.ylabel("kospi200")

plt.show()

plt.figure(figsize=(6,6))
plt.scatter(price_df["nikkei225"],price_df["kospi200"],marker=".")
plt.grid(True, color='0.7', linestyle=':', linewidth=1)
plt.xlabel("nikkei225")
plt.ylabel("kospi200")

plt.show()

plt.figure(figsize=(6,6))
plt.scatter(price_df["shanghai"],price_df["kospi200"],marker=".")
plt.grid(True, color='0.7', linestyle=':', linewidth=1)
plt.xlabel("shanghai")
plt.ylabel("kospi200")

plt.show()

plt.figure(figsize=(6,6))
plt.scatter((price_df["snp500"]+price_df["nikkei225"]+price_df["shanghai"])/3,price_df["kospi200"],marker=".")
plt.grid(True, color='0.7', linestyle=':', linewidth=1)
plt.xlabel("snp500+nikkei225+shanghai / 3")
plt.ylabel("kospi200")

plt.show()

plt.figure(figsize=(6,6))
plt.scatter((price_df["nikkei225"]+price_df["shanghai"])/3,price_df["kospi200"],marker=".")
plt.grid(True, color='0.7', linestyle=':', linewidth=1)
plt.xlabel("nikkei225+shanghai / 2")
plt.ylabel("kospi200")

plt.show()

