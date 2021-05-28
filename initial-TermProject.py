import pandas as pd
import numpy as np

data=pd.read_csv("C:/Users/nahye/DataScience/WA_Fn-UseC_-Telco-Customer-Churn-2.csv")

print("----- Data shape\n")
print(data.shape)

print("\n\n----- Data features\n")
print(data.columns.values)

print("\n\n----- Data head\n")
print(data.head())

print("\n\n----- Data info\n")
print(data.info())

#statistical description
print("\n\n----- Data sum(axis=0)\n")
print(data.sum())

print("\n\n----- Data sum(axis=1)\n")
print(data.sum(1))

print("\n\n----- Data mean\n")
print(data.mean())

print("\n\n----- Data std\n")
print(data.std())

#칼럼의 개체수
print("\n\n----- Data count\n")
print(data.count())

print("\n\n----- Data median\n")
print(data.median())

#최빈수
print("\n\n----- Data mode\n")
print(data.mode())

print("\n\n----- Data min\n")
print(data.min())

print("\n\n----- Data max\n")
print(data.max())

print("\n\n----- Data describe\n")
print(data.describe())

#descriptive statistics 메소드 들 중에서 prod,cumsum,cumprod,abs 빼고 다 적었습니당
# prod - 오버플로우발생
# cumsum - row기반 누적합, 데이터 타입때문에 에러, 쓸거면 데이터 타입별로 나눠서 사용해야 될거같아요
# cumprod - row기반 누적곱
# abs - 단일 정수형변수에대한 절대값반환, 값 하나씩만 받을수있어서 안썼어요
