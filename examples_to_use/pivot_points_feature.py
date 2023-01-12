"""This is a example for pivot points feature
"""
import pandas as pd
import ta
from ta.trend import PivotPointsIndicator

# Load data
df = pd.read_csv("../test/data/datas.csv", sep=",")

# Clean nan values
df = ta.utils.dropna(df)

obj=PivotPointsIndicator(df['High'],df['Low'],df['Close'],fillna=True)

df['pp']=obj.pp()
df['s1']=obj.s1()
df['s2']=obj.s2()
df['s3']=obj.s3()
df['r1']=obj.r1()
df['r2']=obj.r2()
df['r3']=obj.r3()

print(df)