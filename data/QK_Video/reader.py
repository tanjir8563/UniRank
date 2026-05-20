import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 2000)

df1 = pd.read_parquet('./item_info.parquet')
print("df1 前5行：")
print(df1.head())

df2 = pd.read_parquet('./user_info.parquet')
print("df2 前5行：")
print(df2.head())

df3 = pd.read_parquet('./train.parquet')
print("\ndf3 前5行：")
print(df3.head())