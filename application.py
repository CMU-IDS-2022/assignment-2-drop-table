import pandas as pd

#df = pd.read_excel("EADA_2020.xlsx",engine = "openpyxl")
df = pd.read_csv("health-raw-2021.csv",skiprows=3)
#print(df)

print(df.head(10))

#print(df.columns)