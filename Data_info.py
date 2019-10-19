import pandas as pd

pd.set_option('display.max_columns', 10)
df = pd.read_csv('Food_Ingredients.csv',  encoding='latin-1')

print(df.info())