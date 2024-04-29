import pandas as pd

data = pd.read_csv(
    "../../data/Project-Sunroof-master/Raw Data (Sept-07-2018)/project-sunroof-census_tract-09072018.csv"
)
print(data.iloc[0])
