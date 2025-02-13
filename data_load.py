import pandas as pd
import numpy as np
import geopandas as gpd
import os

class DataLoader:
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.df = self.load_data()

    def load_data(self):
        dataframes = [pd.read_parquet(file_path) for file_path in self.file_paths] # loops through paths and loads them
        df = pd.concat(dataframes, ignore_index=True, sort=False) # joins parquet dfs
        df.replace(np.nan, 0, inplace=True)
        return df

    def export_to_parquet(self, output_path):
        # Ensure all columns are of the correct type
        for column in self.df.select_dtypes(include=['object']).columns:
            self.df[column] = self.df[column].astype(str)
        self.df.to_parquet(output_path)

    def get_dataframe(self):
        return self.df

file_paths = [
    "~/Desktop/spatio-temporal/Parquet/yellow_tripdata_2024-01.parquet",
    "~/Desktop/spatio-temporal/Parquet/yellow_tripdata_2024-02.parquet",
    "~/Desktop/spatio-temporal/Parquet/yellow_tripdata_2024-03.parquet",
    "~/Desktop/spatio-temporal/Parquet/yellow_tripdata_2024-04.parquet",
    "~/Desktop/spatio-temporal/Parquet/yellow_tripdata_2024-05.parquet",
    "~/Desktop/spatio-temporal/Parquet/yellow_tripdata_2024-06.parquet",
    "~/Desktop/spatio-temporal/Parquet/yellow_tripdata_2024-07.parquet",
    "~/Desktop/spatio-temporal/Parquet/yellow_tripdata_2024-08.parquet",
    "~/Desktop/spatio-temporal/Parquet/yellow_tripdata_2024-09.parquet",
    "~/Desktop/spatio-temporal/Parquet/yellow_tripdata_2024-10.parquet",
    "~/Desktop/spatio-temporal/Parquet/yellow_tripdata_2024-11.parquet"
]

data_loader = DataLoader(file_paths)
df = data_loader.get_dataframe()
#print(df.tail(20))

# uncomment to save merged df to csv
#output_path = "~/Desktop/spatio-temporal/combined_tripdata_2024.parquet"
#data_loader.export_to_parquet(output_path)

geojson_path = os.path.expanduser("./tlc_taxi_zones.geojson")
taxi_zones = gpd.read_file(geojson_path)

# ensures location_id is treated as an integer
taxi_zones["location_id"] = taxi_zones["location_id"].astype(int)

# merges for pickup location
df = df.merge(taxi_zones[["location_id", "geometry", "zone", "borough"]],
              left_on = "PULocationID", right_on = "location_id", how = "left", suffixed = ("", "_PU"))

