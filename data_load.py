import pandas as pd
import numpy as np
import geopandas as gpd
import os

class DataLoader:
    def __init__(self, file_paths, geojson_path):
        self.file_paths = file_paths
        self.geojson_path = geojson_path
        self.df = self._load_and_process_data()

    def _load_geojson(self):
        taxi_zones = gpd.read_file(self.geojson_path)
        taxi_zones['location_id'] = taxi_zones['location_id'].astype(str)
        return dict(zip(taxi_zones['location_id'], taxi_zones['geometry']))

    def _load_and_process_data(self):
        # Load parquet files
        dataframes = [pd.read_parquet(file_path) for file_path in self.file_paths]
        df = pd.concat(dataframes, ignore_index=True, sort=False)
        df.replace(np.nan, 0, inplace=True)

        # Load and process geojson
        geometry_lookup = self._load_geojson()
        
        # Add geometry to dataframe
        df['PULocationID'] = df['PULocationID'].astype(str)
        df['DOLocationID'] = df['DOLocationID'].astype(str)
        df['PUjson'] = df['PULocationID'].map(geometry_lookup)
        df['DOjson'] = df['DOLocationID'].map(geometry_lookup)
        
        return df

    def get_dataframe(self):
        return self.df

def main():
    file_paths = [
        f"./Parquet/yellow_tripdata_2024-{str(i).zfill(2)}.parquet" 
        for i in range(1, 12)
    ]

    # Load and process all data in one go
    data_loader = DataLoader(file_paths, "./tlc_taxi_zones.geojson")
    df_with_geo = data_loader.get_dataframe()

if __name__ == "__main__":
    main()
