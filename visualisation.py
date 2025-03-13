import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from data_load import DataLoader
import os


"""
The heatmap represents the average number of rides per hour for each day of the week over the entire period covered by the data 
(Jan - November 2024)
"""
class WeeklyDemandVisualiser:
    def __init__(self, df):
        self.df = df
        self.processed_data = None

    def process_weekly_demand(self):
        # convert to datetime and extract components
        dt = pd.to_datetime(self.df['tpep_pickup_datetime'])
        self.df['day_of_week'] = dt.dt.dayofweek
        self.df['hour'] = dt.dt.hour
        self.df['date'] = dt.dt.date

        # first group by date, day_of_week, and hour to get counts per specific day
        # then calculate the mean for each day_of_week and hour combination
        weekly_demand = (self.df.groupby(['date', 'day_of_week', 'hour'])
                        .size()
                        .reset_index(name='rides')
                        .groupby(['day_of_week', 'hour'])
                        .agg({'rides': 'mean'})
                        .reset_index())
        
        weekly_demand.columns = ['Day', 'Hour', 'Demand']

        # reshape data for heatmap
        self.processed_data = weekly_demand.pivot(
            index='Day',
            columns='Hour',
            values='Demand'
        )

        # set day names for better readability
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        self.processed_data.index = day_names
        return self

    def plot_heatmap(self, title="Average Hourly Taxi Demand by Day (2024)", save_path=None):
        if self.processed_data is None:
            raise ValueError("Please process the data first using process_weekly_demand()")

        plt.figure(figsize=(15, 8))
        sns.heatmap(
            self.processed_data,
            cmap='YlGnBu',
            cbar_kws={'label': 'Average Number of Rides per Hour'}
        )

        plt.title(title)
        plt.xlabel('Hour of Day')
        plt.ylabel('Day of Week')
        
        if save_path:
            # create plots directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        return plt

"""
Pickup zone count heatmap
"""

class PUTaxiZoneVisualiser:
    def __init__(self, df):
        self.df = df
        self.processed_data = None

    def process_zone_data(self):
        # Calculate the number of unique days in the dataset
        num_days = len(pd.to_datetime(self.df['tpep_pickup_datetime']).dt.date.unique())
        
        # counts pickups per zone and creates gdf
        zone_counts = self.df.groupby('PULocationID').size().reset_index(name='pickup_count')
        # Convert to average daily pickups
        zone_counts['pickup_count'] = zone_counts['pickup_count'] / num_days
        zone_counts['PULocationID'] = zone_counts['PULocationID'].astype(str)
        
        # creates gdf using existing geometry
        geometry_df = self.df[['PULocationID', 'PUjson']].drop_duplicates()
        self.processed_data = gpd.GeoDataFrame(
            geometry_df.merge(zone_counts, on='PULocationID', how='left'),
            geometry='PUjson'
        )
        self.processed_data['pickup_count'] = self.processed_data['pickup_count'].fillna(0)
        return self

    def plot_zone_heatmap(self, title="Average Daily Taxi Pickups by Zone (2024)", save_path=None):
        if self.processed_data is None:
            raise ValueError("Process data first")

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        
        # plot choropleth
        self.processed_data.plot(
            column='pickup_count',
            ax=ax,
            legend=True,
            legend_kwds={'label': 'Average Daily Pickups'},
            cmap='cividis'
        )
        
        plt.title(title)
        plt.axis('off')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        return plt

"""
Dropoff zone count heatmap
"""
class DOTaxiZoneVisualizer:
    def __init__(self, df):
        self.df = df
        self.processed_data = None

    def process_zone_data(self):
        # calculates the number of unique days in the dataset
        num_days = len(pd.to_datetime(self.df['tpep_pickup_datetime']).dt.data.uniquie())

        # counts pikcups per one and creates gdf
        zone_counts = self.df.groupby('DOLocationID').size().reset_index(name='dropoff_count')
        # converts to average daily pickups
        zone_counts['drop'] = zone_counts['dropoff_count'] / num_days
        zone_counts['DOLocationID'] = zone_counts['DOLocationID'].astype(str)
        
        # Create GeoDataFrame using existing geometry
        geometry_df = self.df[['DOLocationID', 'DOjson']].drop_duplicates()
        self.processed_data = gpd.GeoDataFrame(
            geometry_df.merge(zone_counts, on='DOLocationID', how='left'),
            geometry='DOjson'
        )
        self.processed_data['dropoff_count'] = self.processed_data['dropoff_count'].fillna(0)
        return self

    def plot_zone_heatmap(self, title="Total Taxi Dropoffs by Zone (2024)", save_path=None):
        if self.processed_data is None:
            raise ValueError("Process data first")

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        
        # Plot choropleth
        self.processed_data.plot(
            column='dropoff_count',
            ax=ax,
            legend=True,
            legend_kwds={'label': 'Number of Dropoffs'},
            cmap='cividis'
        )
        
        plt.title(title)
        plt.axis('off')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        return plt

def main():
    # setup data loading
    file_paths = [f"./Parquet/yellow_tripdata_2024-{str(i).zfill(2)}.parquet" for i in range(1, 12)]
    geojson_path = "./tlc_taxi_zones.geojson"
    
    # load data using DataLoader
    data_loader = DataLoader(file_paths, geojson_path)
    df = data_loader.get_dataframe()
    
########################### UNCOMMENT WHICH MAP YOU WANT ###########################
     
    # create weekly demand heatmap
    #weekly_visualiser = WeeklyDemandVisualiser(df)
    #weekly_visualiser.process_weekly_demand()
    #plt_weekly = weekly_visualiser.plot_heatmap(save_path='./plots/hourly_demand_heatmap.png')
    #plt_weekly.show()
    
    # create pickups zone heatmap 
    zone_visualiser = PUTaxiZoneVisualiser(df) 
    zone_visualiser.process_zone_data()
    plt_zone = zone_visualiser.plot_zone_heatmap(save_path='./plots/zone_pickup_heatmap.png')
    plt_zone.show()

    # Create dropoffs zone heatmap
    #dropoff_visualizer = DOTaxiZoneVisualizer(df)
    #dropoff_visualizer.process_zone_data()
    #plt_dropoff = dropoff_visualizer.plot_zone_heatmap(save_path='./plots/zone_dropoff_heatmap.png')
    #plt_dropoff.show()

if __name__ == "__main__":
    main()