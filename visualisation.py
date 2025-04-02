import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from data_load import DataLoader
import os
from shapely.geometry import LineString
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

"""
The heatmap represents the average number of rides per hour for each day of the week over the entire period covered by the data 
(Jan - November 2024)
"""
class WeeklyDemandVisualiser:
    def __init__(self, df):
        self.df = df
        self.processed_data = None

    def process_weekly_demand(self):
        # converts to datetime and extract components
        dt = pd.to_datetime(self.df['tpep_pickup_datetime'])
        self.df['day_of_week'] = dt.dt.dayofweek
        self.df['hour'] = dt.dt.hour
        self.df['date'] = dt.dt.date

        # group by date, day_of_week, and hour to get counts per specific day,
        # then calculate the mean for each day_of_week and hour combination
        weekly_demand = (self.df.groupby(['date', 'day_of_week', 'hour'])
                        .size()
                        .reset_index(name='rides')
                        .groupby(['day_of_week', 'hour'])
                        .agg({'rides': 'mean'})
                        .reset_index())
        
        weekly_demand.columns = ['Day', 'Hour', 'Demand']

        # reshapes data for heatmap
        self.processed_data = weekly_demand.pivot(
            index = 'Day',
            columns = 'Hour',
            values = 'Demand'
        )

        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        self.processed_data.index = day_names
        return self

    def plot_heatmap(self, title="Average Hourly Taxi Demand by Day (2024)", save_path=None):
        if self.processed_data is None:
            raise ValueError("Please process the data first using process_weekly_demand()")

        plt.figure(figsize=(15, 8))
        sns.heatmap(
            self.processed_data,
            cmap = 'YlGnBu',
            cbar_kws = {'label': 'Average Number of Rides per Hour'}
        )

        plt.title(title)
        plt.xlabel('Hour of Day')
        plt.ylabel('Day of Week')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches = 'tight', dpi=300)
            
        return plt

class PUTaxiZoneVisualiser:
    def __init__(self, df):
        self.df = df
        self.processed_data = None

    def process_zone_data(self, metric='average'):
        """
        Process pickup zone data to calculate either average daily or total pickups.

        Parameters:
        - metric (str): 'average' for average daily pickups, 'total' for total pickups.
        """
        # calculate number of unique days in dataset
        num_days = len(pd.to_datetime(self.df['tpep_pickup_datetime']).dt.date.unique())
        
        # count pickups per zone
        zone_counts = self.df.groupby('PULocationID').size().reset_index(name='pickup_count')
        
        # adjust counts based on metric
        if metric == 'average':
            zone_counts['pickup_count'] = zone_counts['pickup_count'] / num_days

        # if metric is 'total', leave pickup_count as is (total count)
        zone_counts['PULocationID'] = zone_counts['PULocationID'].astype(str)
        
        # create gdf
        geometry_df = self.df[['PULocationID', 'PUjson']].drop_duplicates()
        self.processed_data = gpd.GeoDataFrame(
            geometry_df.merge(zone_counts, on = 'PULocationID', how = 'left'),
            geometry = 'PUjson'
        )
        self.processed_data['pickup_count'] = self.processed_data['pickup_count'].fillna(0)
        return self

    def plot_zone_heatmap(self, metric, title=None, save_path=None):
        """
        Plot a heatmap of pickup zones based on the specified metric.

        Parameters:
        - metric (str): 'average' or 'total' to determine title and legend.
        - title (str, optional): Custom title for the plot.
        - save_path (str, optional): Path to save the plot.
        """
        if self.processed_data is None:
            raise ValueError("Process data first")

        # set default title based on metric if not provided
        if title is None:
            title = "Average Daily Taxi Pickups by Zone (2024)" if metric == 'average' else "Total Taxi Pickups by Zone (2024)"

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        
        # set legend label with metric
        legend_label = 'Average Daily Pickups' if metric == 'average' else 'Total Pickups'
        
        # plot heatmap
        self.processed_data.plot(
            column = 'pickup_count',
            ax = ax,
            legend=True,
            legend_kwds = {'label': legend_label},
            cmap = 'Reds',
            linewidth = 0.8,
            edgecolor = 'black'
        )
        
        plt.title(title)
        plt.axis('off')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        return plt


class DOTaxiZoneVisualizer:
    def __init__(self, df):
        self.df = df
        self.processed_data = None

    def process_zone_data(self, metric='average'):
        """
        Process dropoff zone data to calculate either average daily or total dropoffs.

        Parameters:
        - metric (str): 'average' for average daily dropoffs, 'total' for total dropoffs.
        """
        # calculate number of unique days in dataset
        num_days = len(pd.to_datetime(self.df['tpep_pickup_datetime']).dt.date.unique())
        
        # count dropoffs per zone
        zone_counts = self.df.groupby('DOLocationID').size().reset_index(name='dropoff_count')
        
        # adjust counts based on metric
        if metric == 'average':
            zone_counts['dropoff_count'] = zone_counts['dropoff_count'] / num_days

        # if metric is 'total', leave dropoff_count as is (total count)
        zone_counts['DOLocationID'] = zone_counts['DOLocationID'].astype(str)
        
        # create gdf
        geometry_df = self.df[['DOLocationID', 'DOjson']].drop_duplicates()
        self.processed_data = gpd.GeoDataFrame(
            geometry_df.merge(zone_counts, on = 'DOLocationID', how = 'left'),
            geometry = 'DOjson'
        )
        self.processed_data['dropoff_count'] = self.processed_data['dropoff_count'].fillna(0)
        return self

    def plot_zone_heatmap(self, metric, title=None, save_path=None):
        """
        Plot a heatmap of dropoff zones based on the specified metric.

        Parameters:
        - metric (str): 'average' or 'total' to determine title and legend.
        - title (str, optional): Custom title for the plot.
        - save_path (str, optional): Path to save the plot.
        """
        if self.processed_data is None:
            raise ValueError("Process data first")

        # set title with metric
        if title is None:
            title = "Average Daily Taxi Dropoffs by Zone (2024)" if metric == 'average' else "Total Taxi Dropoffs by Zone (2024)"

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        
        # set legend label with metric
        legend_label = 'Average Daily Dropoffs' if metric == 'average' else 'Total Dropoffs'
        
        # plot heatmap
        self.processed_data.plot(
            column = 'dropoff_count',
            ax = ax,
            legend  =True,
            legend_kwds = {'label': legend_label},
            cmap = 'Reds',
            linewidth = 0.8,
            edgecolor = 'black'
        )
        
        plt.title(title)
        plt.axis('off')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok = True)
            plt.savefig(save_path, bbox_inches = 'tight', dpi = 300)
            
        return plt
    
class FlowMapVisualiser:
    def __init__(self, df):
        """
        Initialises the FlowMapVisualiser with a df

        Parameters:
        - df
        """
        self.df = df
        self.processed_data = None
        self.zone_gdf = None

    def process_flow_data(self, top_n = 100, min_trips = 50000):
        """
        Process the data to calculate the trip volumes between pickup and dropoff zones.
        """
        # extract unique zones and their geometries
        zone_geometries = self.df[['PULocationID', 'PUjson']].drop_duplicates().rename(columns = {'PULocationID': 'zone_id', 'PUjson': 'geometry'})
        self.zone_gdf = gpd.GeoDataFrame(zone_geometries, geometry = 'geometry')
        self.zone_gdf['centroid'] = self.zone_gdf.geometry.centroid

        # group trips by pickup and drop-off locations
        flow_counts = self.df.groupby(['PULocationID', 'DOLocationID']).size().reset_index(name = 'trip_count')

        # apply minimum trip threshold
        flow_counts = flow_counts[flow_counts['trip_count'] >= min_trips]

        # merge centroids for pickup and drop-off zones
        flow_counts = flow_counts.merge(self.zone_gdf[['zone_id', 'centroid']], left_on='PULocationID', right_on='zone_id', how='left').rename(columns={'centroid': 'pu_centroid'}).drop('zone_id', axis=1)
        flow_counts = flow_counts.merge(self.zone_gdf[['zone_id', 'centroid']], left_on='DOLocationID', right_on='zone_id', how='left').rename(columns={'centroid': 'do_centroid'}).drop('zone_id', axis=1)

        # create LineStrings, excluding self-loops
        flow_counts['geometry'] = flow_counts.apply(
            lambda row: LineString([row['pu_centroid'], row['do_centroid']]) if row['PULocationID'] != row['DOLocationID'] and pd.notna(row['pu_centroid']) and pd.notna(row['do_centroid']) else None, 
            axis=1
        )
        flow_counts = flow_counts.dropna(subset=['geometry'])

        # create gdf
        flow_gdf = gpd.GeoDataFrame(flow_counts, geometry='geometry')
        flow_gdf.crs = self.zone_gdf.crs

        # filters to top N flows
        if top_n is not None:
            flow_gdf = flow_gdf.nlargest(top_n, 'trip_count')

        # computes line widths
        max_count = flow_gdf['trip_count'].max()
        min_count = flow_gdf['trip_count'].min()
        if max_count > min_count:
            flow_gdf['linewidth'] = 0.5 + 4.5 * (flow_gdf['trip_count'] - min_count) / (max_count - min_count)
        else:
            flow_gdf['linewidth'] = 1.0

        # simple edge bundling: Group flows by discretizing start and end points # used chatgpt for debugging
        flow_gdf['pu_x'] = flow_gdf['pu_centroid'].apply(lambda p: round(p.x, 4))
        flow_gdf['pu_y'] = flow_gdf['pu_centroid'].apply(lambda p: round(p.y, 4))
        flow_gdf['do_x'] = flow_gdf['do_centroid'].apply(lambda p: round(p.x, 4))
        flow_gdf['do_y'] = flow_gdf['do_centroid'].apply(lambda p: round(p.y, 4))

        # groups by discretised coordinates and sum trip counts
        bundled_flows = flow_gdf.groupby(['pu_x', 'pu_y', 'do_x', 'do_y']).agg({
            'trip_count': 'sum',
            'linewidth': 'mean',
            'pu_centroid': 'first',
            'do_centroid': 'first'
        }).reset_index()

        # creates a series of geometries separately
        geometry_series = bundled_flows.apply(
            lambda row: LineString([row['pu_centroid'], row['do_centroid']]), axis=1
        )

        # assigns the geometry series to the df
        bundled_flows['geometry'] = geometry_series

        # convert to gdf
        self.processed_data = gpd.GeoDataFrame(bundled_flows, geometry='geometry')
        self.processed_data.crs = self.zone_gdf.crs
        return self
    
    def plot_flow_map(self, title = "Taxi Trip Flow Map (2024)", save_path = None):
        """
        Plots flow map of trips from pickup to dropoff locations.
        """
        if self.processed_data is None:
            raise ValueError("Process data first.")
        
        if self.zone_gdf is None:
            raise ValueError('Zone GDF not set.')
        
        fig, ax = plt.subplots(1, 1, figsize = (15, 15))

        # plots zone boundaries as base map
        self.zone_gdf.plot(ax=ax, color='lightgrey', edgecolor='black', alpha=0.5)

        # prepares data for LineCollection
        lines = [list(line.coords) for line in self.processed_data.geometry]
        linewidths = self.processed_data['linewidth']
        trip_counts = self.processed_data['trip_count']

        # Create LineCollection with color mapping
        norm = Normalize(vmin=trip_counts.min(), vmax=trip_counts.max())
        cmap = plt.cm.cividis
        line_collection = LineCollection(lines, linewidths=linewidths, colors=cmap(norm(trip_counts)), alpha=0.5)
        ax.add_collection(line_collection)

        # add colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Trip Count')

        # customise plot
        plt.title(title)
        plt.axis('off')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        return plt

def main():
    # data loading
    file_paths = [f"./Parquet/yellow_tripdata_2024-{str(i).zfill(2)}.parquet" for i in range(1, 12)]
    geojson_path = "./tlc_taxi_zones.geojson"
    
    data_loader = DataLoader(file_paths, geojson_path)
    df = data_loader.get_dataframe()
    
    # user input
    print("Available visualisations:")
    print("1. Temporal Heatmap - Average hourly taxi demand by day")
    print("2. Pickup Zone Heatmap")
    print("3. Dropoff Zone Heatmap")
    print("4. Flow Map")
    
    choice = input("Enter your choice (e.g., 'Temporal Heatmap', 'Pickup Zone Heatmap', 'Dropoff Zone Heatmap', 'Flow Map', '1', '2', '3', '4'): ").strip().lower()

    if choice in ["temporal heatmap", "1"]:
        print("Plotting Temporal Heatmap...")
        weekly_visualiser = WeeklyDemandVisualiser(df)
        weekly_visualiser.process_weekly_demand()
        plt_weekly = weekly_visualiser.plot_heatmap(save_path='./plots/hourly_demand_heatmap.png')
        plt_weekly.show()
    
    elif choice in ["pickup zone heatmap", "2"]:
        print("Do you want to plot average daily pickups or total pickups? Enter 'average' or 'total':")
        metric = input().strip().lower()
        if metric not in ['average', 'total']:
            print("Invalid metric. Please enter 'average' or 'total'.")
        else:
            print(f"Plotting Pickup Zone Heatmap with {metric} pickups...")
            zone_visualiser = PUTaxiZoneVisualiser(df)
            zone_visualiser.process_zone_data(metric=metric)
            save_path = f'./plots/zone_pickup_heatmap_{metric}.png'
            plt_zone = zone_visualiser.plot_zone_heatmap(metric=metric, save_path=save_path)
            plt_zone.show()

    elif choice in ["dropoff zone heatmap", "3"]:
        print("Do you want to plot average daily dropoffs or total dropoffs? Enter 'average' or 'total':")
        metric = input().strip().lower()
        if metric not in ['average', 'total']:
            print("Invalid metric. Please enter 'average' or 'total'.")
        else:
            print(f"Plotting Dropoff Zone Heatmap with {metric} dropoffs...")
            dropoff_visualizer = DOTaxiZoneVisualizer(df)
            dropoff_visualizer.process_zone_data(metric=metric)
            save_path = f'./plots/zone_dropoff_heatmap_{metric}.png'
            plt_dropoff = dropoff_visualizer.plot_zone_heatmap(metric=metric, save_path=save_path)
            plt_dropoff.show()
    
    elif choice in ["flow map", "4"]:
        print("Plotting Flow Map...")
        flow_map_visualiser = FlowMapVisualiser(df)
        flow_map_visualiser.process_flow_data(top_n = 100, min_trips = 45000)
        save_path = './plots/flow_map.png'
        plt_flow = flow_map_visualiser.plot_flow_map(save_path = save_path)
        plt_flow.show()
        
    else:
        print("Invalid choice. Please rerun the file and select a valid option.")

if __name__ == "__main__":
    main()