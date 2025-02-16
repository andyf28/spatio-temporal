import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_load import DataLoader
import os


"""
The heatmap represents the average number of rides per hour for each day of the week over the entire period covered by the data 
(Jan - November 2024)
"""
class WeeklyDemandVisualizer:
    def __init__(self, df):
        self.df = df
        self.processed_data = None

    def process_weekly_demand(self):
        # Convert to datetime and extract components
        dt = pd.to_datetime(self.df['tpep_pickup_datetime'])
        self.df['day_of_week'] = dt.dt.dayofweek
        self.df['hour'] = dt.dt.hour
        self.df['date'] = dt.dt.date

        # First group by date, day_of_week, and hour to get counts per specific day
        # Then calculate the mean for each day_of_week and hour combination
        weekly_demand = (self.df.groupby(['date', 'day_of_week', 'hour'])
                        .size()
                        .reset_index(name='rides')
                        .groupby(['day_of_week', 'hour'])
                        .agg({'rides': 'mean'})
                        .reset_index())
        
        weekly_demand.columns = ['Day', 'Hour', 'Demand']

        # Reshape data for heatmap
        self.processed_data = weekly_demand.pivot(
            index='Day',
            columns='Hour',
            values='Demand'
        )

        # Set day names for better readability
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
            # Create plots directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        return plt

def main():
    # Setup data loading
    file_paths = [f"./Parquet/yellow_tripdata_2024-{str(i).zfill(2)}.parquet" for i in range(1, 12)]
    
    # Load data using DataLoader
    data_loader = DataLoader(file_paths, "./tlc_taxi_zones.geojson")
    df = data_loader.get_dataframe()
    
    # Create visualizer and generate heatmap
    visualizer = WeeklyDemandVisualizer(df)
    visualizer.process_weekly_demand()
    
    # Save and display plot
    plt = visualizer.plot_heatmap(save_path='./plots/weekly_demand_heatmap.png')
    plt.show()

if __name__ == "__main__":
    main()
