import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

NAME = "nyc_taxi_demand_heatmap_2024.png"

class TaxiDataVisualizer:
    def __init__(self, parquet_path):
        self.df = pd.read_parquet(os.path.expanduser(parquet_path))
        self.prepare_data()

    def prepare_data(self):
        self.df['pickup_datetime'] = pd.to_datetime(self.df['tpep_pickup_datetime'])
        self.df['hour'] = self.df['pickup_datetime'].dt.hour
        self.df['day_of_week'] = self.df['pickup_datetime'].dt.dayofweek

    def plot_heatmap(self, output_path):
        heatmap_data = self.df.groupby(['day_of_week', 'hour']).size().unstack()
        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap_data, cmap="YlGnBu", linewidths=0.5)
        plt.xlabel("Hour of the Day")
        plt.ylabel("Day of the Week")
        plt.title("NYC Taxi Demand Heatmap (2024)")
        plt.yticks(ticks=range(7), labels=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], rotation=0)
        plt.savefig(os.path.expanduser(output_path))
        plt.show()

parquet_path = "~/Desktop/spatio-temporal/Parquet/combined_tripdata_2024.parquet"
output_path = f"~/Desktop/spatio-temporal/{NAME}"

visualizer = TaxiDataVisualizer(parquet_path)
visualizer.plot_heatmap(output_path)

