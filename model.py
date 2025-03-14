import pandas as pd
import geopandas as gpd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

class TaxiDemandForecast:
    def __init__(self, data_path, geojson_path):
        self.data_path = data_path
        self.geojson_path = geojson_path
        self.df = None
        self.nyc_zones = None
        self.model = None

    def load_data(self):
        """Loads taxi trip data and geojson file, filters data up to 30th November 2024"""
        print("Loading data...")
        self.df = pd.read_parquet(self.data_path)
        self.nyc_zones = gpd.read_file(self.geojson_path)
        self.nyc_zones['location_id'] = self.nyc_zones['location_id'].astype(int)

        # Filter data up to 30th November 2024
        self.df = self.df[self.df['tpep_pickup_datetime'] <= '2024-11-30 23:59:59']

        total_rows = len(self.df)
        with tqdm(total=total_rows, desc="Processing datetime") as pbar:
            self.df['tpep_pickup_datetime'] = pd.to_datetime(self.df['tpep_pickup_datetime'])
            self.df = self.df.sort_values(by=['tpep_pickup_datetime', 'PULocationID'])
            pbar.update(total_rows)

        # Print data range for debugging
        print("Data date range:", self.df['tpep_pickup_datetime'].min(), "to", self.df['tpep_pickup_datetime'].max())

    def aggregate_data(self):
        """Aggregates trips by hour and location"""
        print("Aggregating data...")
        with tqdm(total=1, desc="Aggregating trips") as pbar:
            self.df = self.df.groupby([self.df['tpep_pickup_datetime'].dt.floor('h'), 'PULocationID'])\
                .size().reset_index(name='trip_count')
            pbar.update(1)

    def create_features(self):
        """Creates time-based and lag features"""
        feature_steps = ['Time features', 'Lag features', 'Rolling average', 'Cleaning']
        with tqdm(total=len(feature_steps), desc="Creating features") as pbar:
            # Time-based features
            self.df['hour'] = self.df['tpep_pickup_datetime'].dt.hour
            self.df['dayofweek'] = self.df['tpep_pickup_datetime'].dt.dayofweek
            self.df['month'] = self.df['tpep_pickup_datetime'].dt.month
            self.df['is_weekend'] = (self.df['dayofweek'] >= 5).astype(int)
            pbar.update(1)

            # Lag features
            self.df['trip_count_lag1'] = self.df.groupby('PULocationID')['trip_count'].shift(1)
            self.df['trip_count_lag7'] = self.df.groupby('PULocationID')['trip_count'].shift(7)
            pbar.update(1)

            # Rolling average
            self.df['rolling_avg'] = self.df.groupby('PULocationID')['trip_count']\
                .rolling(window=7).mean().reset_index(0, drop=True)
            pbar.update(1)

            # Drop NaNs from lag features
            self.df.dropna(inplace=True)
            pbar.update(1)

    def split_data(self):
        """Splits data into train, validation, and test splits with adjusted dates"""
        train_end = '2024-10-31'
        val_end = '2024-11-15'
        test_end = '2024-11-30'

        train = self.df[self.df['tpep_pickup_datetime'] < train_end]
        val = self.df[(self.df['tpep_pickup_datetime'] >= train_end) & 
                      (self.df['tpep_pickup_datetime'] < val_end)]
        test = self.df[self.df['tpep_pickup_datetime'] >= val_end]

        # One-hot encode PULocationID
        train = pd.get_dummies(train, columns=['PULocationID'], prefix='loc')
        val = pd.get_dummies(val, columns=['PULocationID'], prefix='loc')
        test = pd.get_dummies(test, columns=['PULocationID'], prefix='loc')

        # Ensure all splits have the same columns
        all_columns = train.columns.union(val.columns).union(test.columns)
        train = train.reindex(columns=all_columns, fill_value=0)
        val = val.reindex(columns=all_columns, fill_value=0)
        test = test.reindex(columns=all_columns, fill_value=0)

        # Extract features and target variable
        self.X_train, self.y_train = train.drop(columns=['trip_count', 'tpep_pickup_datetime']), train['trip_count']
        self.X_val, self.y_val = val.drop(columns=['trip_count', 'tpep_pickup_datetime']), val['trip_count']
        self.X_test, self.y_test = test.drop(columns=['trip_count', 'tpep_pickup_datetime']), test['trip_count']

        # Print split sizes for debugging
        print(f"Train set size: {len(train)}")
        print(f"Validation set size: {len(val)}")
        print(f"Test set size: {len(test)}")

    def train_xgboost(self):
        """Trains an XGBoost model with early stopping"""
        print("Training XGBoost model...")
        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            early_stopping_rounds=50
        )

        with tqdm(total=1, desc="Training model") as pbar:
            self.model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                verbose=False
            )
            pbar.update(1)

    def evaluate_model(self):
        """Evaluates model on the test set"""
        if len(self.y_test) == 0:
            print("Warning: Test set is empty. Skipping evaluation.")
            return
        y_pred = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = mean_squared_error(self.y_test, y_pred) ** 0.5
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    def visualise_predictions(self):
        """Creates a heatmap of predicted average daily taxi pickups for 2025"""
        print("Generating predictions for 2025...")
        
        # Get unique locations
        locations = self.df['PULocationID'].unique()
        
        # Generate future dates for 2025 (hourly)
        future_dates = pd.date_range(start='2025-01-01 00:00:00', 
                                     end='2025-12-31 23:00:00', freq='h')
        
        # Check if future_dates is empty
        if len(future_dates) == 0:
            print("Error: No future dates generated. Check the date range.")
            return
        
        print(f"Predicting for {len(future_dates)} hours from {future_dates[0]} to {future_dates[-1]}")
        
        # Get location columns from X_train for one-hot encoding
        loc_columns = [col for col in self.X_train.columns if col.startswith('loc_')]
        
        predictions = []
        
        # Predict for each location
        for loc in tqdm(locations, desc="Predicting for locations"):
            # Get historical data for this location
            df_loc = self.df[self.df['PULocationID'] == loc].copy()
            df_loc = df_loc.sort_values('tpep_pickup_datetime')
            trip_counts = df_loc['trip_count'].tolist()
            
            # Forecast for each future date in 2025
            for future_date in future_dates:
                # Time-based features
                hour = future_date.hour
                dayofweek = future_date.dayofweek
                month = future_date.month
                is_weekend = int(dayofweek >= 5)
                
                # Lag features (using last known values from training data)
                lag1 = trip_counts[-1] if trip_counts else 0
                lag7 = trip_counts[-7] if len(trip_counts) >= 7 else 0
                rolling_avg = sum(trip_counts[-7:]) / 7 if len(trip_counts) >= 7 else \
                              (sum(trip_counts) / len(trip_counts) if trip_counts else 0)
                
                # Create feature dictionary
                features = {
                    'hour': hour,
                    'dayofweek': dayofweek,
                    'month': month,
                    'is_weekend': is_weekend,
                    'trip_count_lag1': lag1,
                    'trip_count_lag7': lag7,
                    'rolling_avg': rolling_avg
                }
                
                # Add one-hot encoded location
                for col in loc_columns:
                    features[col] = 1 if col == f'loc_{loc}' else 0
                
                # Convert to DataFrame with same columns as X_train
                feature_df = pd.DataFrame([features], columns=self.X_train.columns)
                
                # Predict trip count
                pred = self.model.predict(feature_df)[0]
                trip_counts.append(pred)
            
            # Calculate average hourly prediction for 2025
            mean_pred_hourly = sum(trip_counts[-len(future_dates):]) / len(future_dates)
            
            # Convert to average daily pickups (multiply by 24 hours)
            mean_pred_daily = mean_pred_hourly * 24
            
            predictions.append({'PULocationID': loc, 'pred_trip_count': mean_pred_daily})
        
        # Create DataFrame from predictions
        pred_df = pd.DataFrame(predictions)
        
        # Merge with geojson for plotting
        heatmap_data = self.nyc_zones.merge(pred_df, left_on='location_id', right_on='PULocationID')
        
        # Plot the heatmap
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        heatmap_data.plot(column='pred_trip_count', cmap='Reds', linewidth=0.8, 
                          edgecolor='black', legend=True, ax=ax)
        plt.title("Predicted Average Daily Taxi Pickups in NYC (2025)")
        plt.savefig('zone_pickup_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

    def run_pipeline(self):
        """Runs the entire forecasting pipeline"""
        pipeline_steps = ['Load data', 'Aggregate data', 'Create features', 
                          'Split data', 'Train model', 'Evaluate model', 
                          'Visualize predictions']
        
        with tqdm(total=len(pipeline_steps), desc="Pipeline progress") as pbar:
            self.load_data(); pbar.update(1)
            self.aggregate_data(); pbar.update(1)
            self.create_features(); pbar.update(1)
            self.split_data(); pbar.update(1)
            self.train_xgboost(); pbar.update(1)
            self.evaluate_model(); pbar.update(1)
            self.visualise_predictions(); pbar.update(1)

def main():
    data_path = "./Parquet/combined_tripdata_2024.parquet"
    geojson_path = "./tlc_taxi_zones.geojson"
    forecasting = TaxiDemandForecast(data_path, geojson_path)
    forecasting.run_pipeline()

if __name__ == "__main__":
    main()