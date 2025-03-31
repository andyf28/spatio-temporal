import pandas as pd
import geopandas as gpd
import xgboost as xgb
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
from data_load import DataLoader

class TaxiDemandForecast:
    def __init__(self, file_paths, geojson_path):
        self.file_paths = file_paths
        self.geojson_path = geojson_path
        self.df = None
        self.nyc_zones = None
        self.model = None
        self.allowed_location_ids = {
            "116", "42", "152", "166", "41", "74", "24", "75", "151", "238", "239", "236", "263", "262",
            "43", "143", "142", "237", "141", "140", "202", "50", "48", "163", "230", "161", "162", "229",
            "246", "68", "100", "186", "90", "164", "170", "233", "137", "234", "107", "224", "158", "249",
            "113", "114", "79", "4", "125", "211", "144", "148", "232", "231", "45", "13", "261", "87", "209",
            "1288"
        }

    def load_data(self):
        
        """
        Loads taxi trip data using DataLoader and geojson file, filters up to 30th November 2024
        """

        print("Loading data...")
        data_loader = DataLoader(self.file_paths, self.geojson_path)
        self.df = data_loader.get_dataframe()
        
        total_rows = len(self.df)
        with tqdm(total=total_rows, desc="Processing datetime") as pbar:
            self.df['tpep_pickup_datetime'] = pd.to_datetime(self.df['tpep_pickup_datetime'])
            pbar.update(total_rows)

        # filters data between 2024-01-01 00:00:00 and 2024-11-30 23:59:59 
        self.df = self.df[ 
            (self.df['tpep_pickup_datetime'] >= '2024-01-01 00:00:00') & 
            (self.df['tpep_pickup_datetime'] <= '2024-11-30 23:59:59') 
        ]
        self.df = self.df.sort_values(by=['tpep_pickup_datetime', 'PULocationID'])

        self.nyc_zones = gpd.read_file(self.geojson_path)
        self.nyc_zones['location_id'] = self.nyc_zones['location_id'].astype(str)
        self.nyc_zones = self.nyc_zones[self.nyc_zones['location_id'].isin(self.allowed_location_ids)]

        print("Data date range:", self.df['tpep_pickup_datetime'].min(), "to", self.df['tpep_pickup_datetime'].max())

    def aggregate_data(self):
       
        """
        Aggregates trips by day and location
        """

        # aggregrates by day and loc
        print("Aggregating data...")
        with tqdm(total=1, desc="Aggregating trips") as pbar:
            self.df = self.df.groupby([self.df['tpep_pickup_datetime'].dt.date, 'PULocationID'])\
                .size().reset_index(name='trip_count')
            self.df['tpep_pickup_datetime'] = pd.to_datetime(self.df['tpep_pickup_datetime'])
            pbar.update(1)

    def create_features(self):
        """Creates time-based and lag features for daily data"""
        feature_steps = ['Time features', 'Lag features', 'Rolling average', 'Cleaning']
        with tqdm(total=len(feature_steps), desc="Creating features") as pbar:
            self.df['dayofweek'] = self.df['tpep_pickup_datetime'].dt.dayofweek
            self.df['month'] = self.df['tpep_pickup_datetime'].dt.month
            self.df['is_weekend'] = (self.df['dayofweek'] >= 5).astype(int)
            pbar.update(1)

            # calculate mean trip count per location to use for imputation
            location_means = self.df.groupby('PULocationID')['trip_count'].mean().to_dict()
            
            self.df['trip_count_lag1'] = self.df.groupby('PULocationID')['trip_count'].shift(1)
            self.df['trip_count_lag7'] = self.df.groupby('PULocationID')['trip_count'].shift(7)

            # impute missing lag values with the location's mean
            self.df['trip_count_lag1'] = self.df.apply(
                lambda row: location_means.get(row['PULocationID'], 0) if pd.isna(row['trip_count_lag1']) else row['trip_count_lag1'], axis=1)
            self.df['trip_count_lag7'] = self.df.apply(
                lambda row: location_means.get(row['PULocationID'], 0) if pd.isna(row['trip_count_lag7']) else row['trip_count_lag7'], axis=1)
            pbar.update(1)

            self.df['rolling_avg'] = self.df.groupby('PULocationID')['trip_count']\
                .rolling(window=7).mean().reset_index(0, drop=True)
            
            # impute missing rolling averages with the location's mean
            self.df['rolling_avg'] = self.df.apply(
                lambda row: location_means.get(row['PULocationID'], 0) if pd.isna(row['rolling_avg']) else row['rolling_avg'], axis=1)
            pbar.update(1)

            self.df.dropna(inplace=True)
            pbar.update(1)

    def split_data(self):
       
        """
        Splits data into train, validation, and test splits with adjusted dates
        """

        train_end = '2024-10-31'
        val_end = '2024-11-15'
        test_end = '2024-11-30'

        # sets split end dates
        train = self.df[self.df['tpep_pickup_datetime'] < train_end]
        val = self.df[(self.df['tpep_pickup_datetime'] >= train_end) & 
                      (self.df['tpep_pickup_datetime'] < val_end)]
        test = self.df[self.df['tpep_pickup_datetime'] >= val_end]

        # one-hot encodes location ids
        train = pd.get_dummies(train, columns=['PULocationID'], prefix='loc')
        val = pd.get_dummies(val, columns=['PULocationID'], prefix='loc')
        test = pd.get_dummies(test, columns=['PULocationID'], prefix='loc')

        # aligns columns across datasets
        all_columns = train.columns.union(val.columns).union(test.columns)
        train = train.reindex(columns=all_columns, fill_value=0)
        val = val.reindex(columns=all_columns, fill_value=0)
        test = test.reindex(columns=all_columns, fill_value=0)

        # identifies feature columns and target
        feature_cols = [col for col in all_columns if col not in ['trip_count', 'tpep_pickup_datetime']]

        # converts to numeric handling any potential object columns
        def safe_convert(df):
            numeric_df = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
            return numeric_df

        # convert to CuPy arrays (used chatgpt for code to convert to cupy arrays because model was sending data from cpu to gpu when predicting making it too slow. cupy arrays keep data on gpu)
        self.X_train = cp.asarray(safe_convert(train).values, dtype=cp.float32)
        self.y_train = cp.asarray(train['trip_count'].values, dtype=cp.float32)
        self.X_val = cp.asarray(safe_convert(val).values, dtype=cp.float32)
        self.y_val = cp.asarray(val['trip_count'].values, dtype=cp.float32)
        self.X_test = cp.asarray(safe_convert(test).values, dtype=cp.float32)
        self.y_test = cp.asarray(test['trip_count'].values, dtype=cp.float32)

        # keep feature column names for later
        self.feature_columns = feature_cols

        print(f"Train set size: {len(train)}")
        print(f"Validation set size: {len(val)}")
        print(f"Test set size: {len(test)}")

    def train_xgboost(self): # used this for xgboost code: https://www.datacamp.com/tutorial/xgboost-in-python

        """
        Trains an XGBoost model with early stopping using CuPy arrays
        """

        # hyperparameters
        print("Training XGBoost model...")
        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators = 3000,
            learning_rate = 0.05,
            max_depth = 10,
            subsample = 0.8,
            colsample_bytree = 0.8,
            early_stopping_rounds = 50,
            device = "cuda"  # gpu support
        )

        with tqdm(total=1, desc="Training model") as pbar:
            # converts cupy arrays to numpy to use xgboost on gpu instead of cpu and gpu leading to bottlenecking
            self.model.fit(
                cp.asnumpy(self.X_train), cp.asnumpy(self.y_train),
                eval_set = [(cp.asnumpy(self.X_val), cp.asnumpy(self.y_val))],
                verbose = 2
            )
            pbar.update(1)

    def evaluate_model(self):

        """
        Evaluates model on the test set
        """

        if len(self.y_test) == 0:
            print("Warning: Test set is empty. Skipping evaluation.")
            return
        
        # convert cupy to numpy 
        y_test_np = cp.asnumpy(self.y_test)
        y_pred = self.model.predict(cp.asnumpy(self.X_test))
        
        # model evals
        mae = mean_absolute_error(y_test_np, y_pred)
        rmse = mean_squared_error(y_test_np, y_pred) ** 0.5
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    def visualise_predictions(self):
        """Creates a heatmap of predicted average daily taxi pickups for 2025 and saves to CSV"""
        print("Generating predictions for 2025...")
        
        locations = [loc for loc in self.df['PULocationID'].unique() if loc in self.allowed_location_ids]
        
        future_dates = pd.date_range(start='2025-01-01', 
                                    end='2025-12-31', freq='D')
        
        if len(future_dates) == 0:
            print("Error: No future dates generated. Check the date range.")
            return
        
        print(f"Predicting for {len(future_dates)} days from {future_dates[0]} to {future_dates[-1]}")
        
        predictions = []

        historical_means = self.df.groupby('PULocationID')['trip_count'].mean().to_dict() # calculates historical means for each location
        
        for loc in tqdm(locations, desc="Predicting for locations"):
            df_loc = self.df[self.df['PULocationID'] == loc].copy()
            df_loc = df_loc.sort_values('tpep_pickup_datetime')
            trip_counts = df_loc['trip_count'].tolist()
            
            for future_date in future_dates:
                dayofweek = future_date.dayofweek
                month = future_date.month
                is_weekend = int(dayofweek >= 5)
                
                # uses historical data if available, otherwise uses historical mean
                lag1 = trip_counts[-1] if trip_counts else historical_means.get(loc, 0)
                lag7 = trip_counts[-7] if len(trip_counts) >= 7 else historical_means.get(loc, 0)
                rolling_avg = sum(trip_counts[-7:]) / 7 if len(trip_counts) >= 7 else historical_means.get(loc, 0)
                
                # construct feature vector matching training data
                feature_dict = {
                    'dayofweek': dayofweek,
                    'month': month,
                    'is_weekend': is_weekend,
                    'trip_count_lag1': lag1,
                    'trip_count_lag7': lag7,
                    'rolling_avg': rolling_avg
                }
                # adds one-hot encoded location features
                for col in self.feature_columns:
                    if col.startswith('loc_'):
                        feature_dict[col] = 1 if col == f'loc_{loc}' else 0
                
                # converts to array in the correct order
                feature_vector = [feature_dict[col] for col in self.feature_columns]
                
                # uses cupy for conversion and prediction
                feature_gpu = cp.asarray(feature_vector, dtype=cp.float32).reshape(1, -1)
                pred = self.model.predict(cp.asnumpy(feature_gpu))[0]
                trip_counts.append(pred)
            
            mean_pred_daily = sum(trip_counts[-len(future_dates):]) / len(future_dates)
            
            predictions.append({'PULocationID': loc, 'pred_trip_count': mean_pred_daily})
        
        pred_df = pd.DataFrame(predictions)
        
        csv_path = './outputs/predicted_taxi_pickups_2025.csv'
        pred_df.to_csv(csv_path, index=False)
        print(f"Predictions saved to {csv_path}")
        
        heatmap_data = self.nyc_zones.merge(pred_df, left_on='location_id', right_on='PULocationID')
        
        # plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        heatmap_data.plot(column='pred_trip_count', cmap='Reds', linewidth=0.8, 
                        edgecolor='black', legend=True, ax=ax, 
                        vmin=0, vmax=5000, 
                        legend_kwds={'label': "Average Daily Pickups"})
        plt.title("Predicted Average Daily Taxi Pickups by Zone (2025)")
        plt.axis('off')
        plt.savefig('./plots/zone_pickup_heatmap_predicted.png', dpi=300, bbox_inches='tight')
        plt.show()


    def run_pipeline(self):
        
        """
        Runs the entire forecasting pipeline
        """

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
    file_paths = [f"./Parquet/yellow_tripdata_2024-{str(i).zfill(2)}.parquet" for i in range(1, 12)]
    geojson_path = "./tlc_taxi_zones.geojson"
    forecasting = TaxiDemandForecast(file_paths, geojson_path)
    forecasting.run_pipeline()

if __name__ == "__main__":
    main()