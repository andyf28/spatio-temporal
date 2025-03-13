import pandas as pd
import geopandas as gpd
import xgboost as xgb 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

# MAKE SURE TO ENALBE CUDE ON XGBOOST MODEL OR MAKE XGBOOST RUN ON GPU

class TaxiDemandForecast:
    def __init__(self, data_path, geojson_path): 
        self.data_path = data_path
        self.geojson_path = geojson_path
        self.df = None
        self.nyc_zones = None
        self.model = None

    def load_data(self):
        """Loads taxi trip data and geojson file"""
        print("Loading data...")
        self.df = pd.read_parquet(self.data_path) #trip data
        self.nyc_zones = gpd.read_file(self.geojson_path) #taxi zones
        self.nyc_zones['location_id'] = self.nyc_zones['location_id'].astype(int)

        total_rows = len(self.df)
        with tqdm(total=total_rows, desc="Processing datetime") as pbar:
            self.df['tpep_pickup_datetime'] = pd.to_datetime(self.df['tpep_pickup_datetime'])
            self.df = self.df.sort_values(by=['tpep_pickup_datetime', 'PULocationID'])
            pbar.update(total_rows)

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

            # time-based features
            self.df['hour'] = self.df['tpep_pickup_datetime'].dt.hour 
            self.df['dayofweek'] = self.df['tpep_pickup_datetime'].dt.dayofweek 
            self.df['month'] = self.df['tpep_pickup_datetime'].dt.month
            self.df['is_weekend'] = (self.df['dayofweek'] >= 5).astype(int)
            pbar.update(1)

            # lag features
            self.df['trip_count_lag1'] = self.df.groupby('PULocationID')['trip_count'].shift(1)
            self.df['trip_count_lag7'] = self.df.groupby('PULocationID')['trip_count'].shift(7)
            pbar.update(1)

            # rolling average
            self.df['rolling_avg'] = self.df.groupby('PULocationID')['trip_count'].rolling(window = 7).mean().reset_index(0, drop = True)
            pbar.update(1)

            # drops NaNs from lag features
            self.df.dropna(inplace = True)
            pbar.update(1)

    def split_data(self):
        """Splits data into train, validation and test splits"""
        train_end = '2024-10-31'
        val_end = '2024-12-15'
        tests_end = '2024-12-31'

        train = self.df[self.df['tpep_pickup_datetime'] < train_end]
        val = self.df[(self.df['tpep_pickup_datetime'] >= train_end) & (self.df['tpep_pickup_datetime'] < val_end)]
        test = self.df[self.df['tpep_pickup_datetime'] >= val_end]

        # extracts features and target variable
        self.X_train, self.y_train = train.drop(columns=['trip_count', 'tpep_pickup_datetime']), train['trip_count']
        self.X_val, self.y_val = val.drop(columns = ['trip_count', 'tpep_pickup_datetime']), val['trip_count']
        self.X_test, self.y_test = test.drop(columns = ['trip_count', 'tpep_pickup_datetime']), test['trip_count']

    def train_xgboost(self):
        """Trains an XGBoost model with early stopping to prevent overfitting"""
        print("Training XGBoost model...")
        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            early_stopping_rounds=50  # early stopping to constructor
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
        y_pred = self.model.predict(self.X_test)  # Remove this parameter

        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = mse ** 0.5

        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    def visualise_predictions(self):
        """Creates a heatmap of predicted taxi demand"""
        # predicts on test set
        self.df.loc[self.df['tpep_pickup_datetime'] >= '2024-12-15', 'pred_trip_count'] = self.model.predict(self.X_test)

        # aggregates predictions by PULocationID
        heatmap_data = self.nyc_zones.merge(self.df.groupby('PULocationID')['pred_trip_count'].mean(),
                                            left_on = 'location_id', right_on = 'PULocationID')
        
        # plot the heatmap
        fig, ax = plt.subplots(1, 1, figsize = (10,8))
        heatmap_data.plot(column = 'pred_trip_count', cmap = 'Reds', linewidth = 0.8, edgecolor = 'black', legend = True, ax = ax)
        plt.title("Predicted NYC Taxi Demand (2025)")
        plt.show()

    def run_pipeline(self):
        """Runs te entire forecasting pipeline"""
        pipeline_steps = ['Load data', 'Aggregate data', 'Create features', 
                         'Split data', 'Train model', 'Evaluate model', 
                         'Visualize predictions']
        
        with tqdm(total=len(pipeline_steps), desc="Pipeline progress") as pbar:
            self.load_data()
            pbar.update(1)
            
            self.aggregate_data()
            pbar.update(1)
            
            self.create_features()
            pbar.update(1)
            
            self.split_data()
            pbar.update(1)
            
            self.train_xgboost()
            pbar.update(1)
            
            self.evaluate_model()
            pbar.update(1)
            
            self.visualise_predictions()
            pbar.update(1)

def main():
    data_path = "./Parquet/combined_tripdata_2024.parquet"
    geojson_path = "./tlc_taxi_zones.geojson"

    forecasting = TaxiDemandForecast(data_path, geojson_path)
    forecasting.run_pipeline()

if __name__ == "__main__":
    main()
