# Using XGBoost to Forecast Future 2025 Taxi Demand Hotspots in New York City with TLC Taxi Trip Data

## Reproduction Instructions

### Required Libraries

```
geopandas
pandas 
matplotlib
numpy
seaborn
tqdm
scikit-learn
xgboost
cupy
python >= 3.12.8
```

### Data

#### Parquet files: 
- https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

#### GeoJSON files:
- https://www.kaggle.com/datasets/mxruedag/tlc-nyc-taxi-zones?resource=download


#### Step 1: 

Clone the repository. It contains the necessary .parquet and geojson file for the assignment. If the code throws any errors, the data can be found in the links above. The NYC TLC Taxi Trip data is from 1st January - 30th November 2024.

#### Step 2:

Run each file in the following order:

```
1. data_load.py
2. visualisation.py
3. model_pickup.py / model_dropoff.py
```

The `data_load.py` allows the user to input which plot they want to produce. Make sure to type it correctly as specified in the console.

The outputted plots from `visualisation.py` and `model_pickup.py / model_dropoff.py` will go into the `./plots` folder in the directory.

The outputted predicted .csv files from the models will go into the `./outputs` folder in the directory.
