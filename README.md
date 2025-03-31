# Using XGBoost to forecast future 2025 demand hotspots in NYC with taxi trip data

## Reproduction Instructions

### Required Libraries

```
geopandas
pandas 
matplotlib
numpy
tqdm
scikit-learn
xgboost
cupy
```

### Data

#### parquet files: 
- https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

#### geojson files:
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

The outputted plots from `visualisation.py` and `model_pickup.py / model_dropoff.py` will go into the `./plots` folder in the directory/
The outputted predicted .csv files from the models will go into the `./outputs` folder in the directory.