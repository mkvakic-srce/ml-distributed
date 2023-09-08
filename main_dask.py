import os
from dask.distributed import Client
import numpy as np
import pandas as pd
import dask.dataframe as dd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def separate_date(df: dd.DataFrame) -> dd.DataFrame:
    df['tpep_pickup_datetime'] = dd.to_datetime(df['tpep_pickup_datetime'])
    df['year'] = df['tpep_pickup_datetime'].dt.year
    df['month'] = df['tpep_pickup_datetime'].dt.month
    df['day'] = df['tpep_pickup_datetime'].dt.day

    df = df.drop('tpep_pickup_datetime', axis=1)

    df = df[['PULocationID', 'year', 'month', 'day', 'PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN', 'DAPR', 'MDPR', 'TOBS',
             'WESD', 'WESF', 'WT01', 'WT03', 'WT04', 'WT05', 'WT06', 'WT11']]

    return df


def add_number_of_cars(df: dd.DataFrame) -> dd.DataFrame:
    groupby_cols = ['PULocationID', 'year', 'month', 'day']
    df_grouped = df.groupby(groupby_cols).agg({
        'PULocationID': 'count',
        'PRCP': 'mean',
        'SNOW': 'mean',
        'SNWD': 'mean',
        'TMAX': 'mean',
        'TMIN': 'mean',
        'DAPR': 'mean',
        'MDPR': 'mean',
        'TOBS': 'mean',
        'WESD': 'mean',
        'WESF': 'mean',
        'WT01': 'mean',
        'WT03': 'mean',
        'WT04': 'mean',
        'WT05': 'mean',
        'WT06': 'mean',
        'WT11': 'mean'
    })

    df_grouped = df_grouped.rename(columns={'PULocationID': 'number_of_cars'}).reset_index()

    df_merged = df.merge(df_grouped)

    return df_merged


def load_data() -> dd.DataFrame:
    folder_path = "output/"

    df_dask = dd.read_parquet(folder_path)

    to_remove = ['Vendor_Name', 'tpep_dropoff_datetime', 'passenger_count', 'trip_distance',
                 'Start_Lon', 'Start_Lat', 'RatecodeID', 'store_and_fwd_flag', 'End_Lon', 'End_Lat',
                 'payment_type', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
                 'improvement_surcharge', 'total_amount', 'congestion_surcharge', 'airport_fee']

    df_dask = df_dask.drop(columns=to_remove)

    df_dask = separate_date(df_dask)

    df_dask = add_number_of_cars(df_dask)

    df_dask = df_dask.drop_duplicates().reset_index(drop=True)

    # df_dask = df_dask.sort_values(['PULocationID', 'year', 'month', 'day']).reset_index(drop=True)

    return df_dask

def main():
    # client = Client(os.environ['SCHEDULER_ADDRESS'])
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    df = load_data().compute()

    PULocationID = df.groupby('PULocationID')['year'].count().argmax()

    X = df[ df['PULocationID'] == PULocationID ].drop(columns=['PULocationID', 'number_of_cars'])
    y = df[ df['PULocationID'] == PULocationID ]['number_of_cars']

    seed = 2
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    print(X_train.columns)

    accuracy_scores = []
    n_estimators_values = range(1, 1001, 10)

    for n_estimators in n_estimators_values[-2:]:
        model = XGBRegressor(n_estimators=n_estimators)
        model.fit(X_train,
                  y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=True)

        y_pred = model.predict(X_train)

        mse = mean_squared_error(y_train, y_pred)

        rmse = np.sqrt(mse)

        accuracy_scores.append(rmse)

    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_values, accuracy_scores, marker='o', linestyle='-')
    plt.title('Accuracy Score vs. n_estimators')
    plt.xlabel('n_estimators')
    plt.ylabel('Accuracy Score')
    plt.grid(True)
    plt.show()

    # client.shutdown()
if __name__ == '__main__':
    main()