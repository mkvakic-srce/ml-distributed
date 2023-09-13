import os
from dask.distributed import Client, LocalCluster
import numpy as np
import pandas as pd
import dask.dataframe as dd
from xgboost.dask import DaskXGBRegressor
from dask_ml.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def separate_date(df: dd.DataFrame) -> dd.DataFrame:
    df['tpep_pickup_datetime'] = dd.to_datetime(df['tpep_pickup_datetime'])
    df['year'] = df['tpep_pickup_datetime'].dt.year
    df['month'] = df['tpep_pickup_datetime'].dt.month
    df['day'] = df['tpep_pickup_datetime'].dt.day

    df = df.drop('tpep_pickup_datetime', axis=1)

    df = df[['PULocationID', 'year', 'month', 'day', 'PRCP', 'SNOW', 'TMAX', 'TMIN', 'TOBS']]

    return df


def add_number_of_cars(df: dd.DataFrame) -> dd.DataFrame:
    groupby_cols = ['PULocationID', 'year', 'month', 'day']
    df_grouped = df.groupby(groupby_cols).agg({
        'PULocationID': 'count',
        'PRCP': 'mean',
        'SNOW': 'mean',
        'TMAX': 'mean',
        'TMIN': 'mean',
        'TOBS': 'mean',
    })

    df_grouped = df_grouped.rename(columns={'PULocationID': 'number_of_cars'}).reset_index()

    df_merged = df.merge(df_grouped)

    return df_merged


def load_data() -> dd.DataFrame:
    folder_path = "test/"

    df_dask = dd.read_parquet(folder_path)

    to_remove = ['Vendor_Name', 'tpep_dropoff_datetime', 'passenger_count', 'trip_distance',
                 'Start_Lon', 'Start_Lat', 'RatecodeID', 'store_and_fwd_flag', 'End_Lon', 'End_Lat',
                 'payment_type', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
                 'improvement_surcharge', 'total_amount', 'congestion_surcharge', 'airport_fee', 'SNWD', 'DAPR', 'MDPR',
                 'WESD', 'WESF', 'WT01', 'WT03', 'WT04', 'WT05', 'WT06', 'WT11']

    df_dask = df_dask.drop(columns=to_remove)

    df_dask = separate_date(df_dask)

    df_dask = add_number_of_cars(df_dask)

    df_dask = df_dask.drop_duplicates().reset_index(drop=True)
    # df_dask = df_dask.sort_values(['PULocationID', 'year', 'month', 'day']).reset_index(drop=True)

    return df_dask


if __name__ == '__main__':
    client = Client(os.environ['SCHEDULER_ADDRESS'])
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    df = load_data()
    result = pd.DataFrame(columns=['PULocationID', 'number_of_cars', 'rmse', 'r2'])

    for id in range(1, 265):
        X = df[df['PULocationID'] == id].drop(columns=['PULocationID', 'number_of_cars'])
        y = df[df['PULocationID'] == id]['number_of_cars']

        size = len(X.index)

        if size > 4:
            seed = 2
            test_size = 0.3
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed,
                                                                shuffle=False)

            model = DaskXGBRegressor(n_estimators=500)
            model.client = client
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            rmse = mean_squared_error(y_test, y_pred, squared=False)

            r2 = r2_score(y_test, y_pred)

            result.loc[len(result.index)] = [id, y.sum().compute(), rmse, r2]
            result['PULocationID'] = result['PULocationID'].astype(int)
            result['number_of_cars'] = result['number_of_cars'].astype(int)

    print(result)

    client.shutdown()
