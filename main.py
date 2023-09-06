import os
import pandas as pd
import glob
import numpy as np
import geopandas as gpd
from pyarrow.parquet import ParquetFile
import pyarrow as pa
from shapely.geometry import Point
from shapely.ops import transform
from functools import partial
import pyproj

output_df = pd.DataFrame({
    'Vendor_Name': pd.Series(dtype='str'),
    'tpep_pickup_datetime': pd.Series(dtype='datetime64[ns]'),
    'tpep_dropoff_datetime': pd.Series(dtype='datetime64[ns]'),
    'passenger_count': pd.Series(dtype='float64'),
    'trip_distance': pd.Series(dtype='float64'),
    'Start_Lon': pd.Series(dtype='float64'),
    'Start_Lat': pd.Series(dtype='float64'),
    'RatecodeID': pd.Series(dtype='float64'),
    'store_and_fwd_flag': pd.Series(dtype='str'),
    'PULocationID': pd.Series(dtype='int32'),
    'DOLocationID': pd.Series(dtype='int32'),
    'End_Lon': pd.Series(dtype='float64'),
    'End_Lat': pd.Series(dtype='float64'),
    'payment_type': pd.Series(dtype='int32'),
    'fare_amount': pd.Series(dtype='float64'),
    'extra': pd.Series(dtype='float64'),
    'mta_tax': pd.Series(dtype='float64'),
    'tip_amount': pd.Series(dtype='float64'),
    'tolls_amount': pd.Series(dtype='float64'),
    'improvement_surcharge': pd.Series(dtype='float64'),
    'total_amount': pd.Series(dtype='float64'),
    'congestion_surcharge': pd.Series(dtype='float64'),
    'airport_fee': pd.Series(dtype='float64'),
    'PRCP': pd.Series(dtype='float64'),
    'SNOW': pd.Series(dtype='float64'),
    'SNWD': pd.Series(dtype='float64'),
    'TMAX': pd.Series(dtype='float64'),
    'TMIN': pd.Series(dtype='float64'),
    'DAPR': pd.Series(dtype='float64'),
    'MDPR': pd.Series(dtype='float64'),
    'TOBS': pd.Series(dtype='float64'),
    'WESD': pd.Series(dtype='float64'),
    'WESF': pd.Series(dtype='float64'),
    'WT01': pd.Series(dtype='float64'),
    'WT03': pd.Series(dtype='float64'),
    'WT04': pd.Series(dtype='float64'),
    'WT05': pd.Series(dtype='float64'),
    'WT06': pd.Series(dtype='float64'),
    'WT11': pd.Series(dtype='float64')
})


def reproject_point(lon, lat, source_crs_epsg, target_crs_epsg):
    project = partial(
        pyproj.transform,
        pyproj.Proj(init=f'EPSG:{source_crs_epsg}'),
        pyproj.Proj(init=f'EPSG:{target_crs_epsg}')
    )
    return transform(project, Point(lon, lat))


def to_location_id(data_frame: pd.DataFrame, taxi_map) -> pd.DataFrame:
    source_crs_epsg = 4326
    target_crs_epsg = taxi_zones.crs.to_epsg()

    data_frame['start_geometry'] = data_frame.apply(
        lambda row: reproject_point(row['Start_Lon'], row['Start_Lat'], source_crs_epsg, target_crs_epsg),
        axis=1)
    data_frame['end_geometry'] = data_frame.apply(
        lambda row: reproject_point(row['End_Lon'], row['End_Lat'], source_crs_epsg, target_crs_epsg),
        axis=1)

    for index, row in data_frame.iterrows():
        pickup_location = taxi_map[taxi_map.geometry.contains(row['start_geometry'])]
        dropoff_location = taxi_map[taxi_map.geometry.contains(row['end_geometry'])]

        if not pickup_location.empty:
            data_frame.at[index, 'PULocationID'] = pickup_location.iloc[0]['LocationID']
        if not dropoff_location.empty:
            data_frame.at[index, 'DOLocationID'] = dropoff_location.iloc[0]['LocationID']

    data_frame.drop(['start_geometry', 'end_geometry'], axis=1, inplace=True)

    return data_frame


def change_column_names(data_frame: pd.DataFrame) -> pd.DataFrame:
    print("Changing column names for file " + file)
    schema_names = output_df.columns
    data_frame.columns = schema_names

    return data_frame


def change_column_type(data_frame: pd.DataFrame, year: int) -> pd.DataFrame:
    print("Changing column types for file " + file)
    data_frame = data_frame.replace(np.nan, None)
    new_type = {'Vendor_Name': str,
                'passenger_count': float,
                'RatecodeID': float,
                'store_and_fwd_flag': str,
                'Start_Lon': float,
                'Start_Lat': float,
                'End_Lon': float,
                'End_Lat': float,
                'payment_type': int,
                'PULocationID': int,
                'DOLocationID': int,
                'mta_tax': float,
                'improvement_surcharge': float,
                'congestion_surcharge': float,
                'airport_fee': float,
                'PRCP': float,
                'SNOW': float,
                'SNWD': float,
                'TMAX': float,
                'TMIN': float,
                'DAPR': float,
                'MDPR': float,
                'TOBS': float,
                'WESD': float,
                'WESF': float,
                'WT01': float,
                'WT03': float,
                'WT04': float,
                'WT05': float,
                'WT06': float,
                'WT11': float
                }
    if (year == 2009) | (year == 2010):
        data_frame['tpep_pickup_datetime'] = pd.to_datetime(data_frame['tpep_pickup_datetime'])
        data_frame['tpep_dropoff_datetime'] = pd.to_datetime(data_frame['tpep_dropoff_datetime'])
        if year == 2009:
            to_replace = ['CASH', 'Cash', 'Credit', 'CREDIT', 'No Charge', 'Dispute']
            replacement = ['2', '2', '1', '1', '3', '4']
            data_frame['payment_type'] = (data_frame['payment_type'].replace(to_replace, replacement))

            data_frame = data_frame.astype(new_type)
            to_replace = [0., 1.]
            replacement = ['0', '1']
            data_frame['store_and_fwd_flag'] = data_frame['store_and_fwd_flag'].replace(to_replace, replacement)
        else:

            to_replace = ['CAS', 'Cas', 'CSH', 'CRE', 'Cre', 'CRD', 'No ', 'NOC', 'NA ', 'Dis', 'DIS']
            replacement = ['2', '2', '2', '1', '1', '1', '3', '3', '3', '4', '4']
            data_frame['payment_type'] = data_frame['payment_type'].replace(to_replace, replacement)

            data_frame = data_frame.astype(new_type)

            to_replace = ['0', '1']
            replacement = ['N', 'Y']
            data_frame['store_and_fwd_flag'] = data_frame['store_and_fwd_flag'].replace(to_replace, replacement)
    else:
        data_frame = data_frame.astype(new_type)
        to_replace = [1, 2]
        replacement = ['CMT', 'VeriFone Inc.']
        data_frame['Vendor_Name'] = data_frame['Vendor_Name'].replace(to_replace, replacement)

    return data_frame


def add_columns(data_frame: pd.DataFrame, year: int) -> pd.DataFrame:
    print("Adding columns for file " + file)
    if (year == 2009) | (year == 2010):
        data_frame.insert(9, 'PULocationID', value=0)
        data_frame.insert(10, 'DOLocationID', value=0)
        data_frame.insert(19, 'improvement_surcharge', value='')
        data_frame['improvement_surcharge'] = None
        data_frame = data_frame.assign(congestion_surcharge=None, airport_fee=None)

    else:
        data_frame.insert(5, 'Start_Lon', value=0)
        data_frame['Start_Lon'] = None
        data_frame.insert(6, 'Start_Lat', value=0)
        data_frame['Start_Lat'] = None
        data_frame.insert(11, 'End_Lon', value=0)
        data_frame['End_Lon'] = None
        data_frame.insert(12, 'End_Lat', value=0)
        data_frame['End_Lat'] = None

    data_frame = data_frame.assign(PRCP=None, SNOW=None, SNWD=None, TMAX=None, TMIN=None, DAPR=None, MDPR=None,
                                   TOBS=None, WESD=None, WESF=None, WT01=None, WT03=None, WT04=None, WT05=None,
                                   WT06=None, WT11=None)

    return data_frame


def add_ghcn_columns(data_frame: pd.DataFrame, ghcn_data_frame: pd.DataFrame) -> pd.DataFrame:
    print("Changing ghcn column for file " + file)
    for ind in ghcn_data_frame.index:
        if not data_frame[data_frame['tpep_pickup_datetime'].dt.normalize() == ghcn_data_frame['DATE'][ind]].empty:
            index_list = (data_frame[data_frame['tpep_pickup_datetime'].dt.normalize() == ghcn_data_frame['DATE'][ind]]
                          .index.tolist())
            for i in index_list:
                data_frame['PRCP'][i] = ghcn_data_frame['PRCP'][ind]
                data_frame['SNOW'][i] = ghcn_data_frame['SNOW'][ind]
                data_frame['SNWD'][i] = ghcn_data_frame['SNWD'][ind]
                data_frame['TMAX'][i] = ghcn_data_frame['TMAX'][ind]
                data_frame['TMIN'][i] = ghcn_data_frame['TMIN'][ind]
                data_frame['DAPR'][i] = ghcn_data_frame['DAPR'][ind]
                data_frame['MDPR'][i] = ghcn_data_frame['MDPR'][ind]
                data_frame['TOBS'][i] = ghcn_data_frame['TOBS'][ind]
                data_frame['WESD'][i] = ghcn_data_frame['WESD'][ind]
                data_frame['WESF'][i] = ghcn_data_frame['WESF'][ind]
                data_frame['WT01'][i] = ghcn_data_frame['WT01'][ind]
                data_frame['WT03'][i] = ghcn_data_frame['WT03'][ind]
                data_frame['WT04'][i] = ghcn_data_frame['WT04'][ind]
                data_frame['WT05'][i] = ghcn_data_frame['WT05'][ind]
                data_frame['WT06'][i] = ghcn_data_frame['WT06'][ind]
                data_frame['WT11'][i] = ghcn_data_frame['WT11'][ind]

    return data_frame


if __name__ == '__main__':

    pd.set_option('mode.chained_assignment', None)
    folder_path = "test/"
    file_pattern = "yellow_tripdata_20??-??.parquet"

    file_list = glob.glob(os.path.join(folder_path, file_pattern))
    taxi_zones = gpd.read_file('taxi_zones/taxi_zones.shp')

    ghcn_df = pd.read_csv('data/USC00283704.csv')
    ghcn_df['DATE'] = pd.to_datetime(ghcn_df['DATE'])
    mask = ghcn_df['DATE'] >= '2009-01-01'
    ghcn_df = ghcn_df.loc[mask]

    for file in file_list:
        df = pd.read_parquet(file)
        print("Read file " + file)

        if ('2009' in file) | ('2010' in file):
            if '2009' in file:
                df = add_columns(df, 2009)
                df = change_column_names(df)
                df = change_column_type(df, 2009)
            else:
                df = add_columns(df, 2010)
                df = change_column_names(df)
                df = change_column_type(df, 2010)

            df = to_location_id(df, taxi_zones)

        else:
            df = add_columns(df, 0)
            df = change_column_names(df)
            df = change_column_type(df, 0)

        df = add_ghcn_columns(df, ghcn_df)

        file_output = file.replace("test/", "output/")
        df.to_parquet(file_output, index=False)
        print("Saved to .parquet for file " + file)
