import pandas as pd
import numpy as np
import json

bank_dict = {
    'ВТБ': 5478,
    'АЛЬФА-БАНК': 1942,
    'РОСБАНК': 8083,
    'РОССЕЛЬХОЗБАНК': 496,
    'ГАЗПРОМБАНК': 3185,
    'АК БАРС': 1022,
    'УРАЛСИБ БАНК': 32
}

settlements = pd.read_csv('data/settlements_processed.csv', sep=';')
salary = pd.read_csv('data/salary_processed.csv', sep=';')


def calc_dist(lat_1, long_1, lat_2, long_2):
    lat_diff = lat_1 - lat_2
    long_diff = long_1 - long_2
    distance = 6378.137 * 2 * np.arcsin(np.sqrt(np.sin(lat_diff / 2.0) ** 2 +
                                                np.cos(lat_1) * np.cos(lat_2) * np.sin(long_diff / 2.0) ** 2))
    return distance


def find_population(lat, long):
    distances = settlements.apply(
        lambda settlement: calc_dist(
            lat, long, settlement['latitude_rad'], settlement['longitude_rad']),
        axis=1)
    return settlements.loc[distances.idxmin(), ['population', 'region']]


def transform_data(data):

    data['atm_id'] = range(data.shape[0])
    data['atm_group'] = data['atm_group'].map(bank_dict)
    data[['lat_rad', 'long_rad']] = np.radians(data[['lat', 'long']])

    # Количество заведений поблизости
    data['key'] = 0
    for category in ['mall', 'bank', 'department_store', 'station', 'alcohol',
                     'police', 'university', 'railway_station']:
        lat_long = []
        with open(f'data/osm_node_{category}.json', encoding='utf8') as f:
            json_data = json.load(f)
            for elem in json_data['elements']:
                if elem['type'] == 'node':
                    lat_long.append(
                        [elem['lat'], elem['lon']]
                    )

        cat_df = pd.DataFrame(lat_long, columns=['lat', 'long'])
        cat_df[['cat_lat_rad', 'cat_long_rad']
               ] = np.radians(cat_df[['lat', 'long']])
        cat_df['key'] = 0

        cross_merge = data.merge(cat_df, on='key', how='outer')

        # Haversine distance
        cross_merge['lat_diff'] = cross_merge['cat_lat_rad'] - \
            cross_merge['lat_rad']
        cross_merge['long_diff'] = cross_merge['cat_long_rad'] - \
            cross_merge['long_rad']
        cross_merge['distance'] = 6378.137 * 2 * np.arcsin(np.sqrt(np.sin(cross_merge['lat_diff']/2.0)**2 +
                                                                   np.cos(cross_merge['lat_rad']) *
                                                                   np.cos(cross_merge['cat_lat_rad']) *
                                                                   np.sin(cross_merge['long_diff']/2.0)**2))

        cross_merge[f'n_{category}_100'] = (
            cross_merge['distance'] <= 0.3).astype(np.uint8)
        cross_merge[f'n_{category}_300'] = (
            cross_merge['distance'] <= 0.1).astype(np.uint8)
        data = data.merge(cross_merge.groupby('atm_id').aggregate({f'n_{category}_100': 'sum',
                                                                   f'n_{category}_300': 'sum'}).reset_index(),
                          on='atm_id',
                          how='left')

    # Население
    data[['population', 'region']] = data.apply(
        lambda row: find_population(row['lat_rad'], row['long_rad']),
        axis=1
    )
    data['region'] = data['region'].str.lower()
    data = data.rename(columns={'region': 'subject'})

    # Зарплата
    data = data.merge(salary, how='left', on='subject')
    data = data.drop(columns=['key', 'atm_id', 'lat_rad', 'long_rad'])

    return data
