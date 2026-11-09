import sys
sys.path.append('..')
import h3
import folium
import torch
from config import Config
import pickle
from model.node2vec_ import train_node2vec

import logging
from utils import tool_funcs
import pandas as pd
import time
from ast import literal_eval
import numpy as np
import datetime

class Traj2Hex:
    def __init__(self, min_lon, min_lat, max_lon, max_lat, res):
        self.resolution = res
        print(res)
        self.min_lon = min_lon
        self.min_lat = min_lat
        self.max_lon = max_lon
        self.max_lat = max_lat


    def tokenize(self, traj):
        return [h3.geo_to_h3(p[1], p[0], self.resolution) for p in traj]    # lat, lon


    def detokenize(self, token):
        return tuple(h3.h3_to_geo(token))

    def get_adjacency(self, hex_index):
        """ Returns the indices of adjacent hexagons for a given hexagon. """
        return h3.hex_ring(hex_index, 1)

    def get_hexagons(self):
        # Define the bounding box in GeoJSON format
        geojson = {
            'type': 'Polygon',
            'coordinates': [[
                [self.min_lon, self.min_lat],  # Lower left corner
                [self.min_lon, self.max_lat],  # Upper left corner
                [self.max_lon, self.max_lat],  # Upper right corner
                [self.max_lon, self.min_lat],  # Lower right corner
                [self.min_lon, self.min_lat]  # Back to the start to close the polygon
            ]]
        }
        # x_min, y_min, x_max, y_max are corresponding to min_lon, min_lat, max_lon, max_lat
        # Use h3.polyfill to find all hexagons within the bounding box at the specified resolution
        hex_indices = h3.polyfill(geojson, self.resolution, geo_json_conformant=True)
        # print(hex_indices)
        return sorted(list(hex_indices))

    def create_graph(self, hex_indices):
        """ Creates edges based on hexagon adjacency. """
        edge_index = []
        index_mapping = {hex_index: i for i, hex_index in enumerate(hex_indices)}

        for hex_index in hex_indices:
            try:
                neighbors = self.get_adjacency(hex_index)
                for neighbor in neighbors:
                    if neighbor in index_mapping:
                        edge_index.append([index_mapping[hex_index], index_mapping[neighbor]])
            except h3.H3RuntimeError:
                continue

        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        with open(Config.index_mapping_file, 'wb') as f:
            pickle.dump(index_mapping, f, protocol=pickle.HIGHEST_PROTOCOL)

        return edge_index_tensor

    def plot_hexagons(self, traj):
        """Plot both the trajectory hexagons and raw points on a Folium map."""
        # Create a map centered on the first point of the trajectory
        if traj:
            initial_location = [traj[0][1], traj[0][0]]  # Reverse lat and lon for folium
        else:
            initial_location = [0, 0]  # Default to a neutral location if traj is empty

        map = folium.Map(location=initial_location, zoom_start=12, tiles='cartodbpositron')

        # Plot each hexagon in green
        hex_indices = self.tokenize(traj)
        for hex_index in hex_indices:
            boundary = h3.h3_to_geo_boundary(hex_index, geo_json=True)
            folium.Polygon(
                locations=[(coord[1], coord[0]) for coord in boundary],  # Lat, Lon order for Folium
                color='green',
                fill=True,
                fill_color='green',
                fill_opacity=0.4
            ).add_to(map)

        # Plot each raw point in blue
        for point in traj:
            folium.CircleMarker(
                location=[point[1], point[0]],  # Reverse lat and lon for folium
                radius=3,
                color='blue',
                fill=True,
                fill_color='blue'
            ).add_to(map)

        # Automatically adjust the bounds to show all the points
        map.fit_bounds(map.get_bounds())
        return map


def plot(hex_indices, initial_location, zoom_start=12):
    map = folium.Map(location=initial_location, zoom_start=zoom_start, tiles='cartodbpositron')

    for hex_index in hex_indices:
        boundary = h3.h3_to_geo_boundary(hex_index, geo_json=True)
        # print(boundary)
        folium.Polygon(
            locations=[(coord[1], coord[0]) for coord in boundary], # latitude, longitude for Folium
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.4
        ).add_to(map)

    return map



def unix2dt(time):
    # Convert to datetime object
    dt_object = datetime.datetime.fromtimestamp(int(time))

    # Extracting detailed components from the datetime object
    month = dt_object.month
    day = dt_object.day
    weekday = dt_object.weekday()  # Monday is 0 and Sunday is 6
    hour = dt_object.hour
    minute = dt_object.minute
    second = dt_object.second
    # print(month, day, weekday, hour, minute)
    return month, day, weekday, hour, minute, second


def expand_timestamps(row):
    # Start timestamp and number of steps
    base_time = int(row['TIMESTAMP'])
    steps = int(row['trajlen'])

    # Generate all timestamps increasing by 15 seconds each
    all_timestamps = [base_time + i * 15 for i in range(steps)]

    # Convert all timestamps to detailed components
    return [unix2dt(ts) for ts in all_timestamps]

def inrange(lon, lat):
    if lon <= Config.min_lon or lon >= Config.max_lon \
            or lat <= Config.min_lat or lat >= Config.max_lat:
        return False
    return True


def clean_and_output_data():
    _time = time.time()
    dfraw = pd.read_csv(Config.root_dir + '/data/porto.csv')
    dfraw = dfraw.rename(columns = {"POLYLINE": "wgs_seq"})
    dfraw = dfraw[dfraw.MISSING_DATA == False]

    dfraw.wgs_seq = dfraw.wgs_seq.apply(literal_eval)
    dfraw['trajlen'] = dfraw.wgs_seq.apply(lambda traj: len(traj))
    dfraw = dfraw[(dfraw.trajlen >= Config.min_traj_len) & (dfraw.trajlen <= Config.max_traj_len)]
    logging.info('Preprocessed-rm length. #traj={}'.format(dfraw.shape[0]))

    # range requirement
    dfraw['inrange'] = dfraw.wgs_seq.map(lambda traj: sum([inrange(p[0], p[1]) for p in traj]) == len(traj) ) # True: valid
    dfraw = dfraw[dfraw.inrange == True]
    logging.info('Preprocessed-rm range. #traj={}'.format(dfraw.shape[0]))

    total_points = dfraw['trajlen'].sum()
    print(f"num of total points: {total_points}.")
    dfraw['DateTime'] = dfraw.apply(expand_timestamps, axis=1)


    # logging.info('Preprocessed-output. #traj={}'.format(dfraw.shape[0]))
    dfraw = dfraw[['TAXI_ID', 'trajlen', 'wgs_seq', 'DateTime']].reset_index(drop = True)
    print(dfraw.columns)
    dfraw.to_pickle(Config.dataset_file)
    logging.info('Preprocess end. @={:.0f}'.format(time.time() - _time))
    # print(dfraw['traj_p'].loc[:10])
    # print(dfraw['DateTime'].loc[:10])
    return

def get_neighbors(hex_indices):
    """
    generate neighbor matrix.
    """
    index_mapping = pickle.load(open(Config.index_mapping_file, 'rb'))

    dense_neighbors = []
    for t in hex_indices:
        try:
            ring = sorted(list(h3.hex_ring(t, 1)))  # ordered
        except (h3.H3Error, ValueError):
            ring = []  # fallback

        # position-preserving mapping
        neighbors = []
        for n in ring:
            neighbors.append(index_mapping.get(n, -1))  # keep order, add 0 if missing

        # If ring is incomplete (e.g., near pentagon), pad to full length
        if len(neighbors) < 6:
            print(1)
            neighbors += [0] * (6 - len(neighbors))

        dense_neighbors.append(neighbors)

    dense_neighbors = np.array(dense_neighbors, dtype=np.int64)

    print(dense_neighbors)

    with open(Config.neighbours_file, "wb") as f:
        pickle.dump(dense_neighbors, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved dense neighbors to {Config.neighbours_file}")


if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG,
                        format = "[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
                        handlers = [logging.FileHandler(Config.root_dir+'/log/'+tool_funcs.log_file_name(), mode = 'w'),
                                    logging.StreamHandler()]
                        )
    Config.dataset = ('migration')
    Config.post_value_updates()

    """
    initialize the hexagon tokens.
    """
    if Config.dataset == 'porto':
        hex = Traj2Hex(-8.7005 - 0.005, 41.1001 - 0.005, -8.5192 + 0.005, 41.2086 + 0.005, Config.cell_res)

    elif Config.dataset == 'ais_au':
        hex = Traj2Hex(75.0000 - 1, -89.8300 - 1, 163.0000 + 1, -2.0000 + 1, Config.cell_res)

    elif Config.dataset == 'geolife' or Config.dataset == 'beijing':
        hex = Traj2Hex(116.109382 - 0.005, 39.717276 - 0.005, 116.734229 + 0.005, 40.046068 + 0.005, Config.cell_res)

    elif Config.dataset == 'tokyo':
        hex = Traj2Hex(139.47176970135368 - 0.005, 35.5130593801101366 - 0.005, 139.90628600120544 + 0.005, 35.86602997137174 + 0.005, Config.cell_res)

    elif Config.dataset == 'nyc':
        hex = Traj2Hex(-74.27075087427112 - 0.005, 40.557295 - 0.005, -73.68581968909615 + 0.005, 40.98764596081398 + 0.005, Config.cell_res)

    elif Config.dataset == 'migration':
        hex = Traj2Hex(4.02517 - 1, -2.40633 - 1, 58.52283 + 1,
                       65.2325 + 1, Config.cell_res)

    print("finished hex creation.")
    hex_indices = hex.get_hexagons()
    print("finished index retrieval.")
    # print(hex_indices)
    print(len(hex_indices))




    with open(Config.dataset_cell_file, 'wb') as fh:
        pickle.dump(hex_indices, fh, protocol=pickle.HIGHEST_PROTOCOL)
    min_lon = float(Config.min_lon) - 0.005
    max_lon = float(Config.min_lat) + 0.005
    min_lat = float(Config.max_lon) - 0.005
    max_lat = float(Config.max_lat) + 0.005
    
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2
    """
    for map plotting.
    """
    # center = (center_lat, center_lon)
    # map = plot(hex_indices, center)
    # map.save('test_map.html')
    """
    pre-train the hexagon tokens.
    """
    edges = hex.create_graph(hex_indices)
    print("finished graph creation.")
    print(edges)
    train_node2vec(edges)
    """
    save neighbours.
    """
    # clean_and_output_data()
    # get_neighbors(hex_indices)


