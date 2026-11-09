import sys
sys.path.append('..')
import os
import math
import time
import random
import logging
import torch
import pickle
import pandas as pd
from ast import literal_eval
import numpy as np
import traj_dist.distance as tdist
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import folium
import h3

from config import Config
from utils import tool_funcs
from utils.cellspace import CellSpace
from utils.tool_funcs import lonlat2meters
# from model.node2vec_ import train_node2vec
from utils.edwp import edwp
from utils.edwp import edwp
from utils.data_loader import read_trajsimi_traj_dataset
# from utils.traj import *
import datetime
# from utils.hex import *

from pybind11_rdp import rdp


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


def get_offset(x, y , cellspace):
    """Convert a single point to cell ID and offset."""
    assert cellspace.x_min <= x <= cellspace.x_max and cellspace.y_min <= y <= cellspace.y_max
    cell_id = cellspace.get_cellid_by_point(x, y)
    i_x, i_y = cellspace.get_xyidx_by_cellid(cell_id)
    cell_x_min = cellspace.x_min + i_x * cellspace.x_unit
    cell_y_min = cellspace.y_min + i_y * cellspace.y_unit
    offset_x = x - cell_x_min
    offset_y = y - cell_y_min
    assert 0 <= offset_x < 100 and 0 <= offset_y < 100
    return (i_x, i_y, offset_x, offset_y)

def inv_offset(cell_id, offset_x, offset_y, cellspace):
    # Convert cell ID back to grid indices
    i_x, i_y = cellspace.get_xyidx_by_cellid(cell_id)

    # Calculate the bottom-left corner of the cell
    cell_x_min = cellspace.x_min + i_x * cellspace.x_unit
    cell_y_min = cellspace.y_min + i_y * cellspace.y_unit

    # Reconstruct the original Mercator coordinates
    original_x = cell_x_min + offset_x
    original_y = cell_y_min + offset_y

    return original_x, original_y

def clean_and_output_data():
    _time = time.time()
    # cellspace = pickle.load(open(Config.dataset_cell_file, 'rb'))
    # embs = pickle.load(open(Config.dataset_embs_file, 'rb'))
    # https://archive.ics.uci.edu/ml/machine-learning-databases/00339/
    # download train.csv.zip and unzip it. rename train.csv to porto.csv
    dfraw = pd.read_csv(Config.root_dir + '/data/porto.csv')
    dfraw = dfraw.rename(columns = {"POLYLINE": "wgs_seq"})
    # print(dfraw['TAXI_ID'])
    dfraw = dfraw[dfraw.MISSING_DATA == False]
    # length requirement
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
    dfraw['hex'] = dfraw.wgs_seq.apply(lambda traj: [h3.geo_to_h3(p[1], p[0], Config.cell_res) for p in traj])
    # convert to Mercator
    dfraw['merc_seq'] = dfraw.wgs_seq.apply(lambda traj: [list(lonlat2meters(p[0], p[1])) for p in traj])

    logging.info('Preprocessed-output. #traj={}'.format(dfraw.shape[0]))
    dfraw = dfraw[['TAXI_ID', 'trajlen', 'wgs_seq', 'merc_seq', 'hex', 'DateTime']].reset_index(drop = True)
    # print(dfraw.columns)
    dfraw.to_pickle(Config.dataset_file)
    logging.info('Preprocess end. @={:.0f}'.format(time.time() - _time))
    print(dfraw['hex'].loc[:10])
    return



def process_long_traj():
    _time = time.time()
    # cellspace = pickle.load(open(Config.dataset_cell_file, 'rb'))
    # embs = pickle.load(open(Config.dataset_embs_file, 'rb'))
    # https://archive.ics.uci.edu/ml/machine-learning-databases/00339/
    # download train.csv.zip and unzip it. rename train.csv to porto.csv
    dfraw = pd.read_csv(Config.root_dir + '/data/porto.csv')
    dfraw = dfraw.rename(columns={"POLYLINE": "wgs_seq"})
    dfraw = dfraw[dfraw.MISSING_DATA == False]
    # length requirement
    dfraw.wgs_seq = dfraw.wgs_seq.apply(literal_eval)
    dfraw['trajlen'] = dfraw.wgs_seq.apply(lambda traj: len(traj))

    dfraw = dfraw[(dfraw.trajlen >= 201) & (dfraw.trajlen <= 2000)]
    logging.info('Preprocessed-rm length. #traj={}'.format(dfraw.shape[0]))

    dfraw['inrange'] = dfraw.wgs_seq.map(
        lambda traj: sum([inrange(p[0], p[1]) for p in traj]) == len(traj))  # True: valid
    dfraw = dfraw[dfraw.inrange == True]
    logging.info('Preprocessed-rm range. #traj={}'.format(dfraw.shape[0]))

    total_points = dfraw['trajlen'].sum()
    print(f"num of total points: {total_points}.")
    dfraw['DateTime'] = dfraw.apply(expand_timestamps, axis=1)
    dfraw['hex'] = dfraw.wgs_seq.apply(lambda traj: [h3.geo_to_h3(p[1], p[0], Config.cell_res) for p in traj])
    # convert to Mercator
    dfraw['merc_seq'] = dfraw.wgs_seq.apply(lambda traj: [list(lonlat2meters(p[0], p[1])) for p in traj])

    logging.info('Preprocessed-output. #traj={}'.format(dfraw.shape[0]))
    dfraw = dfraw[['TAXI_ID', 'trajlen', 'wgs_seq', 'merc_seq', 'hex', 'DateTime']].reset_index(drop = True)
    # print(dfraw.columns)
    dfraw.to_pickle(f"{Config.dataset_file}_long")
    logging.info('Preprocess end. @={:.0f}'.format(time.time() - _time))
    print(dfraw['hex'].loc[:10])
    return

def data_stats():
    # cellspace = pickle.load(open(Config.dataset_cell_file, 'rb'))

    df = pd.read_pickle(Config.dataset_file)
    # print(df['TAXI_ID'].unique())

    l = df.shape[0]
    train_idx = (int(l * 0), int(l * 0.7))
    train = df[train_idx[0]: train_idx[1]]

    taxi_20000589 = train[train['TAXI_ID'] == 20000589]
    # print(taxi_20000589.shape[0])

    trajs = taxi_20000589['wgs_seq'].tolist()
    # print(trajs.head(5))

    # Calculate the center point for initializing the map
    all_points = [point for traj in trajs for point in traj]
    center_lat = sum([point[1] for point in all_points]) / len(all_points)
    center_lon = sum([point[0] for point in all_points]) / len(all_points)

    # Create the map
    folium_map = folium.Map(location=[center_lat, center_lon], zoom_start=14)

    # Add points to the map
    for traj in trajs:
        for lon, lat in traj:
            folium.CircleMarker(
                location=[lat, lon],  # Folium uses [lat, lon]
                radius=4,  # Circle radius
                color='red',  # Circle color
                fill=True,
                fill_color='red',
                fill_opacity=0.7
            ).add_to(folium_map)

    folium_map.save("taxi_20000589.html")


def init_cellspace():
    # 1. create cellspase
    # 2. initialize cell embeddings (create graph, train, and dump to file)

    x_min, y_min = lonlat2meters(Config.min_lon, Config.min_lat)
    x_max, y_max = lonlat2meters(Config.max_lon, Config.max_lat)
    x_min -= Config.cellspace_buffer
    y_min -= Config.cellspace_buffer
    x_max += Config.cellspace_buffer
    y_max += Config.cellspace_buffer

    cell_size = int(Config.cell_size)
    cs = CellSpace(cell_size, cell_size, x_min, y_min, x_max, y_max)
    with open(Config.dataset_cell_file, 'wb') as fh:
        pickle.dump(cs, fh, protocol = pickle.HIGHEST_PROTOCOL)

    _, edge_index = cs.all_neighbour_cell_pairs_permutated_optmized()
    edge_index = torch.tensor(edge_index, dtype = torch.long, device = Config.device).T
    train_node2vec(edge_index)
    return


def generate_longsimi_test_dataset():
    trajs = pd.read_pickle(Config.dataset_file)  # using test part only
    long_trajs = pd.read_pickle(f"{Config.dataset_file}_long")
    l_long = long_trajs.shape[0]
    print(l_long)
    l = trajs.shape[0]
    n_query = 1000
    n_db = 100000

    n_reg_traj = n_db - l_long
    test_idx = (int(l * 0.8), int(l * 0.8) + n_reg_traj)
    test_reg_trajs = trajs[test_idx[0]: test_idx[1]]
    # test_trajs = np.concatenate((long_trajs, test_reg_trajs), axis=0)
    test_trajs = pd.concat(
        [long_trajs, test_reg_trajs],
        ignore_index=True,
        axis=0
    )

    logging.info("Test trajs loaded.")

    def _raw_dataset():
        query_lst = []  # [N, len, 2]
        db_lst = []
        i = 0
        for _, v in test_trajs.wgs_seq.iteritems():
            if i < n_query:
                # query_lst.append(np.array(v)[::2].tolist())
                query_traj = np.array(v)[::2].tolist()
                query_h3 = [h3.geo_to_h3(p[1], p[0], Config.cell_res) for p in query_traj]
                query_lst.append(query_h3)

            # db_lst.append(np.array(v)[1::2].tolist())
            db_traj = np.array(v)[1::2].tolist()
            db_h3 = [h3.geo_to_h3(p[1], p[0], Config.cell_res) for p in db_traj]
            db_lst.append(db_h3)

            i += 1

        output_file_name = Config.dataset_file + '_longsimi_raw.pkl'
        with open(output_file_name, 'wb') as fh:
            pickle.dump((query_lst, db_lst), fh, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info("_raw_dataset done.")
        return

    def _downsample_dataset(rate):
        unrate = 1 - rate
        query_lst, db_lst = [], []

        def sample_preserve_ends(seq):
            if len(seq) <= 2:
                return seq
            first, last = seq[0], seq[-1]
            mids = seq[1:-1]
            k = math.ceil(len(mids) * unrate)
            idx = np.sort(np.random.choice(len(mids), k, replace=False))
            return np.vstack([first, mids[idx], last])

        for i, v in enumerate(test_trajs.wgs_seq.values):
            traj = np.array(v)

            # query trajectories (every other starting at 0)
            if i < n_query:
                q_seq = traj[::2]
                q_sampled = sample_preserve_ends(q_seq)
                query_lst.append([h3.geo_to_h3(pt[1], pt[0], Config.cell_res) for pt in q_sampled])

            # db trajectories (every other starting at 1)
            db_seq = traj[1::2]
            db_sampled = sample_preserve_ends(db_seq)
            db_lst.append([h3.geo_to_h3(pt[1], pt[0], Config.cell_res) for pt in db_sampled])

        output = f"{Config.dataset_file}_longsimi_downsampling_{rate}.pkl"
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, 'wb') as fh:
            pickle.dump((query_lst, db_lst), fh, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"_downsample_dataset done. rate={rate}")


    def _distort_dataset(rate):
        query_lst = []  # [N, len, 2]
        db_lst = []
        i = 0
        for _, v in test_trajs.wgs_seq.iteritems():
            if i < n_query:
                _q = np.array(v)[::2]
                for _row in range(_q.shape[0]):
                    if random.random() < rate:
                        # print(_q[_row])
                        # _q[_row] = _q[_row] + [tool_funcs.truncated_rand(), tool_funcs.truncated_rand()]
                        noise_lon, noise_lat = tool_funcs.truncated_gps_noise(_q[_row, 1])
                        _q[_row] = _q[_row] + np.array([noise_lon, noise_lat])
                # query_lst.append(_q.tolist())
                query_h3 = [h3.geo_to_h3(p[1], p[0], Config.cell_res) for p in _q]
                query_lst.append(query_h3)

            _db = np.array(v)[1::2]
            for _row in range(_db.shape[0]):
                if random.random() < rate:
                    # _db[_row] = _db[_row] + [tool_funcs.truncated_rand(), tool_funcs.truncated_rand()]
                    noise_lon, noise_lat = tool_funcs.truncated_gps_noise(_db[_row, 1])
                    _db[_row] = _db[_row] + np.array([noise_lon, noise_lat])
            # db_lst.append(_db.tolist())
            db_h3 = [h3.geo_to_h3(p[1], p[0], Config.cell_res) for p in _db]
            db_lst.append(db_h3)

            i += 1

        output_file_name = Config.dataset_file + '_longsimi_distort_' + str(rate) + '.pkl'
        with open(output_file_name, 'wb') as fh:
            pickle.dump((query_lst, db_lst), fh, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info("_distort_dataset done. rate={}".format(rate))
        return


    _raw_dataset()

    for rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
        _downsample_dataset(rate)

    for rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
        _distort_dataset(rate)



def generate_newsimi_test_dataset():
    trajs = pd.read_pickle(Config.dataset_file)  # using test part only
    l = trajs.shape[0]
    n_query = 1000
    n_db = 100000
    test_idx = (int(l * 0.8), int(l * 0.8) + n_db)
    test_trajs = trajs[test_idx[0]: test_idx[1]]
    logging.info("Test trajs loaded.")

    # for varying db size
    def _raw_dataset():
        query_trajs_raw = []  # raw [lon, lat] sequences for queries
        db_trajs_raw = []
        query_lst = []  # [N, len, 2]
        db_lst = []
        i = 0
        for _, v in test_trajs.wgs_seq.iteritems():
            if i < n_query:
                # query_lst.append(np.array(v)[::2].tolist())
                query_traj = np.array(v)[::2].tolist()
                query_h3 = [h3.geo_to_h3(p[1], p[0], Config.cell_res) for p in query_traj]
                query_lst.append(query_h3)
                query_trajs_raw.append(query_traj)

            # db_lst.append(np.array(v)[1::2].tolist())
            db_traj = np.array(v)[1::2].tolist()
            db_h3 = [h3.geo_to_h3(p[1], p[0], Config.cell_res) for p in db_traj]
            db_lst.append(db_h3)
            db_trajs_raw.append(db_traj)

            i += 1

        with open(Config.dataset_file + '_raw_pts.pkl', 'wb') as fh:
            pickle.dump((query_trajs_raw, db_trajs_raw), fh, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info("_raw_dataset done.")

        # output_file_name = Config.dataset_file + '_newsimi_raw.pkl'
        # with open(output_file_name, 'wb') as fh:
        #     pickle.dump((query_lst, db_lst), fh, protocol=pickle.HIGHEST_PROTOCOL)
        #     logging.info("_raw_dataset done.")
        return

    # for varying downsampling rate
    # def _downsample_dataset(rate):
    #     unrate = 1 - rate  # preserved rate
    #     query_lst = []  # [N, len, 2]
    #     db_lst = []
    #     i = 0
    #     for _, v in test_trajs.wgs_seq.iteritems():
    #         if i < n_query:
    #             _q = np.array(v)[::2]
    #             _q_len = _q.shape[0]
    #             _idx = np.sort(np.random.choice(_q_len, math.ceil(_q_len * unrate), replace=False))
    #             # query_lst.append(_q[_idx].tolist())
    #             q_sampled = _q[_idx]
    #             q_h3 = [h3.geo_to_h3(p[1], p[0], Config.cell_res) for p in q_sampled]
    #             query_lst.append(q_h3)
    #
    #         _db = np.array(v)[1::2]
    #         _db_len = _db.shape[0]
    #         _idx = np.sort(np.random.choice(_db_len, math.ceil(_db_len * unrate), replace=False))
    #         # db_lst.append(_db[_idx].tolist())
    #         db_sampled = _db[_idx]
    #         db_h3 = [h3.geo_to_h3(p[1], p[0], Config.cell_res) for p in db_sampled]
    #         db_lst.append(db_h3)
    #
    #         i += 1
    #
    #     output_file_name = Config.dataset_file + '_newsimi_downsampling_' + str(rate) + '.pkl'
    #     with open(output_file_name, 'wb') as fh:
    #         pickle.dump((query_lst, db_lst), fh, protocol=pickle.HIGHEST_PROTOCOL)
    #         logging.info("_downsample_dataset done. rate={}".format(rate))
    #     return

    def _downsample_dataset(rate):
        unrate = 1 - rate
        query_lst, db_lst = [], []

        def sample_preserve_ends(seq):
            if len(seq) <= 2:
                return seq
            first, last = seq[0], seq[-1]
            mids = seq[1:-1]
            k = math.ceil(len(mids) * unrate)
            idx = np.sort(np.random.choice(len(mids), k, replace=False))
            return np.vstack([first, mids[idx], last])

        for i, v in enumerate(test_trajs.wgs_seq.values):
            traj = np.array(v)

            # query trajectories (every other starting at 0)
            if i < n_query:
                q_seq = traj[::2]
                q_sampled = sample_preserve_ends(q_seq)
                query_lst.append([h3.geo_to_h3(pt[1], pt[0], Config.cell_res) for pt in q_sampled])

            # db trajectories (every other starting at 1)
            db_seq = traj[1::2]
            db_sampled = sample_preserve_ends(db_seq)
            db_lst.append([h3.geo_to_h3(pt[1], pt[0], Config.cell_res) for pt in db_sampled])

        output = f"{Config.dataset_file}_newsimi_downsampling_{rate}.pkl"
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, 'wb') as fh:
            pickle.dump((query_lst, db_lst), fh, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"_downsample_dataset done. rate={rate}")

    # def _downsample_dataset(rate):
    #     unrate = 1 - rate  # preserved rate
    #     query_lst = []  # [N, len, 2]
    #     db_lst = []
    #     i = 0
    #
    #     for _, v in test_trajs.wgs_seq.iteritems():
    #         traj = np.array(v)
    #
    #         # Query trajectory
    #         if i < n_query:
    #             _q = traj[::2]
    #             _q_len = _q.shape[0]
    #
    #             if _q_len <= 2:
    #                 query_lst.append(_q.tolist())
    #             else:
    #                 n_keep = math.ceil(_q_len * unrate)
    #                 candidate_idx = np.arange(1, _q_len - 1)  # exclude first & last
    #                 n_sample = max(0, n_keep - 2)  # how many random in middle
    #
    #                 sampled_idx = np.sort(np.random.choice(candidate_idx, size=n_sample, replace=False))
    #                 full_idx = np.concatenate([[0], sampled_idx, [_q_len - 1]])
    #                 # query_lst.append(_q[full_idx].tolist())
    #                 q_sampled = _q[full_idx]
    #                 q_h3 = [h3.geo_to_h3(p[1], p[0], Config.cell_res) for p in q_sampled]
    #                 query_lst.append(q_h3)
    #
    #         # Database trajectory
    #         _db = traj[1::2]
    #         _db_len = _db.shape[0]
    #
    #         if _db_len <= 2:
    #             db_lst.append(_db.tolist())
    #         else:
    #             n_keep = math.ceil(_db_len * unrate)
    #             candidate_idx = np.arange(1, _db_len - 1)
    #             n_sample = max(0, n_keep - 2)
    #
    #             sampled_idx = np.sort(np.random.choice(candidate_idx, size=n_sample, replace=False))
    #             full_idx = np.concatenate([[0], sampled_idx, [_db_len - 1]])
    #             # db_lst.append(_db[full_idx].tolist())
    #             db_sampled = _db[full_idx]
    #             db_h3 = [h3.geo_to_h3(p[1], p[0], Config.cell_res) for p in db_sampled]
    #             db_lst.append(db_h3)
    #
    #         i += 1
    #
    #     output_file_name = Config.dataset_file + '_newsimi_downsampling_' + str(rate) + '.pkl'
    #     with open(output_file_name, 'wb') as fh:
    #         pickle.dump((query_lst, db_lst), fh, protocol=pickle.HIGHEST_PROTOCOL)
    #         logging.info("_downsample_dataset done. rate={}".format(rate))
    #
    #     return

    # def _downsample_dataset(rate):
    #     unrate = 1 - rate  # preserved rate
    #     query_lst = []  # [N, len, 2]
    #     db_lst = []
    #     i = 0
    #     for _, v in test_trajs.merc_seq.iteritems():
    #         traj = np.array(v)  # Convert to numpy array
    #
    #         _q = traj[::2]  # Query trajectory
    #         _db = traj[1::2]  # Database trajectory
    #         # Get anchor points using RDP separately for both
    #         rdp_mask_q = rdp(_q, epsilon=100, return_mask=True)
    #         rdp_mask_db = rdp(_db, epsilon=100, return_mask=True)
    #
    #         def downsample(traj, rdp_mask, unrate):
    #             """
    #             Downsamples a trajectory but keeps anchor points.
    #             """
    #             traj_len = traj.shape[0]
    #
    #             anchor_indices = np.where(rdp_mask)[0]
    #
    #             # Compute downsampling candidates (excluding anchor points)
    #             non_anchor_indices = np.where(~rdp_mask)[0]
    #             num_to_keep = max(1, int(len(non_anchor_indices) * unrate))
    #
    #             # Randomly select non-anchor points to keep
    #             sampled_indices = np.sort(np.random.choice(non_anchor_indices, num_to_keep, replace=False))
    #             # Merge anchor points and sampled points, ensuring order is maintained
    #             final_indices = np.sort(np.concatenate((anchor_indices, sampled_indices)))
    #             return traj[final_indices].tolist()
    #
    #         # Process queries
    #         if i < n_query:
    #             query_lst.append(downsample(_q, rdp_mask_q, unrate))
    #
    #         # Process database
    #         db_lst.append(downsample(_db, rdp_mask_db, unrate))
    #
    #         i += 1
    #
    #     output_file_name = Config.dataset_file + '_newsimi_downsampling_' + str(rate) + '.pkl'
    #     with open(output_file_name, 'wb') as fh:
    #         pickle.dump((query_lst, db_lst), fh, protocol=pickle.HIGHEST_PROTOCOL)
    #         logging.info("_downsample_dataset done. rate={}".format(rate))
    #     return



    # for varying distort rate
    def _distort_dataset(rate):
        query_lst = []  # [N, len, 2]
        db_lst = []
        i = 0
        for _, v in test_trajs.wgs_seq.iteritems():
            if i < n_query:
                _q = np.array(v)[::2]
                for _row in range(_q.shape[0]):
                    if random.random() < rate:
                        # print(_q[_row])
                        # _q[_row] = _q[_row] + [tool_funcs.truncated_rand(), tool_funcs.truncated_rand()]
                        noise_lon, noise_lat = tool_funcs.truncated_gps_noise(_q[_row, 1])
                        _q[_row] = _q[_row] + np.array([noise_lon, noise_lat])
                # query_lst.append(_q.tolist())
                query_h3 = [h3.geo_to_h3(p[1], p[0], Config.cell_res) for p in _q]
                query_lst.append(query_h3)

            _db = np.array(v)[1::2]
            for _row in range(_db.shape[0]):
                if random.random() < rate:
                    # _db[_row] = _db[_row] + [tool_funcs.truncated_rand(), tool_funcs.truncated_rand()]
                    noise_lon, noise_lat = tool_funcs.truncated_gps_noise(_db[_row, 1])
                    _db[_row] = _db[_row] + np.array([noise_lon, noise_lat])
            # db_lst.append(_db.tolist())
            db_h3 = [h3.geo_to_h3(p[1], p[0], Config.cell_res) for p in _db]
            db_lst.append(db_h3)

            i += 1

        output_file_name = Config.dataset_file + '_newsimi_distort_' + str(rate) + '.pkl'
        with open(output_file_name, 'wb') as fh:
            pickle.dump((query_lst, db_lst), fh, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info("_distort_dataset done. rate={}".format(rate))
        return

    # def _distort_dataset(rate):
    #     query_lst = []  # [N, len, 2]
    #     db_lst = []
    #     i = 0
    #
    #     for _, v in test_trajs.merc_seq.iteritems():
    #         traj = np.array(v)  # Convert to NumPy array
    #
    #         # First, sample every second point to create query and database sets
    #         _q = traj[::2]  # Query trajectory
    #         _db = traj[1::2]  # Database trajectory
    #
    #         # Get anchor points using RDP separately for both
    #         rdp_mask_q = rdp(_q, epsilon=100, return_mask=True)
    #         rdp_mask_db = rdp(_db, epsilon=100, return_mask=True)
    #
    #         def distort(traj, rdp_mask, rate):
    #             """
    #             Distorts a trajectory while keeping anchor points fixed.
    #             """
    #             traj_len = traj.shape[0]
    #
    #             for i in range(traj_len):
    #                 if not rdp_mask[i] and random.random() < rate:
    #                     # print(traj[i, 0])
    #                     # traj[i] = traj[i] + [tool_funcs.truncated_gps_noise(traj[i, 1]), tool_funcs.truncated_gps_noise(traj[i, 1])]
    #                     traj[i] = traj[i] + [tool_funcs.truncated_rand(), tool_funcs.truncated_rand()]
    #
    #             return traj.tolist()
    #
    #         # Process queries
    #         if i < n_query:
    #             query_lst.append(distort(_q, rdp_mask_q, rate))
    #
    #         # Process database
    #         db_lst.append(distort(_db, rdp_mask_db, rate))
    #
    #         i += 1
    #
    #     # Save the distorted dataset
    #     output_file_name = Config.dataset_file + '_newsimi_distort_' + str(rate) + '.pkl'
    #     with open(output_file_name, 'wb') as fh:
    #         pickle.dump((query_lst, db_lst), fh, protocol=pickle.HIGHEST_PROTOCOL)
    #         logging.info("_distort_dataset done. rate={}".format(rate))
    #
    #     return

    _raw_dataset()

    # for rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
    #     _downsample_dataset(rate)
    # for rate in [0.2, 0.4, 0.6, 0.8, 1.0]:
    #     _downsample_dataset(rate)

    # for rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
    #     _distort_dataset(rate)
    # for rate in [0.2, 0.4, 0.6, 0.8, 1.0]:
    #     _distort_dataset(rate)

    return









def generate_newsimi_merc_test_dataset():
    trajs = pd.read_pickle(Config.dataset_file)  # using test part only
    l = trajs.shape[0]
    n_query = 1000
    n_db = 100000
    test_idx = (int(l * 0.8), int(l * 0.8) + n_db)
    test_trajs = trajs[test_idx[0]: test_idx[1]]
    logging.info("Test trajs loaded.")

    # for varying db size
    def _raw_dataset():
        query_trajs_raw = []  # raw [lon, lat] sequences for queries
        db_trajs_raw = []
        query_lst = []  # [N, len, 2]
        db_lst = []
        i = 0
        for _, v in test_trajs.merc_seq.iteritems():
            if i < n_query:
                # query_lst.append(np.array(v)[::2].tolist())
                query_traj = np.array(v)[::2].tolist()
                # query_h3 = [h3.geo_to_h3(p[1], p[0], Config.cell_res) for p in query_traj]
                # query_lst.append(query_h3)
                query_trajs_raw.append(query_traj)

            # db_lst.append(np.array(v)[1::2].tolist())
            db_traj = np.array(v)[1::2].tolist()
            # db_h3 = [h3.geo_to_h3(p[1], p[0], Config.cell_res) for p in db_traj]
            # db_lst.append(db_h3/)
            db_trajs_raw.append(db_traj)

            i += 1

        with open(Config.dataset_file + '_raw_merc.pkl', 'wb') as fh:
            pickle.dump((query_trajs_raw, db_trajs_raw), fh, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info("_raw_dataset done.")

        return



    def _downsample_dataset(rate):
        unrate = 1 - rate
        query_lst, db_lst = [], []

        def sample_preserve_ends(seq):
            if len(seq) <= 2:
                return seq
            first, last = seq[0], seq[-1]
            mids = seq[1:-1]
            k = math.ceil(len(mids) * unrate)
            idx = np.sort(np.random.choice(len(mids), k, replace=False))
            return np.vstack([first, mids[idx], last])

        for i, v in enumerate(test_trajs.merc_seq.values):
            traj = np.array(v)

            # query trajectories (every other starting at 0)
            if i < n_query:
                q_seq = traj[::2]
                q_sampled = sample_preserve_ends(q_seq)
                # query_lst.append([h3.geo_to_h3(pt[1], pt[0], Config.cell_res) for pt in q_sampled])
                query_lst.append(q_sampled)

            # db trajectories (every other starting at 1)
            db_seq = traj[1::2]
            db_sampled = sample_preserve_ends(db_seq)
            # db_lst.append([h3.geo_to_h3(pt[1], pt[0], Config.cell_res) for pt in db_sampled])
            db_lst.append(db_sampled)

        output = f"{Config.dataset_file}_newsimi_downsampling_merc_{rate}.pkl"
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, 'wb') as fh:
            pickle.dump((query_lst, db_lst), fh, protocol=pickle.HIGHEST_PROTOCOL)
        logging.info(f"_downsample_dataset done. rate={rate}")




    # for varying distort rate
    def _distort_dataset(rate):
        query_lst = []  # [N, len, 2]
        db_lst = []
        i = 0
        for _, v in test_trajs.merc_seq.iteritems():
            if i < n_query:
                _q = np.array(v)[::2]
                for _row in range(_q.shape[0]):
                    if random.random() < rate:
                        # print(_q[_row])
                        _q[_row] = _q[_row] + [tool_funcs.truncated_rand(), tool_funcs.truncated_rand()]
                        # noise_lon, noise_lat = tool_funcs.truncated_gps_noise(_q[_row, 1])
                        # _q[_row] = _q[_row] + np.array([noise_lon, noise_lat])
                # query_lst.append(_q.tolist())
                # query_h3 = [h3.geo_to_h3(p[1], p[0], Config.cell_res) for p in _q]
                # query_lst.append(query_h3)
                query_lst.append(_q.tolist())

            _db = np.array(v)[1::2]
            for _row in range(_db.shape[0]):
                if random.random() < rate:
                    _db[_row] = _db[_row] + [tool_funcs.truncated_rand(), tool_funcs.truncated_rand()]
                    # noise_lon, noise_lat = tool_funcs.truncated_gps_noise(_db[_row, 1])
                    # _db[_row] = _db[_row] + np.array([noise_lon, noise_lat])
            # db_lst.append(_db.tolist())
            # db_h3 = [h3.geo_to_h3(p[1], p[0], Config.cell_res) for p in _db]
            # db_lst.append(db_h3)
            db_lst.append(_db.tolist())

            i += 1

        output_file_name = Config.dataset_file + '_newsimi_distort_merc_' + str(rate) + '.pkl'
        with open(output_file_name, 'wb') as fh:
            pickle.dump((query_lst, db_lst), fh, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info("_distort_dataset done. rate={}".format(rate))
        return



    # _raw_dataset()

    # for rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
    #     _downsample_dataset(rate)


    for rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
        _distort_dataset(rate)


    return









# ===calculate trajsimi distance matrix for trajsimi learning===
def traj_simi_computation(fn_name = 'hausdorff'):
    # 1. read trajs from file, and split to 3 datasets, and data normalization
    # 2. calculate simi in 3 datasets separately.
    # 3. dump 3 datasets into files

    logging.info("traj_simi_computation starts. fn={}".format(fn_name))
    _time = time.time()
    
    # 1.
    trains, evals, tests = read_trajsimi_traj_dataset(Config.dataset_file)
    trains, evals, tests = _normalization([trains, evals, tests])

    logging.info("traj dataset sizes. traj: trains/evals/tests={}/{}/{}" \
                .format(trains.shape[0], evals.shape[0], tests.shape[0]))

    # 2.
    fn = _get_simi_fn(fn_name)
    tests_simi = _simi_matrix(fn, tests)
    evals_simi = _simi_matrix(fn, evals)
    trains_simi = _simi_matrix(fn, trains) # [ [simi, simi, ... ], ... ]

    max_distance = max( max(map(max, trains_simi)), max(map(max, evals_simi)), max(map(max, tests_simi)) )

    _output_file = '{}_traj_simi_dict_{}.pkl'.format(Config.dataset_file, fn_name)
    with open(_output_file, 'wb') as fh:
        tup = trains_simi, evals_simi, tests_simi, max_distance
        pickle.dump(tup, fh, protocol = pickle.HIGHEST_PROTOCOL)
    
    logging.info("traj_simi_computation ends. @={:.3f}".format(time.time() - _time))
    return tup


def _normalization(lst_df):
    # lst_df: [df, df, df]
    xs = []
    ys = []
    for df in lst_df:
        for _, v in df.wgs_seq.iteritems():
            arr = np.array(v)
            xs.append(arr[:,0])
            ys.append(arr[:,1])

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    mean = np.array([xs.mean(), ys.mean()])
    std = np.array([xs.std(), ys.std()])

    for i in range(len(lst_df)):
        lst_df[i].wgs_seq = lst_df[i].wgs_seq.apply(lambda lst: ( (np.array(lst)-mean)/std ).tolist())

    return lst_df


def _get_simi_fn(fn_name):
    fn =  {'lcss': tdist.lcss, 'edr': tdist.edr, 'frechet': tdist.frechet,
            'discret_frechet': tdist.discret_frechet,
            'hausdorff': tdist.hausdorff, 'edwp': edwp}.get(fn_name, None)
    if fn_name == 'lcss' or fn_name == 'edr':
        fn = partial(fn, eps = Config.test_exp1_lcss_edr_epsilon)
    return fn


def _simi_matrix(fn, df):
    _time = time.time()

    l = df.shape[0]
    batch_size = 50
    assert l % batch_size == 0

    # parallel init
    tasks = []
    for i in range(math.ceil(l / batch_size)):
        if i < math.ceil(l / batch_size) - 1:
            tasks.append( (fn, df, list(range(batch_size * i, batch_size * (i+1)))) )
        else:
            tasks.append( (fn, df, list(range(batch_size * i, l))) )
    

    num_cores = int(mp.cpu_count()) - 6
    assert num_cores > 0
    logging.info("pool.size={}".format(num_cores))
    pool = mp.Pool(num_cores)
    lst_simi = pool.starmap(_simi_comp_operator, tasks)
    pool.close()

    # extend lst_simi to matrix simi and pad 0s
    lst_simi = sum(lst_simi, [])
    for i, row_simi in enumerate(lst_simi):
        lst_simi[i] = [0]*(i+1) + row_simi
    assert sum(map(len, lst_simi)) == l ** 2
    logging.info('simi_matrix computation done., @={}, #={}'.format(time.time() - _time, len(lst_simi)))

    return lst_simi


# async operator
def _simi_comp_operator(fn, df_trajs, sub_idx):
    simi = []
    l = df_trajs.shape[0]
    for _i in sub_idx:
        t_i = np.array(df_trajs.iloc[_i].wgs_seq)
        simi_row = []
        for _j in range(_i + 1, l):
            t_j = np.array(df_trajs.iloc[_j].wgs_seq)
            simi_row.append( float(fn(t_i, t_j)) )
        simi.append(simi_row)
    logging.debug('simi_comp_operator ends. sub_idx=[{}:{}], pid={}' \
                    .format(sub_idx[0], sub_idx[-1], os.getpid()))
    return simi


def read_ft():
    logging.info('[Load traj dataset] START.')
    _time = time.time()
    trajs = pd.read_pickle("../data/porto_20200_new")

    l = trajs.shape[0]
    print(l)
    # train_idx = (int(l*0), 200000)        # for porto
    # train_idx = (int(l * 0), 70000)  # for t-drive
    # train_idx = (int(l * 0), int(l*0.7))
    # train_idx = (int(l * 0), 35000)         # for geolife
    offset_idx = int(trajs.shape[0] * 0.7)  # use eval dataset
    print(offset_idx)
    df_trajs = trajs.iloc[offset_idx: offset_idx + 10000]
    # print(df_trajs)
    # print(df_trajs["wgs_seq"])
    # def array_to_polyline(coordinates):
    #     # Create the trajectory string for the entire array
    #     trajectory = [f"[{coord[0]}, {coord[1]}]" for coord in coordinates]
    #     trajectory_str = f"[{', '.join(trajectory)}]"
    #     return trajectory_str
    seq_list = df_trajs['wgs_seq'].tolist()
    # df_trajs['POLYLINE'] = df_trajs["wgs_seq"].apply(array_to_polyline)
    # print(df_trajs['POLYLINE'])
    # print(df_trajs.shape)
    # df_trajs.to_csv("porto_ft.csv", index=False)
    with open("porto_ft.csv", 'wb') as f:
        pickle.dump(seq_list, f)


def read_ft_CLEAR():
    logging.info('[Load traj dataset] START.')
    _time = time.time()
    trajs = pd.read_pickle("../data/porto_20200_new")

    l = trajs.shape[0]
    print(l) 
    # train_idx = (int(l*0), 200000)        # for porto
    # train_idx = (int(l * 0), 70000)  # for t-drive
    # train_idx = (int(l * 0), int(l*0.7))
    # train_idx = (int(l * 0), 35000)         # for geolife
    offset_idx = int(trajs.shape[0] * 0.7)  # use eval dataset
    print(offset_idx)
    df_trajs = trajs.iloc[offset_idx: offset_idx + 10000]

    seq_list = df_trajs['wgs_seq'].tolist()
  
    with open("porto_ft.pkl", 'wb') as f:
        pickle.dump(seq_list, f)
    print(seq_list[0])

def POI_loc(file, output_file='porto_20200_new_poi.pkl'):
    poi = pickle.load(open(file, 'rb'))
    poi = pd.DataFrame(poi)
    # print(poi)

    categories = poi['category'].unique()
    num_cate = len(categories)
    cate_dict = {}
    for i, cate in enumerate(categories):
        cate_dict[cate] = i
    print(cate_dict)
    """
    prepare the cellspace.
    """
    # x_min, y_min = lonlat2meters(Config.min_lon, Config.min_lat)
    # x_max, y_max = lonlat2meters(Config.max_lon, Config.max_lat)
    # x_min -= Config.cellspace_buffer
    # y_min -= Config.cellspace_buffer
    # x_max += Config.cellspace_buffer
    # y_max += Config.cellspace_buffer
    #
    # cell_size = int(Config.cell_size)
    # cs = CellSpace(cell_size, cell_size, x_min, y_min, x_max, y_max)
    #
    # poi[['merc_x', 'merc_y']] = poi.apply(lambda row: pd.Series(lonlat2meters(row['lon'], row['lat'])), axis=1)
    # poi['cellID'] = poi.apply(lambda row: cs.get_cellid_by_point(row['merc_x'], row['merc_y']), axis=1)
    # print(poi)


    poi['cellID'] = poi.apply(lambda row: h3.geo_to_h3(row['lat'], row['lon'], Config.cell_res), axis=1)
    poi = poi.sort_values("cellID").reset_index(drop=True)

    print(poi['cellID'])
    """
    map the POIs on cells by cellID.
    """
    poi_dict = {}
    grouped_loc = poi.groupby(['cellID'])
    for cell, group in grouped_loc:
        # print(f"processing {cell} cell....")
        embedding = np.zeros(num_cate)
        category_counts = group['category'].map(cate_dict).value_counts()
        # print(category_counts)
        for category_idx, count in category_counts.items():
            embedding[category_idx] = count
            # print(embedding)
        poi_dict[cell] = embedding.tolist()

    # print(poi_dict)
    index_mapping = pickle.load(open(Config.index_mapping_file, 'rb'))

    int_poi_dict = {
        index_mapping[h3_id]: poi_dict[h3_id]
        for h3_id in poi_dict
        if h3_id in index_mapping
    }
    print(int_poi_dict)
    with open(output_file, "wb") as f:
        pickle.dump(int_poi_dict, f)


def get_8_neighbors(x, y, cellspace):
    """
    Returns 8-connected neighbors of a grid cell (x, y), excluding out-of-bound ones.
    """
    directions = [(-1, -1), (0, -1), (1, -1),
                  (-1, 0),           (1, 0),
                  (-1, 1),  (0, 1),  (1, 1)]

    neighbors = []
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < cellspace.x_size and 0 <= ny < cellspace.y_size:
            neighbors.append((nx, ny))
    return neighbors

def generate_cellid_neighbors():
    x_min, y_min = lonlat2meters(Config.min_lon, Config.min_lat)
    x_max, y_max = lonlat2meters(Config.max_lon, Config.max_lat)
    x_min -= Config.cellspace_buffer
    y_min -= Config.cellspace_buffer
    x_max += Config.cellspace_buffer
    y_max += Config.cellspace_buffer

    cell_size = int(Config.cell_size)
    cellspace = CellSpace(cell_size, cell_size, x_min, y_min, x_max, y_max)
    cellid_neighbors = {}
    for x in range(cellspace.x_size):
        for y in range(cellspace.y_size):
            cellid = cellspace.get_cellid_by_xyidx(x, y)
            neighbors = get_8_neighbors(x, y, cellspace)
            neighbor_ids = []
            for nx, ny in neighbors:
                neighbor_id = cellspace.get_cellid_by_xyidx(nx, ny)
                neighbor_ids.append(neighbor_id)
            cellid_neighbors[cellid] = neighbor_ids

    with open(Config.neighbours_file, "wb") as f:
        pickle.dump(cellid_neighbors, f)


# nohup python ./preprocessing_porto.py &> ../result &
if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG,
                        format = "[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
                        handlers = [logging.FileHandler(Config.root_dir+'/log/'+tool_funcs.log_file_name(), mode = 'w'),
                                    logging.StreamHandler()]
                        )
    Config.dataset = 'porto'
    Config.post_value_updates()

    # data_stats()
    clean_and_output_data()

    # process_long_traj()
    # generate_cellid_neighbors()
    # with open(Config.neighbours_file, "rb") as f:
    #     cellid_neighbors = pickle.load(f)
    #     print(cellid_neighbors)
    # init_cellspace()
    generate_newsimi_test_dataset()
    # generate_longsimi_test_dataset()
    traj_simi_computation('hausdorff') # edr edwp discret_frechet hausdorff
    traj_simi_computation('edr')
    traj_simi_computation('discret_frechet')
    traj_simi_computation('lcss')


