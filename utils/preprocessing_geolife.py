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
from tqdm import tqdm
import traj_dist.distance as tdist
import multiprocessing as mp
from functools import partial
# from datetime import datetime
import requests
import glob
from datetime import timedelta

from config import Config
from utils import tool_funcs
from utils.cellspace import CellSpace
from utils.tool_funcs import lonlat2meters
from model.node2vec_ import train_node2vec
from utils.edwp import edwp
from utils.data_loader import read_trajsimi_traj_dataset
# from utils.traj import *
import datetime
import re

from pybind11_rdp import rdp


def unix2dt(dt_object):
    # Convert to datetime object
    # dt_object = datetime.datetime.fromtimestamp(int(time))
    # print(dt_object)
    # Extracting detailed components from the datetime object
    month = dt_object.month
    day = dt_object.day
    weekday = dt_object.weekday()  # Monday is 0 and Sunday is 6
    hour = dt_object.hour
    minute = dt_object.minute
    # print(month, day, weekday, hour, minute)
    return month, day, weekday, hour, minute

def expand_timestamps(row):
    # Retrieve the list of UNIX timestamps from the row
    # unix_timestamps = row['Timestamps']
    raw = row['Timestamps'].strip('[]')
    timestamps = re.findall(r"Timestamp\('([^']+)'\)", raw)
    # print(timestamps)
    # Convert each UNIX timestamp to a detailed datetime component
    # return [unix2dt(ts) for ts in timestamps]
    return [unix2dt(datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')) for ts in timestamps]


def inrange(lon, lat):
    if lon <= Config.min_lon or lon >= Config.max_lon \
            or lat <= Config.min_lat or lat >= Config.max_lat:
        return False
    return True


def read_plt_file(file_path):
    """Helper function to read plt file and return DataFrame"""
    data = pd.read_csv(file_path, skiprows=6, header=None,
                       names=['Latitude', 'Longitude', 'Unused', 'Altitude', 'Days', 'Date', 'Time'])
    return data


def process_geolife(root_dir):
    labeled_df = pd.DataFrame()
    unlabeled_df = pd.DataFrame()

    # Determine the number of directories for tqdm progress bar
    total_dirs = [os.path.join(root, 'Trajectory') for root, dirs, files in os.walk(root_dir) if 'Trajectory' in dirs]
    pbar = tqdm(total=len(total_dirs), desc='Processing trajectories')

    for root, dirs, files in os.walk(root_dir):
        if 'Trajectory' in dirs:
            trajectory_dir = os.path.join(root, 'Trajectory')
            has_labels = 'labels.txt' in files
            # Read all plt files in the Trajectory directory
            for plt_file in os.listdir(trajectory_dir):
                file_path = os.path.join(trajectory_dir, plt_file)
                if plt_file.endswith('.plt'):
                    data = read_plt_file(file_path)
                    if has_labels:
                        labeled_df = pd.concat([labeled_df, data], ignore_index=True)
                    else:
                        unlabeled_df = pd.concat([unlabeled_df, data], ignore_index=True)
            pbar.update(1)

    pbar.close()

    # Save to CSV
    labeled_df.to_csv('labeled_trajectories.csv', index=False)
    unlabeled_df.to_csv('unlabeled_trajectories.csv', index=False)

    print(labeled_df)
    print(unlabeled_df)

def create_polyline(group):
    trajectory = group[['Longitude', 'Latitude']].apply(
        lambda row: f"[{row['Longitude']}, {row['Latitude']}]", axis=1).tolist()
    trajectory_str = f"[{', '.join(trajectory)}]"
    return trajectory_str


def segment_unlabeled(csv_path):
    data = pd.read_csv(csv_path, parse_dates={'Timestamp': ['Date', 'Time']})

    # Calculate differences between successive timestamps and check if they exceed 10 minutes
    data['Trajectory_ID'] = (data['Timestamp'].diff() > pd.Timedelta(minutes=10)).cumsum()

    # Apply a function to split large groups
    trajectories = []
    for _, group in data.groupby('Trajectory_ID'):
        split_and_add(group, trajectories)

    # Convert list of trajectories into a DataFrame
    # trajectories_df = pd.DataFrame(trajectories, columns=['Polyline'])
    trajectories_df = pd.DataFrame(trajectories)

    # Save DataFrame to CSV
    trajectories_df.to_csv('geolife_unlabeled.csv', index=False)
    print(trajectories_df.head(5))
    return trajectories_df

def split_and_add(group, trajectories, max_points=200):
    # Function to split trajectories into segments of at most max_points
    for start in range(0, len(group), max_points):
        end = start + max_points
        # Make sure not to exceed the group's length
        segment = group.iloc[start:min(end, len(group))]
        trajectory = create_polyline(segment)
        # Get timestamps for each point in the segment
        timestamps = segment['Timestamp'].tolist()
        # Append the segment information including the polyline and timestamps
        trajectories.append({
            'Polyline': trajectory,
            'Timestamps': timestamps
        })


def read_labels(root_dir):
    labels = []
    # Walk through all directories to find labels.txt
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file == 'labels.txt':
                # Read the labels file
                path = os.path.join(subdir, file)
                temp_df = pd.read_csv(path, delimiter='\t')
                # Convert strings to datetime objects for easier manipulation later
                temp_df['Start Time'] = pd.to_datetime(temp_df['Start Time'])
                temp_df['End Time'] = pd.to_datetime(temp_df['End Time'])
                labels.append(temp_df)
    if labels:
        labels_df = pd.concat(labels, ignore_index=True)
        # Filter by transportation modes and correct column name usage
        return labels_df[labels_df['Transportation Mode'].isin(['walk', 'bus', 'bike', 'car'])]
    else:
        return pd.DataFrame()


import pandas as pd
from tqdm import tqdm

def process_trajectories(labels_df, trajectories_df_path):
    # Assume 'Timestamp' column in trajectories_df is already in datetime format
    trajectories_df = pd.read_csv(trajectories_df_path, parse_dates={'Timestamp': ['Date', 'Time']})
    final_trajectories = []
    for index, label in tqdm(labels_df.iterrows(), total=labels_df.shape[0], desc="Processing labels"):
        # Filtering trajectories that fall within the label's timeframe
        within_range = trajectories_df[
            (trajectories_df['Timestamp'] >= label['Start Time']) &
            (trajectories_df['Timestamp'] <= label['End Time'])
        ]

        # Segment the trajectories if their span is greater than 10 minutes
        if not within_range.empty:
            segments = segment_by_duration(within_range, duration_minutes=10)
            for segment in segments:
                if 20 <= len(segment) <= 100:
                    final_trajectories.append({
                        'Mode': label['Transportation Mode'],
                        'Trajectory': create_polyline(segment),
                        'Timestamps': segment['Timestamp'].tolist()  # Save timestamps
                    })
                elif len(segment) > 100:
                    sampled_segment = segment.sample(n=100)
                    final_trajectories.append({
                        'Mode': label['Transportation Mode'],
                        'Trajectory': create_polyline(sampled_segment),
                        'Timestamps': sampled_segment['Timestamp'].tolist()  # Save timestamps for sampled
                    })

    final = pd.DataFrame(final_trajectories)
    final.to_csv("geolife_labeled.csv", index=False)
    mode_counts = final['Mode'].value_counts()
    print("Number of trajectories per transportation mode:")
    for mode, count in mode_counts.items():
        print(f"{mode}: {count}")



def segment_by_duration(df, duration_minutes=10):
    segments = []
    start_index = 0  # Use an index rather than the DataFrame row itself

    # Iterate over the DataFrame rows using index
    for i in range(1, len(df)):
        # Check if the time difference exceeds the threshold
        if (df.iloc[i]['Timestamp'] - df.iloc[start_index]['Timestamp']).total_seconds() / 60 > duration_minutes:
            # Append the segment from start_index to i (not inclusive of i)
            segments.append(df.iloc[start_index:i])
            start_index = i  # Update start_index to current position

    # Append the last segment from the last start_index to the end of the DataFrame
    segments.append(df.iloc[start_index:])  # This includes the last segment

    return segments

# def process_geolife(directory):
#     """
#     Count the number of .plt files and labels.txt files in a given directory and its subdirectories.
#
#     Parameters:
#     - directory (str): Path to the directory to search.
#
#     Returns:
#     - tuple: (number of .plt files, number of labels.txt files)
#     """
#     total_plt_count = 0
#     labels_count = 0
#     plt_in_labels_dir_count = 0
#
#     # Walk through all directories and files in the root directory
#     for root, dirs, files in os.walk(directory):
#         # Count total .plt files
#         for file in files:
#             if file.endswith('.plt'):
#                 total_plt_count += 1
#
#         # Check if labels.txt is in the current directory
#         if 'labels.txt' in files:
#             labels_count += 1
#             # Path to the 'Trajectory' subdirectory
#             trajectory_dir = os.path.join(root, 'Trajectory')
#             # Count the .plt files in the 'Trajectory' directory
#             if os.path.exists(trajectory_dir):
#                 for traj_file in os.listdir(trajectory_dir):
#                     if traj_file.endswith('.plt'):
#                         plt_in_labels_dir_count += 1
#
#     return total_plt_count, labels_count, plt_in_labels_dir_count

import h3
def clean_and_output_data():
    _time = time.time()
    # cellspace = pickle.load(open(Config.dataset_cell_file, 'rb'))
    # embs = pickle.load(open(Config.dataset_embs_file, 'rb'))
    # https://archive.ics.uci.edu/ml/machine-learning-databases/00339/
    # download train.csv.zip and unzip it. rename train.csv to porto.csv
    dfraw = pd.read_csv(Config.root_dir + '/data/geolife.csv')
    # print(dfraw)
    dfraw = dfraw.rename(columns = {"Polyline": "wgs_seq"})
    print(dfraw.columns)
    # dfraw = dfraw[dfraw.MISSING_DATA == False]
    # print(dfraw.groupby("TAXI_ID").size())
    # print(dfraw["TIMESTAMP"].loc[0])
    # length requirement
    # print(dfraw.head())

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
    # convert to Mercator
    dfraw['merc_seq'] = dfraw.wgs_seq.apply(lambda traj: [list(lonlat2meters(p[0], p[1])) for p in traj])
    dfraw['DateTime'] = dfraw.apply(expand_timestamps, axis=1)
    dfraw['hex'] = dfraw.wgs_seq.apply(lambda traj: [h3.geo_to_h3(p[1], p[0], Config.cell_res) for p in traj])

    # dfraw['traj_p'] = dfraw['merc_seq'].apply(lambda traj: convert_trajectory(traj, cellspace))

    logging.info('Preprocessed-output. #traj={}'.format(dfraw.shape[0]))
    dfraw = dfraw[['trajlen', 'wgs_seq', 'merc_seq', 'hex', 'DateTime']].reset_index(drop = True)
    print(dfraw.columns)
    dfraw.to_pickle(Config.dataset_file)
    logging.info('Preprocess end. @={:.0f}'.format(time.time() - _time))
    # print(dfraw.head(5))
    print(dfraw['hex'].loc[:10])
    return


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


def generate_newsimi_test_dataset():
    trajs = pd.read_pickle(Config.dataset_file)  # using test part only
    # print(Config.dataset_file)
    l = trajs.shape[0]
    print(l)
    n_query = 1000
    n_db = 10000
    test_idx = (int(l * 0.8), int(l * 0.8) + n_db)
    test_trajs = trajs[test_idx[0]: test_idx[1]]
    logging.info("Test trajs loaded.")

    # for varying db size
    # def _raw_dataset():
    #     query_lst = []  # [N, len, 2]
    #     db_lst = []
    #     i = 0
    #     for _, v in test_trajs.wgs_seq.iteritems():
    #         if i < n_query:
    #             # query_lst.append(np.array(v)[::2].tolist())
    #             query_traj = np.array(v)[::2].tolist()
    #             query_h3 = [h3.geo_to_h3(p[1], p[0], Config.cell_res) for p in query_traj]
    #             query_lst.append(query_h3)

    #         # db_lst.append(np.array(v)[1::2].tolist())
    #         db_traj = np.array(v)[1::2].tolist()
    #         db_h3 = [h3.geo_to_h3(p[1], p[0], Config.cell_res) for p in db_traj]
    #         db_lst.append(db_h3)

    #         i += 1

    #     output_file_name = Config.dataset_file + '_newsimi_raw.pkl'
    #     with open(output_file_name, 'wb') as fh:
    #         pickle.dump((query_lst, db_lst), fh, protocol=pickle.HIGHEST_PROTOCOL)
    #         logging.info("_raw_dataset done.")
    #     return

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

    # for varying downsampling rate
    def _downsample_dataset(rate):
        unrate = 1 - rate  # preserved rate
        query_lst = []  # [N, len, 2]
        db_lst = []

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

            if i < n_query:
                q_seq = traj[::2]
                q_sampled = sample_preserve_ends(q_seq)
                query_lst.append([h3.geo_to_h3(pt[1], pt[0], Config.cell_res) for pt in q_sampled])


            # _db = np.array(v)[1::2]
            # _db_len = _db.shape[0]
            # _idx = np.sort(np.random.choice(_db_len, math.ceil(_db_len * unrate), replace=False))
            # db_lst.append(_db[_idx].tolist())
            db_seq = traj[1::2]
            db_sampled = sample_preserve_ends(db_seq)
            db_lst.append([h3.geo_to_h3(pt[1], pt[0], Config.cell_res) for pt in db_sampled])

            i += 1

        output_file_name = Config.dataset_file + '_newsimi_downsampling_' + str(rate) + '.pkl'
        with open(output_file_name, 'wb') as fh:
            pickle.dump((query_lst, db_lst), fh, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info("_downsample_dataset done. rate={}".format(rate))
        return

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
    #         rdp_mask_q = rdp(_q, epsilon=300, return_mask=True)
    #         rdp_mask_db = rdp(_db, epsilon=300, return_mask=True)
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
    #     for _, v in test_trajs.wgs_seq.iteritems():
    #         traj = np.array(v)  # Convert to NumPy array
    #
    #         # First, sample every second point to create query and database sets
    #         _q = traj[::2]  # Query trajectory
    #         _db = traj[1::2]  # Database trajectory
    #
    #         # Get anchor points using RDP separately for both
    #         rdp_mask_q = rdp(_q, epsilon=300, return_mask=True)
    #         rdp_mask_db = rdp(_db, epsilon=300, return_mask=True)
    #
    #         def distort(traj, rdp_mask, rate):
    #             """
    #             Distorts a trajectory while keeping anchor points fixed.
    #             """
    #             traj_len = traj.shape[0]
    #
    #             for i in range(traj_len):
    #                 if not rdp_mask[i] and random.random() < rate:
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


# ===calculate trajsimi distance matrix for trajsimi learning===
def traj_simi_computation(fn_name='hausdorff'):
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
    trains_simi = _simi_matrix(fn, trains)  # [ [simi, simi, ... ], ... ]

    max_distance = max(max(map(max, trains_simi)), max(map(max, evals_simi)), max(map(max, tests_simi)))

    _output_file = '{}_traj_simi_dict_{}.pkl'.format(Config.dataset_file, fn_name)
    with open(_output_file, 'wb') as fh:
        tup = trains_simi, evals_simi, tests_simi, max_distance
        pickle.dump(tup, fh, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info("traj_simi_computation ends. @={:.3f}".format(time.time() - _time))
    return tup


def _normalization(lst_df):
    # lst_df: [df, df, df]
    xs = []
    ys = []
    for df in lst_df:
        for _, v in df.merc_seq.iteritems():
            arr = np.array(v)
            xs.append(arr[:, 0])
            ys.append(arr[:, 1])

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    mean = np.array([xs.mean(), ys.mean()])
    std = np.array([xs.std(), ys.std()])

    for i in range(len(lst_df)):
        lst_df[i].merc_seq = lst_df[i].merc_seq.apply(lambda lst: ((np.array(lst) - mean) / std).tolist())

    return lst_df


def _get_simi_fn(fn_name):
    fn = {'lcss': tdist.lcss, 'edr': tdist.edr, 'frechet': tdist.frechet,
          'discret_frechet': tdist.discret_frechet,
          'hausdorff': tdist.hausdorff, 'edwp': edwp}.get(fn_name, None)
    if fn_name == 'lcss' or fn_name == 'edr':
        fn = partial(fn, eps=Config.test_exp1_lcss_edr_epsilon)
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
            tasks.append((fn, df, list(range(batch_size * i, batch_size * (i + 1)))))
        else:
            tasks.append((fn, df, list(range(batch_size * i, l))))

    num_cores = int(mp.cpu_count()) - 6
    assert num_cores > 0
    logging.info("pool.size={}".format(num_cores))
    pool = mp.Pool(num_cores)
    lst_simi = pool.starmap(_simi_comp_operator, tasks)
    pool.close()

    # extend lst_simi to matrix simi and pad 0s
    lst_simi = sum(lst_simi, [])
    for i, row_simi in enumerate(lst_simi):
        lst_simi[i] = [0] * (i + 1) + row_simi
    assert sum(map(len, lst_simi)) == l ** 2
    logging.info('simi_matrix computation done., @={}, #={}'.format(time.time() - _time, len(lst_simi)))

    return lst_simi


# async operator
def _simi_comp_operator(fn, df_trajs, sub_idx):
    simi = []
    l = df_trajs.shape[0]
    for _i in sub_idx:
        t_i = np.array(df_trajs.iloc[_i].merc_seq)
        simi_row = []
        for _j in range(_i + 1, l):
            t_j = np.array(df_trajs.iloc[_j].merc_seq)
            val = fn(t_i, t_j)
            if val is not None:
                simi_row.append(float(val))
        simi.append(simi_row)
    logging.debug('simi_comp_operator ends. sub_idx=[{}:{}], pid={}' \
                  .format(sub_idx[0], sub_idx[-1], os.getpid()))
    return simi

def read_ft():
    logging.info('[Load traj dataset] START.')
    _time = time.time()
    trajs = pd.read_pickle("../data/geolife")

    l = trajs.shape[0]
    print(l)
    # train_idx = (int(l*0), 200000)        # for porto
    # train_idx = (int(l * 0), 70000)  # for t-drive
    # train_idx = (int(l * 0), int(l*0.7))
    # train_idx = (int(l * 0), 35000)         # for geolife
    offset_idx = int(trajs.shape[0] * 0.7)  # use eval dataset
    print(offset_idx)
    df_trajs = trajs.iloc[offset_idx: offset_idx + 5000]
    # print(df_trajs)
    # print(df_trajs["wgs_seq"])
    def array_to_polyline(coordinates):
        # Create the trajectory string for the entire array
        trajectory = [f"[{coord[0]}, {coord[1]}]" for coord in coordinates]
        trajectory_str = f"[{', '.join(trajectory)}]"
        return trajectory_str

    df_trajs['POLYLINE'] = df_trajs["wgs_seq"].apply(array_to_polyline)
    print(df_trajs['POLYLINE'])
    print(df_trajs.shape)
    df_trajs.to_csv("geolife_ft.csv", index=False)

def read_ft_CLEAR():
    logging.info('[Load traj dataset] START.')
    _time = time.time()
    trajs = pd.read_pickle("../data/geolife/geolife")

    l = trajs.shape[0]
    print(l)
    # train_idx = (int(l*0), 200000)        # for porto
    # train_idx = (int(l * 0), 70000)  # for t-drive
    # train_idx = (int(l * 0), int(l*0.7))
    # train_idx = (int(l * 0), 35000)         # for geolife
    offset_idx = int(trajs.shape[0] * 0.7)  # use eval dataset
    print(offset_idx)
    df_trajs = trajs.iloc[offset_idx: offset_idx + 5000]
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
    # df_trajs.to_csv("geolife_ft.csv", index=False)
    with open("geolife_ft.pkl", 'wb') as f:
        pickle.dump(seq_list, f)
    print(seq_list[0])

def clean_label():
    _time = time.time()

    # root_dir = os.path.dirname(__file__)
    csv_input_path = '../data/geolife_labeled.csv'
    # dataset_file = root_dir + '../data/geolife_labeled.pkl'

    dfraw = pd.read_csv(csv_input_path)
    dfraw = dfraw.rename(columns={"Trajectory": "wgs_seq"})

    dfraw.wgs_seq = dfraw.wgs_seq.apply(literal_eval)
    dfraw['trajlen'] = dfraw.wgs_seq.apply(lambda traj: len(traj))
    dfraw = dfraw[(dfraw.trajlen >= Config.min_traj_len) & (dfraw.trajlen <= Config.max_traj_len)]
    logging.info('Preprocessed-rm length. #traj={}'.format(dfraw.shape[0]))

    # range requirement
    dfraw['inrange'] = dfraw.wgs_seq.map(
        lambda traj: sum([inrange(p[0], p[1]) for p in traj]) == len(traj))  # True: valid
    dfraw = dfraw[dfraw.inrange == True]
    logging.info('Preprocessed-rm range. #traj={}'.format(dfraw.shape[0]))

    # convert to Mercator
    dfraw['merc_seq'] = dfraw.wgs_seq.apply(lambda traj: [list(lonlat2meters(p[0], p[1])) for p in traj])
    dfraw['DateTime'] = dfraw.apply(expand_timestamps, axis=1)
    logging.info('Preprocessed-output. #traj={}'.format(dfraw.shape[0]))

    dfraw = dfraw[['trajlen', 'wgs_seq', 'merc_seq', "DateTime", 'Mode']].reset_index(drop=True)

    dfraw.to_pickle(Config.labeled_traj_file)
    logging.info('Preprocess end. @={:.0f}'.format(time.time() - _time))
    # print(dfraw.head(5))
    return


def retrieve_poi():
    node_file = Config.poi_node_file
    way_file = Config.poi_way_file

    poi_nodes = pickle.load(open(node_file, 'rb'))
    print(poi_nodes.shape)
    # print(poi_nodes.head())

    type_counts = poi_nodes.groupby('category').size()
    print(type_counts)
    cutoff_date = datetime.datetime(2012, 8, 31)

    x_min, y_min = lonlat2meters(Config.min_lon, Config.min_lat)
    x_max, y_max = lonlat2meters(Config.max_lon, Config.max_lat)
    x_min -= Config.cellspace_buffer
    y_min -= Config.cellspace_buffer
    x_max += Config.cellspace_buffer
    y_max += Config.cellspace_buffer

    cell_size = int(Config.cell_size)
    cs = CellSpace(cell_size, cell_size, x_min, y_min, x_max, y_max)

    batch_size = 100  # Number of IDs to query at once
    total = poi_nodes.shape[0]
    batches = [poi_nodes.iloc[i:i + batch_size] for i in range(0, total, batch_size)]


    filtered_rows = []
    overpass_url = "http://overpass-api.de/api/interpreter"
    count = 0
    for batch in tqdm(batches, desc="Processing batches of POIs"):
        node_ids = ','.join(batch['id'].astype(str))
        # Define the query to fetch the history of the node
        query = f"""
                [output:json];
                node(id:{node_ids});
                out meta;
                """

        # Send the request
        response = requests.get(overpass_url, params={'data': query})
        data = response.json()

        # Print the result
        # print(json.dumps(data, indent=2))
        if 'elements' not in data:
            print("no data found")
        else:
            for element in data['elements']:
                timestamp = element.get('timestamp', 'No timestamp available')
                tags = element.get('tags', {})
                print(f"Timestamp: {timestamp}, Tags: {tags}")
                date_string = timestamp[:-1]
                date_obj = datetime.datetime.fromisoformat(date_string)
                poi_x, poi_y = lonlat2meters(element['lon'], element['lat'])
                if cs.x_min <= poi_x <= cs.x_max and cs.y_min <= poi_y <= cs.y_max:
                    if date_obj < cutoff_date:
                        row = batch[batch['id'] == element['id']].iloc[0]
                        filtered_rows.append(row)
                        print("POI added before August 31, 2012")
                        count += 1


    print(f"Number of POIs filtered: {count}")

    filtered_pois = pd.DataFrame(filtered_rows)
    filtered_pois.to_pickle('../data/geolife_filtered_poi.pkl')
    print("Filtered POIs saved to '../data/geolife_filtered_poi.pkl'")

def POI_loc(file, output_file='geolife_poi.pkl'):
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
    x_min, y_min = lonlat2meters(Config.min_lon, Config.min_lat)
    x_max, y_max = lonlat2meters(Config.max_lon, Config.max_lat)
    x_min -= Config.cellspace_buffer
    y_min -= Config.cellspace_buffer
    x_max += Config.cellspace_buffer
    y_max += Config.cellspace_buffer

    cell_size = int(Config.cell_size)
    cs = CellSpace(cell_size, cell_size, x_min, y_min, x_max, y_max)

    poi[['merc_x', 'merc_y']] = poi.apply(lambda row: pd.Series(lonlat2meters(row['lon'], row['lat'])), axis=1)
    poi['cellID'] = poi.apply(lambda row: cs.get_cellid_by_point(row['merc_x'], row['merc_y']), axis=1)
    # print(poi)

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

    print(poi_dict)

    with open(output_file, "wb") as f:
        pickle.dump(poi_dict, f)


if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG,
                        format = "[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
                        handlers = [logging.FileHandler(Config.root_dir+'/exp/log/'+tool_funcs.log_file_name(), mode = 'w'),
                                    logging.StreamHandler()]
                        )
    Config.dataset = 'geolife'
    Config.post_value_updates()
    # process_geolife('../../Geolife Trajectories 1.3')
    # segment_unlabeled('unlabeled_trajectories.csv')
    # raw_labeled = read_labels('../../Geolife Trajectories 1.3')
        # process_trajectories(raw_labeled, "labeled_trajectories.csv")
    # init_cellspace()
    clean_and_output_data()
    generate_newsimi_test_dataset()
    traj_simi_computation("hausdorff")
    traj_simi_computation("discret_frechet")
    traj_simi_computation("edr")
    traj_simi_computation("lcss")
    traj_simi_computation("edwp")

