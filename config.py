import os
import random

import numpy
import torch


def set_seed(seed=-1):
    if seed == -1:
        return
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    debug = True
    dumpfile_uniqueid = ""
    seed = 2000
    # device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root_dir = os.path.abspath(__file__)[:-10]  # dont use os.getcwd()
    checkpoint_dir = root_dir + "/checkpoints"
    checkpoint_file = ""

    dataset = "porto"
    dataset_prefix = ""
    dataset_file = ""
    dataset_cell_file = ""
    dataset_embs_file = ""
    labeled_traj_file = ""
    neighbours_file = ""

    min_lon = 0.0
    min_lat = 0.0
    max_lon = 0.0
    max_lat = 0.0
    max_traj_len = 200
    min_traj_len = 20
    cell_res = 11
    all_res = [10, 11, 12]
    cell_size = 100.0
    cellspace_buffer = 500.0
    exec_layer = 3

    tjepa_batch_size = 64
    cell_embedding_dim = 256
    seq_embedding_dim = 256
    l2_dim = 256
    l1_dim = 256
    time_dim = 6
    moco_proj_dim = seq_embedding_dim // 2
    moco_nqueue = 2048
    moco_temperature = 0.05

    trajcl_training_epochs = 20
    trajcl_training_bad_patience = 10
    trajcl_training_lr = 0.0001
    trajcl_training_lr_degrade_gamma = 0.5
    trajcl_training_lr_degrade_step = 5
    trajcl_aug1 = "mask"
    trajcl_aug2 = "subset"
    trajcl_local_mask_sidelen = cell_res * 11

    num_level = 3

    num_neighbours = 6

    hidden_dim = 1024
    attn_layer = 1
    attn_head = 8
    pos_encoder_dropout = 0.1
    use_attention = True
    use_unet_residual = False
    attn_file = "attn_maps_nores_maxpool_norm_256_256_8155_loss_cosine"

    # trans_attention_head = 4
    # trans_attention_dropout = 0.1
    # trans_attention_layer = 3
    # trans_pos_encoder_dropout = 0.1
    # trans_hidden_dim = 2048

    traj_simp_dist = 100
    traj_shift_dist = 200
    traj_mask_ratio = 0.3
    traj_add_ratio = 0.3
    traj_subset_ratio = 0.7  # preserved ratio

    test_exp1_lcss_edr_epsilon = 0.25  # normalized

    # ===========trajsimi=============
    trajsimi_encoder_name = "TrajJEPA"
    trajsimi_encoder_mode = "finetune"
    trajsimi_measure_fn_name = "edwp"

    trajsimi_batch_size = 128
    trajsimi_epoch = 30
    trajsimi_training_bad_patience = 20
    trajsimi_learning_rate = 0.0005
    trajsimi_learning_weight_decay = 0.0001
    trajsimi_finetune_lr_rescale = 0.5

    traj_cls_epoch = 30

    long = False
    # HuMob
    #     humob_pretrain_batch_size = 8
    humob = False
    humob_ft_epoch = 30
    # humob_ft_batch_size =
    humob_learning_rate = 0.0001
    humob_learning_weight_decay = 0.0001
    humob_finetune_lr_rescale = 0.5
    humob_training_bad_patience = 10
    poi_size = 85

    # ==========JEPA_params==========
    sample_number = 4
    mask_ratio = [0.1, 0.15, 0.2, 0.25, 0.3]
    succ_prob = 0.5
    context_lb = 0.85
    context_ratio = 0.2

    @classmethod
    def update(cls, dic: dict):
        for k, v in dic.items():
            if k in cls.__dict__:
                assert type(getattr(Config, k)) == type(v)
            setattr(Config, k, v)
        cls.post_value_updates()

    @classmethod
    def post_value_updates(cls):
        if "porto" == cls.dataset:
            cls.dataset_prefix = "porto_20200_new"
            cls.min_lon = -8.7005
            cls.min_lat = 41.1001
            cls.max_lon = -8.5192
            cls.max_lat = 41.2086
            cls.train_idx = 200000
            cls.poi_size = 6
            # cls.trajcl_training_lr = 0.000005
            # cls.trajcl_training_lr = 0.0001
        elif "beijing" == cls.dataset:
            cls.dataset_prefix = "beijing/beijing"
            # cls.min_lon = 116.216997
            # cls.min_lat = 39.828540
            # cls.max_lon = 116.508135
            # cls.max_lat = 39.991288
            cls.min_lon = 116.109382
            cls.min_lat = 39.717276
            cls.max_lon = 116.734229
            cls.max_lat = 40.046068
            cls.train_idx = 70000
            cls.poi_size = 6
            cls.cell_res = 10
        elif "tokyo" == cls.dataset:
            cls.dataset_prefix = "tokyo/tokyo"
            cls.min_lon = 139.47176970135368
            cls.min_lat = 35.513059380110136
            cls.max_lon = 139.90628600120544
            cls.max_lat = 35.86602997137174
            cls.cell_res = 10
        elif "nyc" == cls.dataset:
            cls.dataset_prefix = "nyc/nyc"
            cls.min_lon = -74.27075087427112
            cls.min_lat = 40.557295
            cls.max_lon = -73.68581968909615
            cls.max_lat = 40.98764596081398
            cls.cell_res = 10
        elif "geolife" == cls.dataset:
            cls.dataset_prefix = "geolife/geolife"
            cls.min_lon = 116.109382
            cls.min_lat = 39.717276
            cls.max_lon = 116.734229
            cls.max_lat = 40.046068
            cls.train_idx = 35000
            cls.cell_res = 10
            cls.poi_size = 6
            # cls.seed = 42
        elif "humob2023" == cls.dataset:
            cls.dataset_prefix = "yjmob_2023/humob2023"
            cls.humob = True
            cls.task = 1
            cls.cell_size = 500
            cls.max_traj_len = 3600
            cls.min_traj_len = 20
            cls.time_dim = 2
            cls.tjepa_batch_size = 4
            cls.poi_size = 85
        elif "ais_au" == cls.dataset:
            cls.dataset_prefix = "ais/ais_au"
            cls.min_lon = 75.0000
            cls.min_lat = -89.8300
            cls.max_lon = 163.0000
            cls.max_lat = -2.0000
            cls.max_traj_len = 200
            cls.min_traj_len = 20
            cls.cell_res = 4
            cls.tjepa_batch_size = 8
            # cls.trajcl_training_lr = 0.00001
            cls.poi_size = 6
        elif "migration" == cls.dataset:
            cls.dataset_prefix = "migration/migration"
            cls.min_lon = 4.02517
            cls.min_lat = -2.40633
            cls.max_lon = 58.52283
            cls.max_lat = 65.2325
            cls.max_traj_len = 200
            cls.min_traj_len = 20
            cls.cell_res = 5

        cls.dataset_file = cls.root_dir + "/data/" + cls.dataset_prefix
        # cls.dataset_cell_file = cls.dataset_file + '_cell' + str(int(cls.cell_size)) + '_cellspace.pkl'
        cls.dataset_cell_file = (
            cls.dataset_file + "_cell" + str(int(cls.cell_res)) + "_cellspace.pkl"
        )
        cls.labeled_traj_file = (
            cls.root_dir + "/data/" + cls.dataset_prefix + "_labeled"
        )
        cls.index_mapping_file = (
            cls.root_dir + "/data/" + cls.dataset_prefix + "_hex_index_mapping.pkl"
        )
        cls.neighbours_file = (
            cls.root_dir + "/data/" + cls.dataset_prefix + "_neighbours.pkl"
        )
        cls.all_cells = cls.root_dir + "/data/" + "all_cells.pkl"
        cls.cell_centers = cls.root_dir + "/data/" + "center_locations.pkl"
        # cls.dataset_embs_file = cls.dataset_file + '_cell' + str(int(cls.cell_size)) + '_embdim' + str(cls.cell_embedding_dim) + '_embs.pkl'
        cls.dataset_embs_file = (
            cls.dataset_file
            + "_cell"
            + str(int(cls.cell_res))
            + "_embdim"
            + str(cls.cell_embedding_dim)
            + "_embs.pkl"
        )
        cls.poi_file = cls.dataset_file + "_poi.pkl"
        cls.poi_node_file = cls.dataset_file + "_nodes.pkl"
        cls.poi_way_file = cls.dataset_file + "_ways.pkl"
        set_seed(cls.seed)

        cls.moco_proj_dim = cls.seq_embedding_dim // 2

    @classmethod
    def to_str(cls):  # __str__, self
        dic = cls.__dict__.copy()
        lst = list(
            filter(
                lambda p: (not p[0].startswith("__")) and type(p[1]) != classmethod,
                dic.items(),
            )
        )
        return "\n".join([str(k) + " = " + str(v) for k, v in lst])
