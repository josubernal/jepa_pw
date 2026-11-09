import sys

import numpy as np

sys.path.append('..')

import os
import math
import time
import logging
import pickle
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
from functools import partial
from model.jepa import JEPA_base
import h3
from config import Config

from utils.data_loader import read_traj_dataset
from utils.traj import *
from utils import tool_funcs
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib.pyplot as plt

torch.cuda.empty_cache()


def _collate_hex(trajs, embs, poi, adj):
    trajs, timestamps = zip(*trajs)
    traj_emb_cell = [embs[list(t)].clone().detach() for t in trajs]

    traj_emb_cell = pad_sequence(traj_emb_cell, batch_first=True).to(Config.device)

    if adj is not None:
        neighbor = [adj[t] for t in trajs]
        # print(neighbor)
        neighbor_seq = [torch.stack([embs[list(p)].clone().detach() for p in n]) for n in neighbor]
        neighbor_seq = pad_sequence(neighbor_seq, batch_first=True).to(Config.device)
    else:
        neighbor_seq = None

    if poi is not None:
        poi_seq = [retrieve_poi(t, poi) for t in trajs]
        poi_seq = [torch.tensor(t) for t in poi_seq]
        poi_seq = pad_sequence(poi_seq, batch_first=True).to(Config.device)
        poi_seq = poi_seq.to(torch.float32)
    else:
        poi_seq = None

    num_points = torch.tensor(list(map(len, trajs)), dtype=torch.long, device=Config.device)

    return traj_emb_cell, num_points, None, neighbor_seq, poi_seq, None

def _collate_test(trajs, embs, poi, adj):
    traj_emb_cell = [embs[list(t)].clone().detach() for t in trajs]

    traj_emb_cell = pad_sequence(traj_emb_cell, batch_first=True).to(Config.device)

    if adj is not None:
        neighbor = [adj[t] for t in trajs]
        # print(neighbor)
        neighbor_seq = [torch.stack([embs[list(p)].clone().detach() for p in n]) for n in neighbor]
        neighbor_seq = pad_sequence(neighbor_seq, batch_first=True).to(Config.device)
    else:
        neighbor_seq = None


    if poi is not None:
        poi_seq = [retrieve_poi(t, poi) for t in trajs]
        poi_seq = [torch.tensor(t) for t in poi_seq]
        poi_seq = pad_sequence(poi_seq, batch_first=True).to(Config.device)
        poi_seq = poi_seq.to(torch.float32)
    else:
        poi_seq = None

    num_points = torch.tensor(list(map(len, trajs)), dtype=torch.long, device=Config.device)

    return traj_emb_cell, num_points, None, neighbor_seq, poi_seq, None


# def get_neighors(traj, nb, embs):
#     """
#     also need to add neighbor embeddings.
#     """
#     return [embs[nb[p]] for p in traj]



class TrajJEPATrainer:

    def __init__(self):
        super(TrajJEPATrainer, self).__init__()
        self.embs = pickle.load(open(Config.dataset_embs_file, 'rb'))  # tensor

        if os.path.exists(Config.poi_file):
            self.poi = pickle.load(open(Config.poi_file, 'rb'))
        else:
            self.poi = None

        self.index_mapping = pickle.load(open(Config.index_mapping_file, 'rb'))


        train_dataset, _, _ = read_traj_dataset(Config.dataset_file)
        self.num_train = len(train_dataset)

        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=Config.tjepa_batch_size,
                                           shuffle=False,
                                           num_workers=0,
                                           drop_last=True,
                                           collate_fn=partial(_collate_hex, embs=self.embs, poi=None, adj=None))      # don't include POI first....

        self.model = JEPA_base().to(Config.device)
        self.checkpoint_file = '{}/{}_{}{}.pt'.format(Config.checkpoint_dir,
                                                            Config.dataset_prefix,
                                                            Config.checkpoint_file,
                                                            Config.dumpfile_uniqueid)

        logging.info(f"Model will be saved to {self.checkpoint_file}")
        self.ema = (0.996, 1.0)
        self.ipe_scale = 1.0
    def train(self):
        training_starttime = time.time()
        training_gpu_usage = training_ram_usage = 0.0
        logging.info("[Training] START! timestamp={:.0f}".format(training_starttime))
        torch.autograd.set_detect_anomaly(True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.trajcl_training_lr, weight_decay=0.0001)

        layer = int(Config.exec_layer)

        scheduler = CosineAnnealingLR(optimizer, T_max=Config.trajcl_training_epochs, eta_min=1e-6)

        training_steps = self.num_train / Config.tjepa_batch_size
        total_iter = training_steps * Config.trajcl_training_epochs
        momentum_scheduler = (self.ema[0] + i * (self.ema[1] - self.ema[0]) / (
                    training_steps * Config.trajcl_training_epochs * self.ipe_scale)
                              for i in range(int(training_steps * Config.trajcl_training_epochs * self.ipe_scale) + 1))

        best_loss_train = 100000
        best_epoch = 0
        bad_counter = 0
        bad_patience = Config.trajcl_training_bad_patience
        

        loss_1, loss_2, loss_3 = [], [], []
        context_encoders_all = [
            self.model.context_encoder_1,
            self.model.context_encoder_2,
            self.model.context_encoder_3
        ]
        target_encoders_all = [
            self.model.target_encoder_1,
            self.model.target_encoder_2,
            self.model.target_encoder_3
        ]
        gamma_history = [[] for _ in range(len(context_encoders_all))]

        for i_ep in range(Config.trajcl_training_epochs):

            if i_ep == 0 or i_ep % 5 == 0:
                self.eval(layer)


            _time_ep = time.time()
            loss_ep = []
            train_gpu = []
            train_ram = []

            self.model.train()

            temp_loss1, temp_loss2, temp_loss3 = [], [], []

            _time_batch_start = time.time()
            for i_batch, batch in enumerate(self.train_dataloader):

                _time_batch = time.time()
                optimizer.zero_grad()

                cell_emb, num_points, timestamps, adj, poi_seq, motion = batch
                loss, losses = self.model(cell_emb, timestamps, num_points, adj, poi_seq, motion, exec_layer=layer)
                loss.backward()

                if len(losses) > 0:
                    temp_loss1.append(losses[0].detach().cpu().item())
                if len(losses) > 1:
                    temp_loss2.append(losses[1].detach().cpu().item())
                if len(losses) > 2:
                    temp_loss3.append(losses[2].detach().cpu().item())

                clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                loss_ep.append(loss.item())
                # EMA.
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for ctx, tgt in zip(context_encoders_all, target_encoders_all):
                        for p_ctx, p_tgt in zip(ctx.parameters(), tgt.parameters()):
                            p_tgt.data.mul_(m).add_((1. - m) * p_ctx.detach().data)
                train_gpu.append(tool_funcs.GPUInfo.mem()[0])
                train_ram.append(tool_funcs.RAMInfo.mem())

                if i_batch % 100 == 0 and i_batch:
                    logging.debug("[Training] ep-batch={}-{}, loss={:.3f}, @={:.3f}, gpu={}, ram={}" \
                                  .format(i_ep, i_batch, loss.item(), time.time() - _time_batch_start,
                                          tool_funcs.GPUInfo.mem(), tool_funcs.RAMInfo.mem()))

            scheduler.step()  # decay before optimizer when pytorch < 1.1

            temp_loss1 = torch.tensor(temp_loss1, dtype=torch.float32, device="cpu").mean().item()
            temp_loss2 = torch.tensor(temp_loss2, dtype=torch.float32, device="cpu").mean().item()
            temp_loss3 = torch.tensor(temp_loss3, dtype=torch.float32, device="cpu").mean().item()
            logging.info(f"loss 1: {temp_loss1}, loss 2: {temp_loss2}, loss 3: {temp_loss3}.")
            logging.info(f"loss {loss}.")
            loss_1.append(temp_loss1)
            loss_2.append(temp_loss2)
            loss_3.append(temp_loss3)

            with torch.no_grad():
                for layer_idx, encoder in enumerate(context_encoders_all):
                    if encoder is None or layer_idx >= layer:
                        continue
                    layer_modules = getattr(encoder.encoder, 'layers', None)
                    if layer_modules is None:
                        continue
                    collected = []
                    for block in layer_modules:
                        gamma_param = getattr(block, 'gamma_param', None)
                        if gamma_param is None:
                            continue
                        gamma_vals = torch.sigmoid(gamma_param.detach())
                        collected.append(gamma_vals.mean().item())
                    if collected:
                        gamma_history[layer_idx].append(float(np.mean(collected)))
                    else:
                        gamma_history[layer_idx].append(float('nan'))

            loss_ep_avg = tool_funcs.mean(loss_ep)
            logging.info("[Training] ep={}: avg_loss={:.3f}, @={:.3f}/{:.3f}, gpu={}, ram={}" \
                         .format(i_ep, loss_ep_avg, time.time() - _time_ep, time.time() - training_starttime,
                                 tool_funcs.GPUInfo.mem(), tool_funcs.RAMInfo.mem()))

            training_gpu_usage = tool_funcs.mean(train_gpu)
            training_ram_usage = tool_funcs.mean(train_ram)

            # early stopping
            # self.save_checkpoint()
            if loss_ep_avg < best_loss_train:
                best_epoch = i_ep
                best_loss_train = loss_ep_avg
                bad_counter = 0
                self.save_checkpoint()
            else:
                bad_counter += 1

            if bad_counter == bad_patience or (i_ep + 1) == Config.trajcl_training_epochs:
                logging.info("[Training] END! @={}, best_epoch={}, best_loss_train={:.6f}" \
                             .format(time.time() - training_starttime, best_epoch, best_loss_train))
                break
        #
        self.test(layer)
        epo = list(range(1, len(loss_1) + 1))

        plt.figure(figsize=(8, 5))
        plt.plot(epo, loss_1, label="Loss Component 1", color="blue", marker="o")
        plt.plot(epo, loss_2, label="Loss Component 2", color="red", marker="s")
        plt.plot(epo, loss_3, label="Loss Component 3", color="green", marker="^")

        # Labels and Title
        plt.xlabel("Epochs")
        plt.ylabel("Loss Value")
        plt.title("Training Loss for Each Component Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("{}_losses.png".format(Config.checkpoint_file))
        plt.close()

        max_gamma_epochs = max((len(series) for series in gamma_history), default=0)
        if max_gamma_epochs > 0:
            plt.figure(figsize=(8, 5))
            for layer_idx, series in enumerate(gamma_history):
                if not series or layer_idx >= layer:
                    continue
                epochs_axis = list(range(1, len(series) + 1))
                plt.plot(epochs_axis, series, marker="o", label=f"Layer {layer_idx + 1}")
            plt.xlabel("Epochs")
            plt.ylabel("Mean γ (sigmoid)")
            plt.title("Layer-wise γ Averages Over Training")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("{}_gamma_mean.png".format(Config.checkpoint_file))
            plt.close()


        return {'enc_train_time': time.time() - training_starttime, \
                'enc_train_gpu': training_gpu_usage, \
                'enc_train_ram': training_ram_usage}



    @torch.no_grad()
    def eval(self, layer):
        # 1. read best model
        # 2. read trajs from file, then -> embeddings
        # 3. run testing
        # n. varying db size, downsampling rates, and distort rates

        logging.info('[Eval]start.')
        self.model.eval()

        # varying db size
        with open(Config.dataset_file + '_newsimi_raw.pkl', 'rb') as fh:
            q_lst, db_lst = pickle.load(fh)

            querys, databases = self.test_merc_seq_to_embs(q_lst, db_lst, layer)
            dists = torch.cdist(querys, databases, p=1)  # [1000, 100000]
            targets = torch.diag(dists)  # [1000]
            results = []
            if Config.dataset == "porto":
                db_1, db_5, skip = 20000, 100001, 20000
            elif  Config.dataset == "beijing" or Config.dataset == "geolife":
                db_1, db_5, skip = 2000, 10001, 2000
            elif Config.dataset == "tokyo":
                db_1, db_5, skip = 600, 3001, 600
            elif Config.dataset == "nyc":
                db_1, db_5, skip = 140, 701, 140
            elif Config.dataset == "ais_au":
                db_1, db_5, skip = 1400, 7001, 1400
            else:
                pass
            for n_db in range(db_1, db_5, skip):      # for porto
                rank = torch.sum(torch.le(dists[:, 0:n_db].T, targets)).item() / len(q_lst)
                results.append(rank)
            logging.info(
                '[EXPFlag]task=newsimi,encoder=TJEPA,varying=dbsize,r1={:.3f},r2={:.3f},r3={:.3f},r4={:.3f},r5={:.3f}' \
                .format(*results))

        # varying downsampling; varying distort
        for vt in ['downsampling', 'distort']:
            results = []
            for rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
                with open(Config.dataset_file + '_newsimi_' + vt + '_' + str(rate) + '.pkl', 'rb') as fh:
                    q_lst, db_lst = pickle.load(fh)
                    # print(q_lst)
                    querys, databases = self.test_merc_seq_to_embs(q_lst, db_lst, layer)
                    dists = torch.cdist(querys, databases, p=1)  # [1000, 100000]
                    targets = torch.diag(dists)  # [1000]
                    result = torch.sum(torch.le(dists.T, targets)).item() / len(q_lst)
                    results.append(result)
            logging.info(
                '[EXPFlag]task=newsimi,encoder=TJEPA,varying={},r1={:.3f},r2={:.3f},r3={:.3f},r4={:.3f},r5={:.3f}' \
                .format(vt, *results))
        return

    @torch.no_grad()
    def test(self, layer):
        # 1. read best model
        # 2. read trajs from file, then -> embeddings
        # 3. run testing
        # n. varying db size, downsampling rates, and distort rates
        import time
        logging.info('[Test]start.')
        s = time.time()
        if Config.dataset == "tokyo" or Config.dataset == "nyc" or Config.dataset == "ais_au" or Config.dataset == "migration":
            logging.info("1")
            self.model.load_checkpoint()
        else:
            self.load_checkpoint()
        self.model.eval()

        # varying db size
        with open(Config.dataset_file + '_newsimi_raw.pkl', 'rb') as fh:
            
            q_lst, db_lst = pickle.load(fh)
            querys, databases = self.test_merc_seq_to_embs(q_lst, db_lst, layer)
            dists = torch.cdist(querys, databases, p=1)  # [1000, 100000]
            targets = torch.diag(dists)  # [1000]
            results = []
            if Config.dataset == "porto":
                db_1, db_5, skip = 20000, 100001, 20000
            elif  Config.dataset == "beijing" or Config.dataset == "geolife":
                db_1, db_5, skip = 2000, 10001, 2000
            elif Config.dataset == "tokyo":
                db_1, db_5, skip = 600, 3001, 600
            elif Config.dataset == "nyc":
                db_1, db_5, skip = 140, 701, 140
            elif Config.dataset == "ais_au":
                db_1, db_5, skip = 1400, 7001, 1400
            elif Config.dataset == "migration":
                db_1, db_5, skip = 160, 801, 160
            else:
                pass
            for n_db in range(db_1, db_5, skip):     # for porto
                rank = torch.sum(torch.le(dists[:, 0:n_db].T, targets)).item() / len(q_lst)
                results.append(rank)
            logging.info(
                '[EXPFlag]task=newsimi,encoder=TJEPA,varying=dbsize,r1={:.3f},r2={:.3f},r3={:.3f},r4={:.3f},r5={:.3f}' \
                .format(*results))

        # varying downsampling; varying distort
        for vt in ['downsampling', 'distort']:
            results = []
            for rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
            # for rate in [0.2, 0.4, 0.6, 0.8, 1.0]:
                with open(Config.dataset_file + '_newsimi_' + vt + '_' + str(rate) + '.pkl', 'rb') as fh:
                    q_lst, db_lst = pickle.load(fh)
                    querys, databases = self.test_merc_seq_to_embs(q_lst, db_lst, layer)
                    dists = torch.cdist(querys, databases, p=1)  # [1000, 100000]
                    targets = torch.diag(dists)  # [1000]
                    result = torch.sum(torch.le(dists.T, targets)).item() / len(q_lst)
                    results.append(result)
            logging.info(
                '[EXPFlag]task=newsimi,encoder=TJEPA,varying={},r1={:.3f},r2={:.3f},r3={:.3f},r4={:.3f},r5={:.3f}' \
                .format(vt, *results))
        
        e = time.time()

        print(s-e)

        return


    @torch.no_grad()
    def test_merc_seq_to_embs(self, q_lst, db_lst, layer):
        querys = []
        databases = []
        num_query = len(q_lst)  # 1000
        num_database = len(db_lst)  # 100000
        batch_size = num_query
        # batch_size = 1
        q_lst = [[self.index_mapping[str(t)] for t in q]for q in q_lst]
        db_lst = [[self.index_mapping[str(t)] for t in d]for d in db_lst]

        # from vis_mlp import MLPRegressor

        for i in range(num_database // batch_size):
            if i == 0:

                traj_emb1_cell, num_points1, timestamps1, adj1, poi_seq1, motion1 \
                    = _collate_test(q_lst, self.embs, None, None)

                trajs1_emb = self.model.interpret(traj_emb1_cell, None, num_points1, poi_seq1, adj1, motion1, exec_layer=layer)


                trajs1_emb = trajs1_emb.permute(1, 0, 2)
                trajs1_emb = torch.sum(trajs1_emb, 0)
                trajs1_emb = trajs1_emb / num_points1.unsqueeze(-1).expand(trajs1_emb.shape)
                querys.append(trajs1_emb)

            traj_emb2_cell, num_points2, timestamps2, adj2, poi_seq2, motion2  \
                = _collate_test(db_lst[i * batch_size: (i + 1) * batch_size], self.embs, None, None)

            trajs2_emb = self.model.interpret(traj_emb2_cell, None, num_points2, poi_seq2, adj2, motion2, exec_layer=layer)
            trajs2_emb = trajs2_emb.permute(1, 0, 2)
            trajs2_emb = torch.sum(trajs2_emb, 0)
            trajs2_emb = trajs2_emb / num_points2.unsqueeze(-1).expand(trajs2_emb.shape)
            databases.append(trajs2_emb)

        querys = torch.cat(querys)  # tensor; traj embeddings
        databases = torch.cat(databases)
        return querys, databases

  
    def dump_embeddings(self):
        return

    def save_checkpoint(self):
        # Ensure the directory for the checkpoint file exists before saving.
        checkpoint_dir = os.path.dirname(self.checkpoint_file)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        torch.save({'model_state_dict': self.model.state_dict()},
                   self.checkpoint_file)
        return

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(Config.device)

        return



