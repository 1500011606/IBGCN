import sys, json, argparse, random, re, os, shutil

sys.path.append("src/")
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import math
import os.path as osp
import networkx as nx
import pdb

# from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch import optim
import torch.multiprocessing as mp
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.utils import to_dense_batch, k_hop_subgraph

from utils import common_tools as ct
from utils.my_math import masked_mae_np, masked_mape_np, masked_mse_np
from utils.data_convert import generate_samples
from src.model.model import Basic_Model, Integrated_Model
# from src.model.ewc import EWC
from src.trafficDataset import TrafficDataset
from src.model.MINE import compute_mutual_information, compute_mi

# from src.model import detect
# from src.model import replay

result = {3: {"mae": {}, "mape": {}, "rmse": {}}, 6: {"mae": {}, "mape": {}, "rmse": {}},
          12: {"mae": {}, "mape": {}, "rmse": {}}}
pin_memory = True
n_work = 0


def update(src, tmp):
    for key in tmp:
        if key != "gpuid":
            src[key] = tmp[key]


def load_best_model(args):
    # 获取loss最低的模型并加载 (保存的文件以loss值命名,故对文件名做排序)
    loss = []
    for filename in os.listdir(osp.join(args.model_path, args.logname)):
        loss.append(filename[0:-4])
    loss = sorted(loss)
    load_path = osp.join(args.model_path, args.logname, loss[0] + ".pkl")

    args.logger.info("[*] load from {}".format(load_path))
    state_dict = torch.load(load_path, map_location=args.device)["model_state_dict"]
    model = Basic_Model(args)
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    return model, loss[0]


def init(args):
    # 读取conf文件,初始化
    conf_path = osp.join(args.conf)
    info = ct.load_json_file(conf_path)
    info["time"] = datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
    update(vars(args), info)
    vars(args)["path"] = osp.join(args.model_path, args.logname + args.time)
    ct.mkdirs(args.path)
    del info


def init_log(args):
    # 初始化log
    log_dir, log_filename = args.path, args.logname
    logger = logging.getLogger(__name__)
    ct.mkdirs(log_dir)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(osp.join(log_dir, log_filename + ".log"))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info("logger name:%s", osp.join(log_dir, log_filename + ".log"))
    vars(args)["logger"] = logger
    return logger


def seed_set(seed=0):
    # 设置随机数种子
    max_seed = (1 << 32) - 1
    random.seed(seed)
    np.random.seed(random.randint(0, max_seed))
    torch.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed(random.randint(0, max_seed))
    torch.cuda.manual_seed_all(random.randint(0, max_seed))
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True


def train(inputs, args):
    # Model Setting
    global result
    path = args.path
    ct.mkdirs(path)

    if args.loss == "mse":
        lossfunc = func.mse_loss
    elif args.loss == "huber":
        lossfunc = func.smooth_l1_loss

    # Dataset Definition

    train_loader = DataLoader(TrafficDataset(inputs, "train"), batch_size=args.batch_size, shuffle=True,
                              pin_memory=pin_memory, num_workers=n_work)
    val_loader = DataLoader(TrafficDataset(inputs, "val"), batch_size=args.batch_size, shuffle=False,
                            pin_memory=pin_memory, num_workers=n_work)
    vars(args)["sub_adj"] = vars(args)["adj"]
    test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False,
                             pin_memory=pin_memory, num_workers=n_work)

    args.logger.info("[*] Dataset load!")

    # Model Definition
    if args.init == True:
        gnn_model, _ = load_best_model(args)
    else:
        gnn_model = Integrated_Model(args).to(args.device)
    model = gnn_model

    # Model Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    args.logger.info("[*] Training start")
    global_train_steps = len(train_loader) // args.batch_size + 1

    iters = len(train_loader)
    lowest_validation_loss = 1e7
    counter = 0
    patience = 5
    model.train()
    use_time = []
    for epoch in range(args.epoch):
        training_loss = 0.0
        start_time = datetime.now()
        # Train Model
        cn = 0
        for batch_idx, data in enumerate(train_loader):
            if epoch == 0 and batch_idx == 0:
                args.logger.info("node number {}".format(data.x.shape))
            data = data.to(device, non_blocking=pin_memory)

            pred, high_flow, eigen_flow = model(data, args.sub_adj)

            # loss = lossfunc(data.y, pred, reduction="mean")
            mine_net_high = compute_mutual_information(data.x, high_flow.detach(), args.device)
            mine_net_eigen = compute_mutual_information(data.x, eigen_flow.detach(), args.device)
            loss = lossfunc(data.y, pred, reduction="mean") \
                   + compute_mi(mine_net_high, data.x, high_flow) \
                   - compute_mi(mine_net_eigen, data.x, eigen_flow)

            optimizer.zero_grad()
            training_loss += float(loss)
            loss.backward()
            optimizer.step()

            cn += 1

        if epoch == 0:
            total_time = (datetime.now() - start_time).total_seconds()
        else:
            total_time += (datetime.now() - start_time).total_seconds()
        use_time.append((datetime.now() - start_time).total_seconds())
        training_loss = training_loss / cn

        # Validate Model
        validation_loss = 0.0
        cn = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                data = data.to(device, non_blocking=pin_memory)
                pred, _, _ = model(data, args.sub_adj)
                loss = masked_mae_np(data.y.cpu().data.numpy(), pred.cpu().data.numpy(), 0)
                validation_loss += float(loss)
                cn += 1
        validation_loss = float(validation_loss / cn)

        args.logger.info(f"epoch:{epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f}")

        # Early Stop
        if validation_loss <= lowest_validation_loss:
            counter = 0
            lowest_validation_loss = round(validation_loss, 4)
            torch.save({'model_state_dict': gnn_model.state_dict()},
                       osp.join(path, str(round(validation_loss, 4)) + ".pkl"))
        else:
            counter += 1
            if counter > patience:
                break

    best_model_path = osp.join(path, str(lowest_validation_loss) + ".pkl")
    best_model = Integrated_Model(args)
    best_model.load_state_dict(torch.load(best_model_path, args.device)["model_state_dict"])
    best_model = best_model.to(args.device)

    # Test Model
    test_model(best_model, args, test_loader, pin_memory)
    result[args.year] = {"total_time": total_time, "average_time": sum(use_time) / len(use_time),
                         "epoch_num": epoch + 1}
    args.logger.info("Finished optimization, total time:{:.2f} s, best model:{}".format(total_time, best_model_path))


def test_model(model, args, testset, pin_memory):
    model.eval()
    pred_ = []
    truth_ = []
    loss = 0.0
    with torch.no_grad():
        cn = 0
        for data in testset:
            data = data.to(args.device, non_blocking=pin_memory)
            pred, _, _ = model(data, args.adj)
            loss += func.mse_loss(data.y, pred, reduction="mean")
            pred, _ = to_dense_batch(pred, batch=data.batch)
            data.y, _ = to_dense_batch(data.y, batch=data.batch)
            pred_.append(pred.cpu().data.numpy())
            truth_.append(data.y.cpu().data.numpy())
            cn += 1
        loss = loss / cn
        args.logger.info("[*] loss:{:.4f}".format(loss))
        pred_ = np.concatenate(pred_, 0)
        truth_ = np.concatenate(truth_, 0)
        mae = metric(truth_, pred_, args)
        return loss


def metric(ground_truth, prediction, args):
    global result
    pred_time = [3, 6, 12]
    args.logger.info("[*] year {}, testing".format(args.year))
    for i in pred_time:
        mae = masked_mae_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
        rmse = masked_mse_np(ground_truth[:, :, :i], prediction[:, :, :i], 0) ** 0.5
        mape = masked_mape_np(ground_truth[:, :, :i], prediction[:, :, :i], 0)
        args.logger.info("T:{:d}\tMAE\t{:.4f}\tRMSE\t{:.4f}\tMAPE\t{:.4f}".format(i, mae, rmse, mape))
        result[i]["mae"][args.year] = mae
        result[i]["mape"][args.year] = mape
        result[i]["rmse"][args.year] = rmse
    return mae


def main(args):
    year = 2012
    logger = init_log(args)
    logger.info("params : %s", vars(args))
    ct.mkdirs(args.save_data_path)

    #graph = nx.from_numpy_matrix(np.load(osp.join(args.graph_path, str(year)+"_adj.npz"))["x"])
    graph = nx.from_numpy_matrix(np.load(osp.join(args.graph_path)))
    vars(args)["graph_size"] = graph.number_of_nodes()
    vars(args)["year"] = year
    # 如果data_process为真,则预处理数据,否则直接加载
    # inputs = generate_samples(31, osp.join(args.save_data_path, str(year)+'_30day'), np.load(osp.join(args.raw_data_path, str(year)+".npz"))["x"], graph, val_test_mix=True) \
    inputs = generate_samples(10, args.save_data_path, np.load(args.raw_data_path)["data"], graph,
                              val_test_mix=True) \
        if args.data_process else np.load(osp.join(args.save_data_path), allow_pickle=True)

    # args.logger.info("[*] Year {} load from {}_30day.npz".format(args.year, osp.join(args.save_data_path, str(year))))
    # 加载邻接矩阵
    adj = np.load(osp.join(args.graph_path))
    adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
    vars(args)["adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)

    # 训练还是测试
    if args.train:
        train(inputs, args)
    else:
        model, _ = load_best_model(args)
        test_loader = DataLoader(TrafficDataset(inputs, "test"), batch_size=args.batch_size, shuffle=False,
                                 pin_memory=pin_memory, num_workers=n_work)
        test_model(model, args, test_loader, pin_memory=True)

    # 3,6,12 表示对时间长度为15/30/60min的序列的预测,指标为 mae,mape,rmse
    for i in [3, 6, 12]:
        info = ""
        for j in ['mae', 'rmse', 'mape']:
            info += "{:.2f}\t".format(result[i][j][year])
            logger.info("{}\t{}\t".format(i, j) + info)

    info = "year\t{}\ttotal_time\t{}\taverage_time\t{}\tepoch\t{}".format(year, result[year]["total_time"],
                                                                          result[year]["average_time"],
                                                                          result[year]['epoch_num'])
    logger.info(info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--conf", type=str, default="conf/test.json")
    parser.add_argument("--paral", type=int, default=0)
    parser.add_argument("--gpuid", type=int, default=0)
    parser.add_argument("--logname", type=str, default="info")
    parser.add_argument("--load_first_year", type=int, default=0,
                        help="0: training first year, 1: load from model path of first year")
    parser.add_argument("--first_year_model_path", type=str,
                        default="res/district3F11T17/TrafficStream2021-05-09-11:56:33.516033/2011/27.4437.pkl",
                        help='specify a pretrained model root')
    args = parser.parse_args()
    init(args)
    seed_set(13)

    device = torch.device("cuda:{}".format(args.gpuid)) if torch.cuda.is_available() and args.gpuid != -1 else "cpu"
    vars(args)["device"] = device
    main(args)
