from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.metrics import f1_score, recall_score, precision_score
import utils
from sklearn.model_selection import KFold, train_test_split
from prettytable import PrettyTable
from model import HGANDDS
import torch.nn.functional as F
from dgllife.utils import EarlyStopping
from transformers import AdamW
from dataset import DrugSynergyDataset
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import random
import os
import argparse
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss


def read_data_file(data_file):
    smiles_1 = []
    smiles_2 = []
    Y = []
    cell_line = []

    with open(data_file, 'r') as f:
        all_lines = f.readlines()

        for line in all_lines[1:]:
            row = line.rstrip().split(',')[1:]
            # print(row)
            smiles_1.append(row[0])
            smiles_2.append(row[1])
            cell_line.append((row[2]))
            Y.append(int(row[3]))

    return smiles_2, smiles_1, cell_line, Y


def define_dataloader(train_index, test_index, smiles_1, smiles_2, cell_line, Y, maxCompoundLen, batch_size, device, dataset_type):
    test_index, valid_index = train_test_split(test_index, test_size=0.5, random_state=2023)
    train_drug_2_cv = np.array(smiles_2)[train_index]
    train_drug_1_cv = np.array(smiles_1)[train_index]
    train_cell_line_cv = np.array(cell_line)[train_index]
    train_Y_cv = np.array(Y)[train_index]

    test_drug_2_cv = np.array(smiles_2)[test_index]
    test_drug_1_cv = np.array(smiles_1)[test_index]
    test_cell_line_cv = np.array(cell_line)[test_index]
    test_Y_cv = np.array(Y)[test_index]

    valid_drug_2_cv = np.array(smiles_2)[valid_index]
    valid_drug_1_cv = np.array(smiles_1)[valid_index]
    valid_cell_line_cv = np.array(cell_line)[valid_index]
    valid_Y_cv = np.array(Y)[valid_index]

    train_dataset = DrugSynergyDataset(train_drug_1_cv,
                                       train_drug_2_cv,
                                       train_Y_cv,
                                       train_cell_line_cv, device, maxCompoundLen, dataset_type=dataset_type)

    valid_dataset = DrugSynergyDataset(valid_drug_1_cv,
                                       valid_drug_2_cv,
                                       valid_Y_cv,
                                       valid_cell_line_cv, device, maxCompoundLen, dataset_type=dataset_type)

    test_dataset = DrugSynergyDataset(test_drug_1_cv,
                                      test_drug_2_cv,
                                      test_Y_cv,
                                      test_cell_line_cv, device, maxCompoundLen, dataset_type=dataset_type)

    trainLoader = DataLoader(train_dataset,
                             batch_size=batch_size, shuffle=True)

    validLoader = DataLoader(valid_dataset,
                             batch_size=batch_size, shuffle=True)

    testLoader = DataLoader(test_dataset,
                            batch_size=batch_size, shuffle=True)

    return trainLoader, validLoader, testLoader


def validate_new(valid_loader, model, comb_data):
    model.eval()
    preds = torch.Tensor()
    trues = torch.Tensor()
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            compounds_1, compounds_2, synergyScores, cell_line, fp1, fp2, cid_nodeid_list1, cid_nodeid_list2, cell_node_list = batch
            compounds_1, compounds_2, synergyScores, cell_line, fp1, fp2, cid_nodeid_list1, cid_nodeid_list2, cell_node_list = compounds_1.to(device), compounds_2.to(
                device), synergyScores.to(device), cell_line.to(device), fp1.to(device), fp2.to(device), cid_nodeid_list1.to(device), cid_nodeid_list2.to(device), cell_node_list.to(device)
            pre_synergy = model(comb_data, cid_nodeid_list1, cid_nodeid_list2, cell_node_list)
            pre_synergy = torch.nn.functional.softmax(pre_synergy)[:, 1]
            preds = torch.cat((preds, pre_synergy.cpu()), 0)
            trues = torch.cat((trues, synergyScores.view(-1, 1).cpu()), 0)
        y_pred = np.array(preds) > 0.5
        roc_auc = roc_auc_score(trues, preds)
        acc = accuracy_score(trues, y_pred)
        f1 = f1_score(trues, y_pred, average='binary')
        prec = precision_score(trues, y_pred, average='binary')
        rec = recall_score(trues, y_pred, average='binary')
        aupr = average_precision_score(trues, preds)
        return acc, prec, rec, f1, roc_auc, aupr


def train(train_loader, model, epoch, optimizer, device, scheduler, comb_data, print_freq=50):
    model.train()
    cross_entropy_loss = nn.CrossEntropyLoss()
    losses = AverageMeter()
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        compounds_1, compounds_2, synergyScores, cell_line, fp1, fp2, cid_nodeid_list1, cid_nodeid_list2, cell_node_list = batch
        compounds_1, compounds_2, synergyScores, cell_line, fp1, fp2, cid_nodeid_list1, cid_nodeid_list2, cell_node_list = compounds_1.to(device), compounds_2.to(
            device), synergyScores.to(device), cell_line.to(device), fp1.to(device), fp2.to(device), cid_nodeid_list1.to(device), cid_nodeid_list2.to(device), cell_node_list.to(device)
        pre_synergy = model(comb_data, cid_nodeid_list1, cid_nodeid_list2, cell_node_list)
        pre_synergy2 = model(comb_data, cid_nodeid_list1, cid_nodeid_list2, cell_node_list)
        ce_loss = 0.5 * (cross_entropy_loss(pre_synergy, synergyScores.squeeze(1)) +
                         cross_entropy_loss(pre_synergy2, synergyScores.squeeze(1)))
        kl_loss = compute_kl_loss(pre_synergy, pre_synergy2)
        α = 5
        loss = ce_loss + α * kl_loss
        losses.update(loss.item(), len(compounds_1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % print_freq == 0:
            log_str = 'TRAIN -> Epoch{epoch}: \tIter:{iter}\t Loss:{loss.val:.5f} ({loss.avg:.5f})'.format(
                epoch=epoch, iter=i, loss=losses)
            print(log_str)


def seed_torch(
    seed: int = 42
):
    """_summary_

    Args:
        seed (int, optional): _description_. Defaults to 42.
    """
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def run_expriments(device, comb_data, data_set_path, log_filename, args):
    split = args.split
    seed_torch(2023)
    all_acc = np.zeros((split, 1))
    all_prec = np.zeros((split, 1))
    all_rec = np.zeros((split, 1))
    all_f1 = np.zeros((split, 1))
    all_roc_auc = np.zeros((split, 1))
    all_aupr = np.zeros((split, 1))
    n_epochs = args.n_epochs
    smiles_1, smiles_2, cell_line, Y = read_data_file(data_set_path)
    kf = KFold(n_splits=split, shuffle=True, random_state=2023)
    for split, (train_index, test_index) in enumerate(kf.split(Y)):
        trainLoader, validLoader, testLoader = define_dataloader(train_index, test_index, smiles_1, smiles_2, cell_line, Y,
                                                                 128, args.batch_size, device, data_type)
        model = HGANDDS(data=comb_data, hidden_channels=args.hidden_channels, is_gnn=False,
                        drug_feature_length=args.drug_feature_length, data_type=data_type)
        filename = utils.add_time_suffix('hgandds_model'+data_type)
        stopper = EarlyStopping(mode='higher', filename=filename, patience=10)
        model = model.to(device)
        comb_data = comb_data.to(device)
        lr = args.lr
        optimizer = AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=n_epochs,
                                                        steps_per_epoch=len(trainLoader))
        for epochind in range(n_epochs):
            train(trainLoader, model, epochind, optimizer, device, scheduler, comb_data)
            acc, prec, rec, f1, roc_auc, aupr = validate_new(validLoader, model, comb_data)
            e_tables = PrettyTable(['epoch', 'acc', 'pre', 'rec', 'f1', 'auc', 'aupr'])
            e_tables.float_format = '.3'
            row = [epochind, acc, prec, rec, f1, roc_auc, aupr]
            e_tables.add_row(row)
            utils.log_to_file_and_console(log_file_name=log_filename, fmt='', log=e_tables)
            early_stop = stopper.step(roc_auc, model)
            if early_stop:
                break
        stopper.load_checkpoint(model)
        acc, prec, rec, f1, roc_auc, aupr = validate_new(testLoader, model, comb_data)
        e_tables = PrettyTable(['test', 'acc', 'pre', 'rec', 'f1', 'auc', 'aupr'])
        e_tables.float_format = '.3'
        row = [epochind, acc, prec, rec, f1, roc_auc, aupr]
        e_tables.add_row(row)
        utils.log_to_file_and_console(log_file_name=log_filename, fmt='', log=e_tables)
        all_acc[split] = acc
        all_prec[split] = prec
        all_rec[split] = rec
        all_f1[split] = f1
        all_roc_auc[split] = roc_auc
        all_aupr[split] = aupr
    utils.log_to_file_and_console(log_file_name=log_filename, fmt='accuracy:  {0:6f}({1:6f})'.format(
        np.mean(all_acc),  np.std(all_acc)), log=None)
    utils.log_to_file_and_console(log_file_name=log_filename, fmt='f1:  {0:6f}({1:6f})'.format(
        np.mean(all_f1), np.std(all_f1)), log=None)
    utils.log_to_file_and_console(log_file_name=log_filename, fmt='roc_auc:  {0:6f}({1:6f})'.format(
        np.mean(all_roc_auc), np.std(all_roc_auc)), log=None)
    utils.log_to_file_and_console(log_file_name=log_filename, fmt='aupr:  {0:6f}({1:6f})'.format(
        np.mean(all_aupr), np.std(all_aupr)), log=None)
    utils.log_to_file_and_console(log_file_name=log_filename, fmt='rec:  {0:6f}({1:6f})'.format(
        np.mean(all_rec), np.std(all_rec)), log=None)
    utils.log_to_file_and_console(log_file_name=log_filename, fmt='prec:  {0:6f}({1:6f})'.format(
        np.mean(all_prec), np.std(all_prec)), log=None)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HGANDDS training')
    parser.add_argument('--data_type', type=str, default='drugcombdb', help="Types of drug combination data sets")
    parser.add_argument('--gpu_index', type=int, default=0, help="GPU index for training")
    parser.add_argument('--data_set_filename', type=str, default="drugcomb.csv",
                        help="The filename of drug combination data")
    parser.add_argument('--hidden_channels', type=int, default=1024,
                        help="The number of hidden neurons in the model")
    parser.add_argument('--drug_feature_length', type=int, default=384, help="Dimensions of drug features")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size of training data")
    parser.add_argument('--n_epochs', type=int, default=200, help="Epochs of training")
    parser.add_argument('--split', type=int, default=10, help="Split of training data")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate in training process")
    args = parser.parse_args()
    
    data_type = args.data_type
    gpu_index = args.gpu_index
    log_filename = utils.generate_log_filename('HGANDDS_{}'.format(data_type))
    device = torch.device("cuda", gpu_index)
    data_set_path = 'data/{}/{}'.format(data_type, args.data_set_filename)
    utils.log_to_file_and_console(log_file_name=log_filename, fmt='args:{}'.format(args), log=None)
    comb_data = utils.init_hetero_data(
        data_type=data_type, device=device
    )
    run_expriments(device, comb_data, data_set_path, log_filename, args)
