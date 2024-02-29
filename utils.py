import json
import pickle
import torch
import time
import pandas as pd
from rdkit import Chem
from sentence_transformers import SentenceTransformer
# from bio_embeddings.embed import SeqVecEmbedder
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdMolDescriptors
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score, average_precision_score
import time
from torch_geometric.data import HeteroData
import os
import torch_geometric.transforms as T
from tqdm import tqdm

nbits = 1024  # 1024
fpFunc_dict = {}
fpFunc_dict['hashap'] = lambda m: rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
    m, nBits=nbits)


def evaluate(
    model_name: str,
    log_filename: str,
    y_pred,
    y_score,
    y_true
):
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    auc = roc_auc_score(y_score=y_score, y_true=y_true)
    aupr = average_precision_score(y_true=y_true, y_score=y_score)
    log_to_file_and_console(log_file_name=log_filename, fmt='model: {}, acc: {}, recall: {}, precision: {}, f1: {}, auc: {}, aupr: {}'.format(
        model_name, acc, recall, precision, f1, auc, aupr))


def log_to_file_and_console(
    log_file_name: str,
    fmt: str = '',
    log=None,
    mode: str = 'a+'
):
    """_summary_

    Args:
        fmt (str): _description_
        log (str): _description_
        log_file_name (str): _description_
    """
    log_file = open(log_file_name, mode=mode, encoding='utf-8')
    if log == None:
        log = ''
    print(fmt, log)
    print(fmt, log,  file=log_file)


def get_filename_with_suffix(filename: str, suffix: str):
    filename_format = filename[-4:]
    filename = filename[:-4] + suffix+filename_format
    return filename


def get_dict_from_json_filename(
    filename: str
) -> dict:
    return json.loads(open(filename, 'r').read())


def save_data_by_pickle(
    data,
    filename: str,
):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def save_json_filename(
    dict: dict,
    filename: str
) -> None:
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dict, f, indent=4, sort_keys=False, separators=(',', ':'))


def get_data_from_pickle(
    filename: str,
):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def get_drug_embedding_by_smile(
    smile: str,
    device: torch.device,
    drug_model,
    cid,
    drug_feature_dict
):
    if cid in drug_feature_dict.keys():
        f = drug_feature_dict[cid]
    else:
        f = drug_model.encode(smile)
        drug_feature_dict[cid] = f.tolist()
    return f, get_cid_fp_by_smile(smile=smile)

# embedder = SeqVecEmbedder()


def get_drug_embedding_by_unimol(
    smile: str,
    device: torch.device,
    drug_model,
    cid: int,
    embedding_dict: dict,
):
    if cid in embedding_dict.keys():
        return embedding_dict[cid], get_cid_fp_by_smile(smile=smile)
    return [1] * 512, get_cid_fp_by_smile(smile=smile)


def get_protein_embedding_by_seq(
    seq: str,
    drug_model,
):
    # embedding = embedder.embed(seq)
    return drug_model.encode(seq)


cell_embedding_dict = get_dict_from_json_filename('../data/context_set_m.json')


def get_cell_line_embedding(
    cell: str,
):
    return cell_embedding_dict[cell]


def get_cid_fp_by_smile(
    smile: str,
    fp_name: str = 'hashap'
):
    mol = Chem.MolFromSmiles(smile)
    fp = fpFunc_dict[fp_name](mol)
    return fp


def get_dict_from_df(
    df: pd.DataFrame,
    key_index: int,
    val_index: int,
) -> dict:
    dict = {}
    for index, row in df.iterrows():
        dict[row[key_index]] = row[val_index]
    return dict


def add_time_suffix(
    prefix: str,
):
    return prefix + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))



def generate_entidy_id_dict(dict):
    entidy_id_dict = {}
    drug_index, protein_index, cell_index, tissue_index = 0, 0, 0, 0
    for index, key in tqdm(enumerate(dict), total=len(dict)):
        if 'cid_' in key and key not in entidy_id_dict.keys():
            entidy_id_dict[key] = drug_index
            drug_index += 1
        elif 'ENSP' in key and key not in entidy_id_dict.keys():
            entidy_id_dict[key] = protein_index
            protein_index += 1
        elif 'cell_' in key and key not in entidy_id_dict.keys():
            entidy_id_dict[key] = cell_index
            cell_index += 1
        elif 'tissue_' in key and key not in entidy_id_dict.keys():
            entidy_id_dict[key] = tissue_index
            tissue_index += 1
        elif 'gene_' in key:
            pass
        else:
            raise ("index:{}, key:{} not any node type".format(index, key))
    return entidy_id_dict


def load_node_id(entidy_id_dict):
    drug_node_id, protein_node_id, cell_node_id, tissue_node_id = [], [], [], []
    cid_list, protein_list, cell_list, tissue_list = [], [], [], []
    for index, key in tqdm(enumerate(entidy_id_dict), total=len(entidy_id_dict)):
        if 'cid_' in key:
            drug_node_id.append(entidy_id_dict[key])
            cid_list.append(key)
        elif 'ENSP' in key:
            protein_node_id.append(entidy_id_dict[key])
            protein_list.append(key)
        elif 'cell_' in key:
            cell_node_id.append(entidy_id_dict[key])
            cell_list.append(key)
        elif 'tissue_' in key:
            tissue_node_id.append(entidy_id_dict[key])
            tissue_list.append(key)
        else:
            raise ("index:{}, key:{} not any node type".format(index, key))
    return torch.tensor(drug_node_id), torch.tensor(protein_node_id), torch.tensor(cell_node_id), torch.tensor(tissue_node_id), cid_list, protein_list, cell_list, tissue_list


def get_drug_feature(
    cid_list,
    device,
    data_type: str,
    is_use_unimol: bool = False
):
    print('get drug feature')
    cid_smile_dict = get_dict_from_json_filename('../data/cid_smile.json')
    cid_drug_feature_filename = './cid_drug_feature.pickle'
    if os.path.exists(cid_drug_feature_filename):
        drug_feature_dict = get_data_from_pickle(cid_drug_feature_filename)
    else:
        drug_feature_dict = {}
    drug_feature_list = []
    drug_fp_list = []
    drug_model = SentenceTransformer('pretrained/simcsesqrt-model', device=device)
    if is_use_unimol:
        embedding_dict = get_dict_from_json_filename('./cid_repr.json')
    for cid in tqdm(cid_list):
        smile = cid_smile_dict[cid]
        # f, fp = get_drug_embedding_by_unimol(smile=smile, device=device, drug_model=drug_model, cid=cid, embedding_dict=embedding_dict)
        f, fp = get_drug_embedding_by_smile(
            smile=smile, drug_model=drug_model, device=device, cid=cid, drug_feature_dict=drug_feature_dict)
        drug_feature_list.append(f)
        drug_fp_list.append(fp)
    # save_data_by_pickle(drug_feature_dict, cid_drug_feature_filename)
    drug_feature_tensor = torch.tensor(drug_feature_list, dtype=torch.float32)
    drug_fp_tensor = torch.tensor(drug_fp_list, dtype=torch.float32)
    return drug_feature_tensor, drug_fp_tensor


not_usefule_set_filename = '../data/not_userful_set.pickle'


def get_protein_feature(
    protein_list,device
):
    print('get protein feature')
    pretein_property_filename = '../data/protein_property.csv'
    protein_seq_df = pd.read_csv(pretein_property_filename)
    protein_seq_dict = get_dict_from_df(protein_seq_df, 1, 2)
    protein_feature_list = []
    model = SentenceTransformer('pretrained/simcsesqrt-model', device=device)
    for protein in tqdm(protein_list):
        seq = protein_seq_dict[protein]
        protein_feature_list.append(get_protein_embedding_by_seq(seq, model))
    return torch.tensor(protein_feature_list, dtype=torch.float32)


def get_cell_feature(
    cell_list,
):
    cell_feature_list = []
    for cell in tqdm(cell_list):
        cell = cell[5:]
        cell_feature_list.append(get_cell_line_embedding(cell=cell))
    return torch.tensor(cell_feature_list, dtype=torch.float32)


def init_drug_cell_edge(
    entidy_id_dict: dict,
    data_type: str = 'drugcombdb'
):
    print("init drug cell edge")
    cid_relation_filename = '../data/{}/drug_comb.csv'.format(data_type)
    save_filename = './{}/drug_cell_edge.pickle'.format(data_type)
    if os.path.exists(save_filename):
        print("load exist data:{}".format(save_filename))
        return get_data_from_pickle(save_filename)
    cid_id_list, cell_line_list = [], []
    df = pd.read_csv(cid_relation_filename, sep=',', dtype='str')
    for index, row in tqdm(df.iterrows(), total=len(df)):
        cid1, cid2, cell = 'cid_'+row[1], 'cid_'+row[2], 'cell_'+row[3]
        cid_id1, cid_id2, cell_id = entidy_id_dict[cid1], entidy_id_dict[cid2], entidy_id_dict[cell]
        cid_id_list.append(cid_id1)
        cell_line_list.append(cell_id)
        cid_id_list.append(cid_id2)
        cell_line_list.append(cell_id)
    return torch.tensor([cid_id_list, cell_line_list])


def init_drug_proetin_edge(
    entidy_id_dict: dict,
    data_type: str = 'drugcombdb'
):
    print("init drug protein proetin edge")
    save_filename = './{}/drug_proetin_edge.pickle'.format(data_type)
    if os.path.exists(save_filename):
        print("load exist data:{}".format(save_filename))
        return get_data_from_pickle(save_filename)
    drug_protein_filename = '../data/drug_protein_links.csv'
    df = pd.read_csv(drug_protein_filename, sep=',')
    cid_id_list, protein_id_list, combined_score_list = [], [], []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        cid, combined_score, protein = row[0], row[2], row[1]
        cid_id, protein_id = entidy_id_dict[cid], entidy_id_dict[protein]
        cid_id_list.append(cid_id)
        protein_id_list.append(protein_id)
        combined_score_list.append(combined_score)
    return torch.tensor([cid_id_list, protein_id_list]), torch.tensor([combined_score_list])


def init_proetin_proetin_edge(
    entidy_id_dict,
    data_type: str = 'drugcombdb'
):
    print("init protein proetin edge")
    save_filename: str = './{}/proetin_proetin_edge.pickle'.format(data_type)
    if os.path.exists(save_filename):
        print("load exist data:{}".format(save_filename))
        return get_data_from_pickle(save_filename)
    protein_protein_filename = '../data/protein_protein_links.csv'
    df = pd.read_csv(protein_protein_filename, sep=',')
    protein_id_list1, protein_id_list2, combined_score_list = [], [], []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        protein1, combined_score, protein2 = row[0], row[2], row[1]
        protein_id1, protein_id2 = entidy_id_dict[protein1], entidy_id_dict[protein2]
        protein_id_list1.append(protein_id1)
        protein_id_list2.append(protein_id2)
        combined_score_list.append(combined_score)
    return torch.tensor([protein_id_list1, protein_id_list2]), torch.tensor([combined_score_list])


def init_cell_protein_tissue_edge(
    entidy_id_dict,
    data_type: str = 'drugcombdb'
):
    print("init cell protein tissue edge")
    cell_protein_filename = '../data/cell_protein_tissue.csv'
    df = pd.read_csv(cell_protein_filename)
    cell_list_with_protein, cell_list_with_tissue = [], []
    protein_list_with_cell, tissue_list_with_cell = [], []
    for index, row in tqdm(df.iterrows(), total=(len(df))):
        cell, protein, tissue = row[0], row[1], row[2]
        cell = 'cell_' + cell
        tissue = 'tissue_' + tissue
        cell_id, protein_id, tissue_id = entidy_id_dict[cell], entidy_id_dict[protein], entidy_id_dict[tissue]
        cell_list_with_protein.append(cell_id)
        protein_list_with_cell.append(protein_id)
        cell_list_with_tissue.append(cell_id)
        tissue_list_with_cell.append(tissue_id)
    return torch.tensor([cell_list_with_protein, protein_list_with_cell]), torch.tensor([cell_list_with_tissue, tissue_list_with_cell])



def generate_log_filename(
    mode_name: str,
):
    return './log/'+mode_name+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))+".txt"

def init_hetero_data(
    device,
    save_filename: str = './data/drugcombdb/comb_data.pickle',
    data_type: str = 'drugcombdb',
):
    save_filename = save_filename.replace('drugcombdb', data_type)
    print("try to get hetero data from:{}".format(save_filename))
    if os.path.exists(save_filename):
        print("load exist data from:{}".format(save_filename))
        data = get_data_from_pickle(save_filename)
        return data
    entidy_id_dict_filename = '../data/{}/id_dict.json'.format(data_type)
    entidy_id_dict = get_dict_from_json_filename(entidy_id_dict_filename)
    drug_node_id, protein_node_id, cell_node_id, tissue_node_id, cid_list, protein_list, cell_list, tissue_list = load_node_id(
        entidy_id_dict)
    data = HeteroData()
    data['drug'].node_id = drug_node_id
    data['protein'].node_id = protein_node_id
    data['cell'].node_id = cell_node_id
    data['tissue'].node_id = tissue_node_id

    data['protein'].feature = get_protein_feature(protein_list,device)
    data['drug'].feature, data['drug_fp'].feature = get_drug_feature(cid_list, device, data_type, False)
    data['cell'].feature = get_cell_feature(cell_list)

    data['drug', 'with', 'drug_fp'].edge_index = torch.tensor([list(drug_node_id), list(drug_node_id)])
    data['drug', 'with', 'cell'].edge_index = init_drug_cell_edge(entidy_id_dict, data_type=data_type)
    data['drug', 'combined_score', 'protein'].edge_index, \
        data['drug', 'combined_score', 'protein'].edge_attr = init_drug_proetin_edge(
            entidy_id_dict, data_type=data_type)
    data['protein', 'combined_score', 'protein'].edge_index,\
        data['drug', 'combined_score', 'protein'].edge_attr = init_proetin_proetin_edge(
            entidy_id_dict, data_type=data_type)
    data['cell', 'with', 'protein'].edge_index, \
        data['cell', 'with', 'tissue'].edge_index = init_cell_protein_tissue_edge(entidy_id_dict, data_type=data_type)
    data = T.ToUndirected()(data)
    save_data_by_pickle(data, save_filename)
    return data

def get_cid(
    cid:str,
):
    if 'cid_' in cid:
        return cid
    return 'cid_'+cid

def get_cell(
    cid:str,
):
    if 'cell_' in cid:
        return cid
    return 'cell_'+cid