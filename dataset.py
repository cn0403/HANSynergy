import torch
import torch.utils.data as data
import numpy as np
import json
import utils
from rdkit import Chem
from sentence_transformers import SentenceTransformer
from rdkit.Chem import rdMolDescriptors

nbits = 1024  # 1024

class DrugSynergyDataset(data.Dataset):
    def __init__(self, drug_1_smiles, drug_2_smiles, Y, cell_line, device, maxCompoundLen=128, dataset_type = 'drugcomb'):
        self.maxCompoundLen = maxCompoundLen
        self.drug_1_smiles = drug_1_smiles
        self.drug_2_smiles = drug_2_smiles
        self.cell_line = cell_line
        self.Y = Y
        self.device = device
        self.len = len(self.Y)
        self.cell_line_features = json.loads(open("./{}/cell_line_set_m.json".format(dataset_type), 'r').read()) # cell_line对应的feature
        self.cid_smile_dict = utils.get_dict_from_json_filename('../data/cid_smile.json') # cid对应的smile
        self.entity_id_dict = utils.get_dict_from_json_filename('../data/{}/id_dict.json'.format(dataset_type)) # 每个实体对应的id
        model_name = 'pretrained/simcsesqrt-model'
        self.drug_model = SentenceTransformer(model_name, device=device)
        self.encode_smiles()

    def __len__(self):
        return self.len

    def encode_smiles(self):
        self.simse = {}
        self.drug2fps = {}
        for smile in list(set(self.drug_1_smiles))+list(set(self.drug_2_smiles)):
            smile_str = self.cid_smile_dict['cid_'+smile]
            self.simse[smile] = self.drug_model.encode(smile_str[:self.maxCompoundLen])
            mol = Chem.MolFromSmiles(smile_str)
            fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nbits)
            self.drug2fps[smile] = np.asarray(fp)

    def __getitem__(self, index):
        compounds_1 = self.simse[self.drug_1_smiles[index]]
        compounds_2 = self.simse[self.drug_2_smiles[index]]
        synergyScore = self.Y[index]
        fp1 = self.drug2fps[str(self.drug_1_smiles[index])]
        fp2 = self.drug2fps[str(self.drug_2_smiles[index])]
        cell_line_features = self.cell_line_features[self.cell_line[index]]
        return [
            torch.FloatTensor(compounds_1),
            torch.FloatTensor(compounds_2),
            torch.LongTensor([synergyScore]),
            torch.FloatTensor([cell_line_features]),
            torch.FloatTensor(fp1),
            torch.FloatTensor(fp2),
            self.entity_id_dict['cid_'+self.drug_1_smiles[index]],
            self.entity_id_dict['cid_'+self.drug_2_smiles[index]],
            self.entity_id_dict['cell_'+self.cell_line[index]]]
