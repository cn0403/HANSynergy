# HANSynergy
Python 3.8.17 

# Installation
```
conda create -n HANSynergy python=3.8.17
conda activate HANSynergy
pip install -r requirements.txt
```

# Dataset
The data used in this paper are all in the data folder, which can be decompressed by the following command.

```
unzip data/drugcomb.zip
unzip data/drugcombdb.zip
unzip data/drugcomb_early.zip  
unzip data/drug_protein_links.csv.zip
unzip data/drug_property.csv.zip   
unzip data/protein_protein_links.csv.zip
```

Drugcomb Drugcombdb Drugcombdb_early three folders are the three data sets used in this paper.
`drug_property.csv`
It is the cid and the corresponding smiles sequence we collected.
`drug_protein_links.csv`
It's the data set of the link between drug and protein that we collected from DrugCombDB.
`protein_protein_links.csv`
It's the data set of the link between protein and protein that we collected from DrugCombDB.
`cell_protein_tissue.csv`
It is the data set of cell line, tissue and protein link that we collected.
# Training
To train HANSynergy, please input following in terminal.

```
python main.py
```

There are some optional parameters.
`--data_type`
Types of drug combination data.
`--data_set_filename`
File name of drug combination data.
`--hidden_channels`
The number of hidden neurons in the model.
`--drug_feature_length`
Dimensions of drug features.
`--batch_size`
Batch size of training data.
`--n_epochs`
Epochs of training.
`--split`
Split of training data
`--lr`
Learning rate in training process

For example, to train HANSynergy on Drugcomb dataset,

```
python main.py --data_type drugcomb --data_set_filename drug_comb.csv --hidden_channels 1024 --drug_feature_length 384 --batch_size 256 --n_epochs 200 --split 10 --lr 1e-3
```
