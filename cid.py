import requests

import pubchempy as pcp

def drugbank_ids_to_cids(drugbank_ids):
    cids = []
    for drugbank_id in drugbank_ids:
        try:
            compound = pcp.get_compounds(drugbank_id, 'name', record_type='3d')[0]
            cids.append(compound.cid)
        except:
            cids.append(None)  # Handle cases where CID is not found for a DrugBank ID
    return cids

# 示例使用
drugbank_ids = ["DB01234", "DB04567", "DB08901"]
cids = drugbank_ids_to_cids(drugbank_ids)

for i, drugbank_id in enumerate(drugbank_ids):
    cid = cids[i]
    if cid is not None:
        print(f"DrugBank ID {drugbank_id} 对应的 PubChem CID 是 {cid}")
    else:
        print(f"未找到 PubChem CID for DrugBank ID {drugbank_id}")

