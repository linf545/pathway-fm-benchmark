import torch
from torch_geometric.data import Data
from gnn.models.gnn import PretrainedGNN

ckpt_path = "/private/vina/model_repos/NaFM/NaFM-Official/NaFM.ckpt"
ckpt = torch.load(ckpt_path, map_location="cpu")
hp = ckpt["hyper_parameters"]

model = PretrainedGNN(
    num_layer=hp["num_layer"],
    emb_dim=hp["emb_dim"],
    drop_ratio=hp["drop_ratio"],
    gnn_type=hp["gnn_type"],
    feat_dim=hp["feat_dim"],
)

prefix = "model.pretrain_gnn."
state = {
    k[len(prefix):]: v
    for k, v in ckpt["state_dict"].items()
    if k.startswith(prefix)
}

missing, unexpected = model.load_state_dict(state, strict=False)

print("NaFM checkpoint:", ckpt_path)
print("num_layer:", hp["num_layer"])
print("emb_dim:", hp["emb_dim"])
print("feat_dim:", hp["feat_dim"])
print("gnn_type:", hp["gnn_type"])
print("missing:", len(missing), missing[:20])
print("unexpected:", len(unexpected), unexpected[:20])

model.eval()

x = torch.tensor([[6, 0, 0], [6, 0, 0]], dtype=torch.long)
edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t().contiguous()
edge_attr = torch.tensor([[1, 0], [1, 0]], dtype=torch.long)
batch = torch.zeros(x.size(0), dtype=torch.long)

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

with torch.no_grad():
    graph_emb = model(data, masked=False)
    node_emb = model.representation_model(data)

print("graph_emb shape:", tuple(graph_emb.shape))
print("node_emb shape:", tuple(node_emb.shape))
print("NaFM smoke test passed")