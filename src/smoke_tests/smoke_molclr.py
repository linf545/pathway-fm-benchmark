import inspect
import torch

from models.gcn_molclr import GCN
from models.ginet_molclr import GINet

for cls in [GCN, GINet]:
    print("=" * 80)
    print(cls.__name__)
    print(inspect.signature(cls))

tests = [
    ("GCN", GCN, "ckpt/pretrained_gcn/checkpoints/model.pth"),
    ("GINet", GINet, "ckpt/pretrained_gin/checkpoints/model.pth"),
]

for name, cls, ckpt_path in tests:
    print("=" * 80)
    print(name)
    print("checkpoint:", ckpt_path)

    state = torch.load(ckpt_path, map_location="cpu")

    model = cls(
        num_layer=5,
        emb_dim=300,
        feat_dim=512,
        drop_ratio=0.0,
        pool="mean",
    )

    missing, unexpected = model.load_state_dict(state, strict=False)

    print("state_dict keys:", len(state))
    print("missing:", len(missing), missing[:20])
    print("unexpected:", len(unexpected), unexpected[:20])