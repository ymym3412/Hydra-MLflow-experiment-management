from pathlib import Path

import torch


# モデルを保存する
def save_model(state, file_name):
    p_path = Path(file_name).parent
    if not p_path.exists():
        p_path.mkdir(parents=True)
    torch.save(state, file_name)


# モデルをロードする
def load_model(model, file_name):
    state_dict = torch.load(file_name, map_location='cpu')
    model.load_state_dict(state_dict)
