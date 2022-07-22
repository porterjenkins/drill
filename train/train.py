import argparse

from models.seq_transformer import build
from utils import get_yaml_cfg


def train(trn_cfg_path: str, model_cfg_path: str):
    trn_cfg = get_yaml_cfg(trn_cfg_path)
    model_cfg = get_yaml_cfg(model_cfg_path)

    model = build(model_cfg)




if __name__ == "__main__":
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--trn-cfg', type=str, default="./cfg_density_neonet.yaml", help='Path to train config file')
    parser.add_argument('--model-cfg', type=str, default=None, help='optional model cfg path override')
    args = parser.parse_args()

    train(args.trn_cg, args.model_cfg_path)
