import yaml


def get_n_params(model):
    n_params = 0
    for layer in model.parameters():
        dims = layer.size()
        cnt = dims[0]
        for d in dims[1:]:
            cnt *= d
        n_params += cnt

    return n_params

def print_yaml(cfg):
    for k, v in cfg.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for k2, v2 in v.items():
                print(f"\t - {k2}: {v2}")
        else:
            print(f"{k}: {v}")


def get_yaml_cfg(fpath):
    with open(fpath, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    print("CONFIG SETTINGS:")
    print_yaml(cfg)
    return cfg