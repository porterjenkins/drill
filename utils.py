def get_n_params(model):
    n_params = 0
    for layer in model.parameters():
        dims = layer.size()
        cnt = dims[0]
        for d in dims[1:]:
            cnt *= d
        n_params += cnt

    return n_params