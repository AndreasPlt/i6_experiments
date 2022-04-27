
def make_subsampling_001(
    net=None,
    in_l="source0",
    time_reduction=None,

    #specific:
    embed_dropout = None,
    embed_l2 = None,

    # General
    initialization = None,
    model_dim = None
):
    assert net, "need network"
    net.update({
        "conv0_0" : {
            "class": "conv", 
            "from": in_l, 
            "padding": "same",
            "filter_size": (3, 3),
            "n_out": 32, 
            "activation": None, 
            "with_bias": True, 
            "in_spatial_dims": ["T", "dim:50"]},  # (T,50,32)
        "conv0_1" : {
            "class": "conv", 
            "from": f"conv0_0", 
            "padding": "same", 
            "filter_size": (3, 3),
            "n_out": 32, 
            "activation": 'relu', 
            "with_bias": True, 
            "in_spatial_dims": ["T", "dim:50"]},  # (T,50,32)
        "conv0p" : {
            "class": "pool", 
            "mode": "max", 
            "padding": "same", 
            "pool_size": (1, 2), 
            'strides': (1, 2),
            "from": "conv0_1", 
            "in_spatial_dims": ["T", "dim:50"] }, # (T, 25, 32)
        "conv1_0" : {
            "class": "conv", 
            "from": "conv0p", 
            "padding": "same", 
            "filter_size": (3, 3), 
            "n_out": 64,
            "activation": None, 
            "with_bias": True,
            "in_spatial_dims": ["T", "dim:25"] }, # (T, 25, 64)
        "conv1_1" : {
            "class": "conv", 
            "from": "conv1_0", 
            "padding": "same", 
            "filter_size": (3, 3),
            "n_out": 64, 
            "activation": 'relu', 
            "with_bias": True, 
            "in_spatial_dims": ["T", "dim:25"]}, # (T,25,64)
        "conv1p" : {
            "class": "pool", 
            "mode": "max", 
            "padding": "same", 
            "pool_size": (time_reduction, 1), 
            'strides': (time_reduction, 1),
            "from": "conv1_1", 
            "in_spatial_dims": ["T", "dim:25"]},
        "conv_merged" : {
            "class": "merge_dims", 
            "from": "conv1p", 
            "axes": ["dim:25", "dim:64"]},
        'embedding': { 
            'L2': embed_l2,
            'activation': None,
            'class': 'linear',
            'forward_weights_init': initialization,
            'from': ['conv_merged'],
            'n_out': model_dim,
            'with_bias': True},
        'embedding_dropout': {
            'class': 'dropout', 
            'dropout': embed_dropout, 
            'from': ['embedding']},
    })
    return net, "embedding_dropout"


def make_unsampling_001(
    net=None,
    in_l=None,

    time_reduction=None
):
    net.update({
        'encoder' : {
            'class': 'layer_norm',
            'from': [in_l]},
        'upsampled2': { 
            'activation': 'relu',
            'class': 'transposed_conv',
            'filter_size': (3,),
            'from': ['encoder'],
            'n_out': 256,
            'strides': (time_reduction,),
            'with_bias': True}
    })

    return net, "upsampled2"

# The espnet subsampling:

