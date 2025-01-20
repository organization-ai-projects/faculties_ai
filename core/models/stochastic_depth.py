################################################################################
#                           Stochastic Depth                                   #
################################################################################
def stochastic_depth(x, layer_dropout_prob, mode="row", training=True):
    """
    Applique un “drop” de la sortie de la couche avec une certaine probabilité,
    pour régulariser. Plus la couche est profonde, plus la proba peut être grande.
    """
    if not training or layer_dropout_prob == 0.0:
        return x
    # mode="row" => on génère un masque par batch
    keep_prob = 1.0 - layer_dropout_prob
    size = (x.shape[0], 1, 1) if mode == "row" else x.shape
    mask = torch.rand(size, device=x.device) < keep_prob
    x = x / keep_prob * mask
    return x