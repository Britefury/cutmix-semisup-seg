
def freeze_bn_module(m):
    """ Freeze the module `m` if it is a batch norm layer.

    :param m: a torch module
    :param mode: 'eval' or 'no_grad'
    """
    classname = type(m).__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
