import torch
def get_sample_size(data):
    if data.dim() == 1:
        return( data.size(dim = 0) )
    elif data.dim() == 2:
        return( data.size(dim = 0) )
    else:
        return # we only use up to 2-dimensional tensor, i.e. matrix  