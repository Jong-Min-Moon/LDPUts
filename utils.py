import torch




def get_sample_size(data):
    if data.dim() == 1:
        return( data.size(dim = 0) )
    elif data.dim() == 2:
        return( data.size(dim = 0) )
    else:
        raise Exception("we only use up to 2-dimensional tensor")
    

def get_dimension(data):
    if data.dim() == 1:
        return(1)
    elif data.dim() == 2:
        return( data.size(dim = 1) )
    else:
        raise Exception("we only use up to 2-dimensional tensor")

def chi_sq_dist(x1,x2):
    return( torch.norm((x1-x2)/((x1+x2).sqrt()), p=2))



