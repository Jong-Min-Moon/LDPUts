import torch

def projection_orth_one(alphabet_size):
    identity = torch.eye(alphabet_size)
    one_one_t = torch.ones( torch.Size([alphabet_size,alphabet_size]) )
    one_one_t_over_d = one_one_t.div(alphabet_size)
    one_projector = identity.sub(one_one_t_over_d)
    return(one_projector)

def get_sample_size(data):
    if data.dim() == 1:
        return( data.size(dim = 0) )
    elif data.dim() == 2:
        return( data.size(dim = 0) )
    else:
        return # we only use up to 2-dimensional tensor, i.e. matrix  
    

def get_dimension(data):
    if data.dim() == 1:
        return(1)
    elif data.dim() == 2:
        return( data.size(dim = 1) )
    else:
        return # we only use up to 2-dimensional tensor, i.e. matrix
    
def chi_sq_dist(x1,x2):
    return( torch.norm((x1-x2)/((x1+x2).sqrt()), p=2))