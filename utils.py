import torch

<<<<<<< HEAD
def projection_orth_one(alphabet_size):
    identity = torch.eye(alphabet_size)
    one_one_t = torch.ones( torch.Size([alphabet_size,alphabet_size]) )
    one_one_t_over_d = one_one_t.div(alphabet_size)
    one_projector = identity.sub(one_one_t_over_d)
    return(one_projector)
=======


>>>>>>> f117be6494ab04651dcf973c34fce5639a55bf1b

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
<<<<<<< HEAD
        return # we only use up to 2-dimensional tensor, i.e. matrix
    
def chi_sq_dist(x1,x2):
    return( torch.norm((x1-x2)/((x1+x2).sqrt()), p=2))
=======
        raise Exception("we only use up to 2-dimensional tensor")

def chi_sq_dist(x1,x2):
    return( torch.norm((x1-x2)/((x1+x2).sqrt()), p=2))




def get_permuted_twosample(data_y, data_z):
    dim_1 = get_dimension(data_y)
    dim_2 = get_dimension(data_z)
    n_1 = get_sample_size(data_y)
    n_2 = get_sample_size(data_z)
    if dim_1 != dim_2:
        raise Exception("different data dimensions")
    if dim_1 != 

    idx_to_1_from_1, idx_to_1_from_2, idx_to_2_from_1, idx_to_2_from_2 = get_idx_permute_twosample(n_1, n_2)
    if dim_1 == 1:
        data_y_permute = torch.cat(
            (data_y[idx_to_1_from_1], data_z[idx_to_1_from_2])
            )
        data_z_permute = torch.cat(
            (data_y[idx_to_2_from_1], data_z[idx_to_2_from_2])
            )       
    elif dim_1 == 2:
        data_y_permute =  torch.vstack(
            (self.data_y[idx_to_1_from_1], self.data_z[idx_to_1_from_2])
            )
        data_z_permute =  torch.vstack(
            (self.data_y[idx_to_2_from_1], self.data_z[idx_to_2_from_2])
            )
    
    return data_y_permute, data_z_permute
>>>>>>> f117be6494ab04651dcf973c34fce5639a55bf1b
