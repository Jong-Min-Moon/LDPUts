import torch

def get_uniform_perturb(alphabet_size, bump_size):
    p = torch.ones(alphabet_size).div(alphabet_size)
    p2 = p.add(
        torch.remainder(
        torch.tensor(range(alphabet_size)),
        2
        ).add(-1/2).mul(2).mul(bump_size)
    )
    p1_idx = torch.cat( ( torch.arange(1, alphabet_size), torch.tensor([0])), 0)
    p1 = p2[p1_idx]
    return(p1, p2)

def split_perm(perm, n_1):
    perm_toY = perm[     :n_1]
    perm_toY_fromY = perm_toY[perm_toY < n_1]
    perm_toY_fromZ = perm_toY[perm_toY >= n_1] - n_1

    perm_toZ = perm[ n_1 :   ]
    perm_toZ_fromY = perm_toZ[perm_toZ < n_1]
    perm_toZ_fromZ = perm_toZ[perm_toZ >= n_1] - n_1
    return perm_toY_fromY, perm_toY_fromZ, perm_toZ_fromY, perm_toZ_fromZ

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



