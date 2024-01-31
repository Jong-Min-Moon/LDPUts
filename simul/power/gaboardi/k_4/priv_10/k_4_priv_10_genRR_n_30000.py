import sys
sys.path.insert(0, '/mnt/nas/users/mjm/LDPUts')

from discretizer import discretizer
from client import client
import torch
from server import server_ell2, server_multinomial_genrr, server_multinomial_bitflip
from data_generator import data_generator
from discretizer import discretizer
import time
import numpy as np
from scipy.stats import chi2
from utils import chi_sq_dist

device = torch.device("cpu")
print(device)

sample_size = 30000
privacy_level = 1.0











n_test = 300
n_permutation = 399
significance_level = 0.05
alphabet_size = 4

p1 = torch.ones(alphabet_size).div(alphabet_size)

bump_size = 0.01
p2 = p1.add(
    torch.remainder(
    torch.tensor(range(alphabet_size)),
    2
    ).add(-1/2).mul(2).mul(bump_size)
)
print(p2)

def run_simul_k_4_genrr(device, n_test, n_permutation, p1, p2, sample_size, privacy_level):
    alphabet_size = 4
    
    data_gen = data_generator(device)
    LDPclient = client(device, privacy_level)


    server_genrr = server_multinomial_genrr(device, privacy_level)

    p_value_array = np.zeros([n_test, 2])
    t = time.time()
    for i in range(n_test):
        torch.manual_seed(i)
        data_y = data_gen.generate_multinomial_data(p1, sample_size)
        data_z = data_gen.generate_multinomial_data(p2, sample_size)
        LDPclient.load_data_multinomial(data_y, data_z, alphabet_size)
    
       
        data_genrr_y, data_genrr_z = LDPclient.release_genrr()
        server_genrr.load_private_data_multinomial(data_genrr_y, data_genrr_z, alphabet_size)
        p_value_array[i,0] = server_genrr.release_p_value_permutation(n_permutation)
        p_value_array[i,1] = server_genrr.release_p_value()
        print(f"{i+1}th test: {p_value_array[i,0]}, {p_value_array[i,1]}")

    elapsed = time.time() - t
    print(elapsed)
    with open('./p_value_array_priv_genrr_priv_' + str(int(privacy_level*10)) +'_k_4_n_' + str(int(sample_size)) + '.npy', 'wb') as f:
        np.save(f, p_value_array)
    return(p_value_array)

result = run_simul_k_4_genrr(device, n_test, n_permutation, p1, p2, sample_size, privacy_level)