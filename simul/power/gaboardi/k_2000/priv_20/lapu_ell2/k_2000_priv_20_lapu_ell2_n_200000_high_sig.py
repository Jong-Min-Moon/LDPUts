import sys
sys.path.insert(0, '/mnt/nas/users/user213/LDPUts')

from discretizer import discretizer
from client import client
import torch
import matplotlib.pyplot as plt
from server import server_ell2, server_multinomial_genRR, server_multinomial_bitflip
from data_generator import data_generator
from discretizer import discretizer
import time
import numpy as np
from scipy.stats import chi2
from utils import chi_sq_dist

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

sample_size = 200000
privacy_level = 2.0
bump_size = 0.0004
alphabet_size = 2000

n_permutation = 399
n_test = 300
significance_level = 0.05

p = torch.ones(alphabet_size).div(alphabet_size)
p2 = p.add(
    torch.remainder(
    torch.tensor(range(alphabet_size)),
    2
    ).add(-1/2).mul(2).mul(bump_size)
)
print(p2)
p1_idx = torch.cat( ( torch.arange(1, alphabet_size), torch.tensor([0])), 0)
p1 = p2[p1_idx]
print(p1)

    
data_gen = data_generator(device)
LDPclient = client(device, privacy_level)
server_elltwo = server_ell2(device, privacy_level)
p_value_array = np.zeros([n_test, 1])

t = time.time()
for i in range(n_test):
    t_start_i = time.time()
    torch.manual_seed(i)
    data_y = data_gen.generate_multinomial_data(p1, sample_size)
    data_z = data_gen.generate_multinomial_data(p2, sample_size)
    LDPclient.load_data_multinomial(data_y, data_z, alphabet_size)
    data_lapu_y, data_lapu_z = LDPclient.release_LapU()
    del data_y
    del data_z

    
    server_elltwo.load_private_data_multinomial(data_lapu_y, data_lapu_z, alphabet_size)
    p_value_array[i,0] = server_elltwo.release_p_value_permutation(n_permutation)

    server_elltwo.delete_data()
    t_end_i = time.time() - t_start_i
    print(f"pval: {p_value_array[i,0]} -- {i+1}th test, time elapsed {t_end_i}")    

elapsed = time.time() - t
print(elapsed)

with open('./p_value_array_priv_lapu_ell2_priv_' + str(int(privacy_level*10)) +'_k_2000_n_' + str(int(sample_size)) + 'high_sig.npy', 'wb') as f:
    np.save(f, p_value_array)





    

    
    
    


