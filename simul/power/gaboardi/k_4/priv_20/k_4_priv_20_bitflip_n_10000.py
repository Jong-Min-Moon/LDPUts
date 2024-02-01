import sys
sys.path.insert(0, '/mnt/nas/users/user213/LDPUts')
import gc
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

sample_size = 10000
privacy_level = 2.0




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


alphabet_size = 4
    
data_gen = data_generator()
LDPclient = client()


server_bitflip = server_multinomial_bitflip(privacy_level)

p_value_array = np.zeros([n_test, 2])
t_start = time.time()
for i in range(n_test):
    t_start_i = time.time()
    torch.manual_seed(i)
 
    server_bitflip.load_private_data_multinomial_y(
        LDPclient.release_bitflip(
            data_gen.generate_multinomial_data(p1, sample_size),
            alphabet_size,
            privacy_level
            ),
        alphabet_size
    )

    server_bitflip.load_private_data_multinomial_z(
        LDPclient.release_bitflip(
            data_gen.generate_multinomial_data(p2, sample_size),
            alphabet_size,
            privacy_level
            ),
        alphabet_size
    )
    p_value_array[i,0] = server_bitflip.release_p_value_permutation(n_permutation)
    p_value_array[i,1] = server_bitflip.release_p_value()
    
    t_end_i = time.time() - t_start_i
    print(f"pval: {p_value_array[i,0]} -- {p_value_array[i,1]}(perm), {i+1}th test, time elapsed {t_end_i}")
elapsed = time.time() - t_start
print(elapsed)
with open('./p_value_array_priv_bitflip_priv_' + str(int(privacy_level*10)) +'_k_4_n_' + str(int(sample_size)) + '.npy', 'wb') as f:
    np.save(f, p_value_array)
