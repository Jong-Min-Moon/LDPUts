import sys
import logging

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

sample_size = 20
privacy_level = 2.0
bump_size = 0.0005
alphabet_size = 1000


n_test = 300
significance_level = 0.05

p1 = torch.ones(alphabet_size).div(alphabet_size)
p2 = p1.add(
    torch.remainder(
    torch.tensor(range(alphabet_size)),
    2
    ).add(-1/2).mul(2).mul(bump_size)
)
print(p2)

    
data_gen = data_generator(device)
LDPclient = client(device, privacy_level)


server_bitflip = server_multinomial_bitflip(device, privacy_level)

p_value_array = np.zeros([n_test, 1])
t_start = time.time()
for i in range(n_test):
    t_start_i = time.time()
    torch.manual_seed(i)
    data_y = data_gen.generate_multinomial_data(p1, sample_size)
    data_z = data_gen.generate_multinomial_data(p2, sample_size)
    LDPclient.load_data_multinomial(data_y, data_z, alphabet_size)
    
       
    data_bitflip_y, data_bitflip_z = LDPclient.release_bitflip()
    server_bitflip.load_private_data_multinomial(data_bitflip_y, data_bitflip_z, alphabet_size)
    p_value_array[i,0] = server_bitflip.release_p_value()
    
    t_end_i = time.time() - t_start_i
    print(f"pval: {p_value_array[i,0]} -- {i+1}th test, time elapsed {t_end_i}")
elapsed = time.time() - t_start
print(elapsed)
with open('./p_value_array_priv_bitflip_priv_' + str(int(privacy_level*10)) +'_k_500_n_' + str(int(sample_size)) + '.npy', 'wb') as f:
    np.save(f, p_value_array)

