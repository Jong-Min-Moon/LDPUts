alphabet_size = 1000
bump_size = 0.0009
privacy_level = 1
device_num = 
code_dir = '/mnt/nas/users/user213/LDPUts/experiment/power_ptbunif_k1000'
priv_mech = 'genrr'
statistic = 'chi'



import sys
sys.path.insert(0, '/mnt/nas/users/user213/LDPUts')
import gc
from client import client
import torch
from server import server_multinomial_bitflip, server_multinomial_genrr, server_ell2
from data_generator import data_generator
import time
import numpy as np

method_name = priv_mech + statistic
device_y = torch.device("cuda:0")
device_z = torch.device("cuda:0")

sample_size_vec = (50000 + np.arange(0,20)*30000)
server_private_vec = {
    "ell2":server_ell2(privacy_level),
    "chi":server_multinomial_genrr(privacy_level),
    "bitflip":server_multinomial_bitflip(privacy_level)
    }
server_private = server_private_vec[statistic]

n_permutation = 999
print(device_y)
print(device_z)
n_test = 200
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
    
data_gen = data_generator()
LDPclient = client()

for sample_size in sample_size_vec:
    print(f"{method_name}, alpha={privacy_level}, sample size={sample_size}")
    print("#########################################")
    p_value_array = np.zeros([n_test, 1])
    t = time.time()
            
    for i in range(n_test):
        t_start_i = time.time()
        torch.manual_seed(i)
        server_private.load_private_data_multinomial(
            LDPclient.release_private(
                priv_mech,
                data_gen.generate_multinomial_data(p1, sample_size),
                alphabet_size,
                privacy_level,
                device_y
            ),
            LDPclient.release_private(
                priv_mech,
                data_gen.generate_multinomial_data(p2, sample_size),
                alphabet_size,
                privacy_level,
                device_z
            ),
        alphabet_size,
        device_y,
        device_z
        )
            
    
        p_value_array[i,0] = server_private.release_p_value_permutation(n_permutation)
        server_private.delete_data()
    
        t_end_i = time.time() - t_start_i
        print(f"pval: {p_value_array[i,0]} -- {i+1}th test, time elapsed {t_end_i} -- emperical power so far: {(p_value_array[0:(i+1)] < significance_level).mean()}")
 
    elapsed = time.time() - t
    print(elapsed)

    with open(code_dir+'/result/k' + str(int(alphabet_size)) + '_priv' + str(int(privacy_level*10)) + "_method" + method_name + '_n' + str(int(sample_size)) + '.npy', 'wb') as f:
        np.save(f, p_value_array)





    

    
    
    


