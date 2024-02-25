import sys
sys.path.insert(0, '/mnt/nas/users/user213/LDPUts')
import gc
from discretizer import discretizer
from client import client
import torch
from server import server_multinomial_bitflip
from data_generator import data_generator
import time
import numpy as np



device_y = torch.device("cuda:1")
device_z = torch.device("cuda:1")

priv_mech = "bitflip"
statistic = "projchi"


sample_size_list = [500000]
privacy_level = 0.5
bump_size = 0.0009
alphabet_size = 1000
n_permutation = 999
print(priv_mech + "_" + statistic)
print(f"privacy level = {privacy_level}")

n_test = 200
significance_level = 0.05
server_private = server_multinomial_bitflip(privacy_level)



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
p_value_array = np.zeros([n_test, 1])

for sample_size in sample_size_list:
    print("\n")
    print("#########################################")
    print(f"sample size = {sample_size}")

    t = time.time()
    for i in range(n_test):
        t_start_i = time.time()
        torch.manual_seed(i)
  

        server_private.load_private_data_multinomial(
            LDPclient.release_private(priv_mech,
                data_gen.generate_multinomial_data(p1, sample_size),
                alphabet_size,
                privacy_level,
                device_y
                ),
            LDPclient.release_private(priv_mech,
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

    
        t_end_i = time.time() - t_start_i
        print(f"pval: {p_value_array[i,0]} -- {i+1}th test, time elapsed {t_end_i} -- emperical power so far: {(p_value_array[0:(i+1)] < significance_level).mean()}")


    elapsed = time.time() - t
    print(elapsed)

    with open('./pval_k_' + str(int(alphabet_size)) + '_priv_' + str(int(privacy_level*10)) + "_" + priv_mech + "_" + statistic + '_n_' + str(int(sample_size)) + '.npy', 'wb') as f:
        np.save(f, p_value_array)





    

    
    
    


