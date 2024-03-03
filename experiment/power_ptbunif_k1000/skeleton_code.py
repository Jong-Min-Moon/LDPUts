


import sys
sys.path.insert(0, '/mnt/nas/users/user213/LDPUts')
import gc
from client import client
import torch
from server import server_multinomial_bitflip, server_multinomial_genrr, server_ell2
from data_generator import data_generator
import time
import numpy as np



device = torch.device("cuda:0")


#priv_mech_name_vec = ["lapu", "genrr", "bitflip"]
#statistic_name_vec = ["ell2", "chi", "projchi"]
priv_mech_name_vec = ["genrr"]
statistic_name_vec = ["chi"]

sample_size_vec = np.arange(1,21)*10000
privacy_level_vec = np.array([0.5, 1, 2])


n_permutation = 999
print(device)


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

for zz, method_name in enumerate(priv_mech_name_vec):
    statistic_name = statistic_name_vec[zz]
    for privacy_level in privacy_level_vec:
        server_private = [server_ell2(privacy_level), server_multinomial_genrr(privacy_level), server_multinomial_bitflip(privacy_level)]
        for sample_size in sample_size_vec:
            print(f"{method_name}, alpha={privacy_level}, sample size={sample_size}")
            print("#########################################")
            p_value_array = np.zeros([n_test, 1])
            t = time.time()
            
            for i in range(n_test):
                t_start_i = time.time()
                torch.manual_seed(i)
  

                server_private[zz].load_private_data_multinomial_y(
                    LDPclient.release_private(
                        method_name,
                        data_gen.generate_multinomial_data(p1, sample_size),
                        alphabet_size,
                        privacy_level,
                        device
                        ),
                    alphabet_size
                )

                server_private[zz].load_private_data_multinomial_z(
                    LDPclient.release_private(
                        method_name,
                        data_gen.generate_multinomial_data(p2, sample_size),
                        alphabet_size,
                        privacy_level,
                        device
                        ),
                    alphabet_size
                )
            
    
                p_value_array[i,0] = server_private[zz].release_p_value_permutation(n_permutation)
                server_private[zz].delete_data()
    
                t_end_i = time.time() - t_start_i
                elapsed = time.time() - t
                print(f"pval: {p_value_array[i,0]} -- {i+1}th test, time elapsed {t_end_i} -- emperical power so far: {(p_value_array[0:(i+1)] < significance_level).mean()}")
                print(elapsed)

                with open(code_dir+'/result/k' + str(int(alphabet_size)) + '_priv' + str(int(privacy_level*10)) + "_method" + method_name + statistic_name + '_n' + str(int(sample_size)) + '.npy', 'wb') as f:
                    np.save(f, p_value_array)





    

    
    
    


