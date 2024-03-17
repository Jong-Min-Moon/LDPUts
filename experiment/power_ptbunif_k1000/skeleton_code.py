
import sys
sys.path.insert(0, '/mnt/nas/users/user213/LDPUts')
import gc
from client import client
import torch
from server import server_multinomial_bitflip, server_multinomial_genrr, server_ell2
from data_generator import data_generator
import time
import numpy as np
import sqlite3
from datetime import datetime
from random import randint
from time import sleep


method_name = priv_mech + statistic
cuda_string = "cuda:" + str(device_num)
device_y = torch.device(cuda_string)
device_z = torch.device(cuda_string)

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
            
    time_now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    p_val_now = server_private.release_p_value_permutation(n_permutation)
    p_value_array[i] = p_val_now
    data_entry = (i+1, alphabet_size, bump_size, privacy_level, sample_size, statistic, priv_mech, p_value_array[i], time_now)
    t_end_i = time.time() - t_start_i
    print(f"pval: {p_val_now} -- {i+1}th test, time elapsed {t_end_i} -- emperical power so far: {(p_value_array[0:(i+1)] < significance_level).mean()}")
    server_private.delete_data()
   
    #insert into database
    try:
        sleep(randint(1,10))
        con = sqlite3.connect('/mnt/nas/users/user213/LDPUts/experiment/LDP_minimax.db')
        cursor_db = con.cursor()
        cursor_db.execute(
                "INSERT INTO ldp_disc_basic_comparison(rep, dim, bump, priv_lev, sample_size, statistic, mechanism, p_val, jobdate) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", data_entry,
            )
        con.close()
    except:
        sleep(randint(1,10))
        con = sqlite3.connect('/mnt/nas/users/user213/LDPUts/experiment/LDP_minimax.db')
        cursor_db = con.cursor()
        cursor_db.execute(
                "INSERT INTO ldp_disc_basic_comparison(rep, dim, bump, priv_lev, sample_size, statistic, mechanism, p_val, jobdate) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", data_entry,
            )
        con.close()       
 
elapsed = time.time() - t
print(elapsed)






    

    
    
    


