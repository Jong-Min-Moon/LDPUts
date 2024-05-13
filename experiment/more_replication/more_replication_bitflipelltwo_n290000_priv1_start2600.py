alphabet_size = 1000
bump_size     = 0.0009
privacy_level = 1
sample_size   = 290000
n_permutation = 999
test_start    = 2600
n_test        = 200
table_name = 'ldp_disc_basic_comparison'
code_dir   = '/home1/jongminm/LDPUts/experiment/more_replication'
priv_mech  = 'bitflip'
statistic  = 'elltwo'

import sys
sys.path.insert(0, '/home1/jongminm/LDPUts')
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

def insert_data(data_entry):
    db_dir = "/home1/jongminm/LDPUts/experiment/LDP_minimax.db"
    sleep(randint(1,20))
    con = sqlite3.connect(db_dir)
    cursor_db = con.cursor()
    cursor_db.execute(
                f"INSERT INTO {table_name}(rep, dim, bump, priv_lev, sample_size, statistic, mechanism, statistic_val, p_val, compute_time, jobdate) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", data_entry
            )
    cursor_db.close()
    con.commit()
    con.close()
    print("db insert success")


method_name = priv_mech + statistic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

server_private_vec = {
    "elltwo":server_ell2(privacy_level),
    "chi":server_multinomial_genrr(privacy_level),
    "projchi":server_multinomial_bitflip(privacy_level)
    }
server_private = server_private_vec[statistic]





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
p_value_vec = np.zeros([n_test, 1])
statistic_vec = np.zeros([n_test, 1])
t = time.time()
            
for i in range(test_start, test_start+n_test):
    t_start_i = time.time()
    torch.manual_seed(i)
    server_private.load_private_data_multinomial(
        LDPclient.release_private(
            priv_mech,
            data_gen.generate_multinomial_data(p1, sample_size),
            alphabet_size,
            privacy_level,
            device
        ),
        LDPclient.release_private(
            priv_mech,
            data_gen.generate_multinomial_data(p2, sample_size),
            alphabet_size,
            privacy_level,
            device
        ),
    alphabet_size,
    device,
    device
    )
            
    time_now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    p_val_now, statistic_now    = server_private.release_p_value_permutation(n_permutation)
    p_value_vec[i-test_start]   = p_val_now
    statistic_vec[i-test_start] = statistic_now
    t_end_i = time.time() - t_start_i
    data_entry = (i+1, alphabet_size, bump_size, privacy_level, sample_size, statistic, priv_mech, statistic_now, p_val_now, float(t_end_i), time_now)
    print(data_entry)
    
    print(f"pval: {p_val_now} -- {i+1}th test, time elapsed {t_end_i} -- emperical power so far: {(p_value_vec[0:(i-test_start+1)] < significance_level).mean()}")
   
    #insert into database
    try:
        insert_data(data_entry)
    except:
        try:
            insert_data(data_entry)
        except:    
            print("db insert fail")
    server_private.delete_data()

elapsed = time.time() - t
print(elapsed)






    

    
    
    


