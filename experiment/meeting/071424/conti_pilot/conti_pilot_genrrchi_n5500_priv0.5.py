d = 3
n_bin = 4
privacy_level = 0.5
sample_size   = 5500
n_permutation = 999
n_test        = 200
test_start    = 1
k             = 64
table_name = 'conti_location'
code_dir   = '/home1/jongminm/LDPUts/experiment/meeting/071424/conti_pilot'
priv_mech  = 'genrr'
statistic  = 'chi'
db_dir = '/home1/jongminm/LDPUts/experiment/db/071424_LDPUts.db'

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

def insert_data(data_entry, db_dir):
    sleep(randint(1,20))
    con = sqlite3.connect(db_dir)
    cursor_db = con.cursor()
    cursor_db.execute(
                f"INSERT INTO {table_name}(rep, dim, bump, priv_lev, sample_size, statistic, mechanism, statistic_val, p_val, compute_time, jobdate, n_bin) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", data_entry
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

multiplier = 1/2
copula_mean_1 = -multiplier * torch.ones(d).to(device)
copula_mean_2 =  multiplier * torch.ones(d).to(device)


cov = (0.5 * torch.ones(d,d) + 0.5 * torch.eye(d)).to(device)

    
data_gen = data_generator()
LDPclient = client()

print(f"{method_name}, alpha={privacy_level}, sample size={sample_size}")
print("#########################################")
p_value_vec = np.zeros([n_test, 1])
statistic_vec = np.zeros([n_test, 1])
t = time.time()
       
for i in range(n_test):
    test_num = i + test_start
    t_start_i = time.time()
    torch.manual_seed(test_num)
    server_private.load_private_data_multinomial(
        LDPclient.release_private_conti(
            priv_mech,
            data_gen.generate_copula_gaussian_data(sample_size, copula_mean_1, cov),
            privacy_level,
            n_bin,
            device
        ),
        LDPclient.release_private_conti(
            priv_mech,
            data_gen.generate_copula_gaussian_data(sample_size, copula_mean_2, cov),
            privacy_level,
            n_bin,
            device
        ),
    k,
    device,
    device
    )
            
    time_now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

    p_value_vec[i], statistic_vec[i] = server_private.release_p_value_permutation(n_permutation)
    t_end_i = time.time() - t_start_i
    data_entry = (test_num, d, 0, privacy_level, sample_size, statistic, priv_mech, statistic_vec[i].item(), p_value_vec[i].item(), float(t_end_i), time_now, n_bin)
    print(data_entry)
    
    print(f"pval: {p_value_vec[i]} -- {test_num}th test, time elapsed {t_end_i} -- emperical power so far (from test_start): {(p_value_vec[0:(i+1)] < significance_level).mean()}")
   
    #insert into database
    try:
        insert_data(data_entry, db_dir)
    except:
        try:
            insert_data(data_entry, db_dir)
        except:    
            print("db insert fail")
    server_private.delete_data()

elapsed = time.time() - t
print(elapsed)






    

    
    
    


