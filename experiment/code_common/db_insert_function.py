
def insert_data(data_entry, db_dir):
    sleep(randint(1,20))
    con = sqlite3.connect(db_dir)
    cursor_db = con.cursor()
    cursor_db.execute(
                f"INSERT INTO {table_name}(rep, dim, bump, priv_lev, sample_size, statistic, mechanism, statistic_val, p_val, compute_time, jobdate, n_bin, permutation) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", data_entry
            )
    cursor_db.close()
    con.commit()
    con.close()
    print("db insert success")


