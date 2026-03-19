import os
from glob import glob
from threading import Thread
from multiprocessing import Process
import numpy as np
import pandas as pd
from tqdm import tqdm
from texturetools.io.mesh_header_loader import parse_mesh_info


SAVE_DIR = "/mnt/jfs/dataset/kujiale_object/check_gltf"
CHECK_LIST = ['path', 'V', 'F', 'NC', 'NM']
os.makedirs(SAVE_DIR, exist_ok=True)


def check_one(input_path):
    return parse_mesh_info(input_path)

def check_batch(input_path_list, pid:int, tid:int, batch_size=128):
    idx_list = list(range(0, len(input_path_list), batch_size))
    for bid, idx in tqdm(enumerate(idx_list), total=len(idx_list)):
        save_list = []
        i_list = list(range(idx, min(idx+batch_size, len(input_path_list))))
        for i in tqdm(i_list, total=len(i_list)):
            path = input_path_list[i]
            item = check_one(path)
            item.update({'path': os.path.join(*(path.split('/')[-3:]))})
            save_list.append(item)
        save_dict = {k: [] for k in CHECK_LIST}
        for item in save_list:
            for k in CHECK_LIST:
                save_dict[k].append(item[k])
        df = pd.DataFrame(save_dict)
        save_path = os.path.join(SAVE_DIR, f'pid-{pid:04d}-tid-{tid:04d}-bid-{bid:06d}.csv')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, escapechar=';', header=CHECK_LIST, index=False)

def check_mt(input_path_list_list, pid:int):
    thread_group = []
    for tid in range(len(input_path_list_list)):
        thread = Thread(
            target=check_batch, 
            args=(input_path_list_list[tid], pid, tid),
        )
        thread.start()
        thread_group.append(thread)
    for thread in thread_group:
        thread.join()

def check_mp(input_path_list_list_list):
    process_group = []
    for pid in range(len(input_path_list_list_list)):
        process = Process(
            target=check_mt, 
            args=(input_path_list_list_list[pid], pid),
        )
        process.start()
        process_group.append(process)
    for process in process_group:
        process.join()

def merge_csv():
    input_path = os.path.join(SAVE_DIR, f'pid-*-tid-*-bid-*.csv')
    input_path_list = glob(input_path)
    df_list = []
    for input_path in tqdm(input_path_list):
        df = pd.read_csv(input_path, escapechar=';', header=0, names=CHECK_LIST)
        df_list.append(df)
    df = pd.concat(df_list)
    save_path = os.path.join(SAVE_DIR, f'merged.csv')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, escapechar=';', header=CHECK_LIST, index=False)


if __name__ == '__main__':
    path_list_list_list = [
        np.array_split(
            np.array(
                glob(f'/mnt/jfs/dataset/kujiale_object/part_{i}_decompressed/*.gltf')
            ), 
            8,
        ) for i in range(9)
    ]
    # check_batch(path_list_list_list[0][0], -1, -1)  # NOTE: for debug
    check_mp(path_list_list_list)
    merge_csv()

