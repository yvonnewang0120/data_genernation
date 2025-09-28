import numpy as np
import random
from data_utils import text_to_matrix
from common_utils import strToSuffix, setup_seed
import os
import re

random.seed(42)

def pack_st_data_from_config(data_source, test_data):

    data_list = []
    for data_name in test_data:
        data_path = f'./data/{data_source}/st_{data_name}'
        data_list.append((load_st_data_from_files(data_path), data_name))
    return data_list


def load_st_data_from_files(directory):
    """
        load all files within the specified directory save吧？
    :param directory: the directory of files
    :return: a list of data (matrix form) in the directory
    """
    if not os.path.exists(directory):
        return [], []

    dataset_setup = []
    for root, dirs, files in os.walk(directory):
        # sort files by index
        files.sort(key=lambda s: int(re.findall("\d+", s)[0]))
        files.sort(key=lambda s: int(re.findall("\d+", s)[-1]))
        for f in files:
            g = open(os.path.join(root, f), 'r').readlines()
            mat = np.loadtxt(g, dtype=int)
            dataset_setup.append(mat)
    return dataset_setup


def _sample_coeff(shape, family="mixU", rng=None, **kw):

    if family == "mixU":
        # 0.9·U(0.05,0.2)+0.1·U(0.8,1.2)
        w   = kw.get("w", 0.9)
        l1, u1 = kw.get("l1", 0.05), kw.get("u1", 0.2)
        # l1, u1 = kw['dist_kwargs'].get("l1"), kw['dist_kwargs'].get("u1")
        l2, u2 = kw.get("l2", 0.8),  kw.get("u2", 1.2)
        # l2, u2 = kw['dist_kwargs'].get("l2"), kw['dist_kwargs'].get("u2")
        ave = (l2 + u2)/2
        mask = rng.random(shape) < w
        a = np.empty(shape, dtype=float)
        a[mask]  = rng.uniform(l1, u1, size=mask.sum())
        a[~mask] = rng.uniform(l2, u2, size=(~mask).sum())
        return a, ave

def generate_setup_time_matrix(
    processing_time_matrix,
    seed=2025,
    family="mixU",
    p_ref_mode="mean",     # 'mean'|'median'|'fixed'
    p_ref_fixed=None,
    dummy_mode="same",     # 'same' | 'zero'
    **dist_kwargs
):
    rng = np.random.default_rng(seed)
    P = np.asarray(processing_time_matrix, dtype=float)   # [N, M]
    N = P.shape[0]

    # 参考加工时长 \bar p
    positive = P[P > 0]
    if positive.size == 0:
        raise ValueError("processing_time_matrix is empty")
    if p_ref_mode == "mean":
        p_ref = float(positive.mean())

    a, w = _sample_coeff((N, N), family=family, rng=rng, **dist_kwargs)
    S = np.rint(a * p_ref).astype(int)
    np.fill_diagonal(S, 0)

    if dummy_mode == "same":
        a0, w = _sample_coeff((N,), family=family, rng=rng, **dist_kwargs)
        dummy = np.rint(a0 * p_ref).astype(int)
    elif dummy_mode == "zero":
        dummy = np.zeros((N,), dtype=int)
    else:
        raise ValueError("dummy_mode must be 'same' or 'zero'。")

    # 组合成 [N+1, N]
    setup_matrix = np.vstack([dummy[None, :], S])
    return setup_matrix, w

def generate_setup_table(op_pt, A, B, seed=24, as_int=True):

    op_pt_flat = np.asarray(op_pt, dtype=float)
    total_rows, M = op_pt_flat.shape
    assert total_rows % M == 0
    N = total_rows // M
    total_ops = M * N

    P = op_pt_flat[:N, :].copy()
    setup = np.zeros((total_ops + 1, total_ops), dtype=float)
    rng = np.random.default_rng(seed)

    for m in range(M):

        p_m = P[:, m]  # shape (N,)
        base = np.minimum(p_m[:, None], p_m[None, :])
        a = rng.uniform(A, B, size=(N, N))
        S_block = a * base  # (N, N)
        row_start = 1 + m * N
        col_start = m * N
        setup[row_start:row_start + N, col_start:col_start + N] = S_block

    if as_int:
        setup = np.rint(setup).astype(int)

    return setup

def save_st_data_from_files(directory, data_source, n_j, n_m, data_type, plant_depend_st):
    """
        load all files within the specified directory save吧？
    :param directory: the directory of files
    :return: a list of data (matrix form) in the directory
    """
    if not os.path.exists(directory):
        return [], []

    dataset_job_length = []
    dataset_op_pt = []
    for root, dirs, files in os.walk(directory):
        # sort files by index
        files.sort(key=lambda s: int(re.findall("\d+", s)[0]))
        files.sort(key=lambda s: int(re.findall("\d+", s)[-1]))
        for f in files:
            # print(f)
            g = open(os.path.join(root, f), 'r').readlines()
            job_length, op_pt = text_to_matrix(g)
            dataset_job_length.append(job_length)
            dataset_op_pt.append(op_pt)

            if plant_depend_st:
                setup_time_matrix = generate_setup_table(op_pt, 0.1, 0.5)
            else:
                setup_time_matrix = generate_setup_time_matrix(op_pt)

            if data_type == 'test':
                dirs = f'./data/{data_source}/st_{n_j}x{n_m}'
                if not os.path.exists(dirs):
                    os.makedirs(dirs)
                np.savetxt(f'./data/{data_source}/st_{n_j}x{n_m}/st_' + f, setup_time_matrix[0], fmt='%d')
            elif data_type == 'vali':
                dirs = f'./data/data_train_vali/{data_source}/st_{n_j}x{n_m}'
                if not os.path.exists(dirs):
                    os.makedirs(dirs)
                np.savetxt(f'./data/data_train_vali/{data_source}/st_{n_j}x{n_m}/st_' + f, setup_time_matrix,fmt='%d')

if __name__ == '__main__':
    data_source = 'SD2'
    n_j = 10
    n_m = 5
    data_suffix = 'mix'
    plant_depend_st = False
    data_type = 'test'

    if data_source == 'SD1':
        data_name = f'{n_j}x{n_m}'
    elif data_source == 'SD2':
        data_name = f'{n_j}x{n_m}{strToSuffix(data_suffix)}'
    test_data_path = f'./data/{data_source}/{data_name}'

    save_st_data_from_files(test_data_path, data_source, n_j, n_m, data_type, plant_depend_st)