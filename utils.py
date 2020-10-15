import numpy as np
from scipy import sparse
from scipy.stats import ttest_rel


def load_data(data_path, delimiter):
    raw_data = np.loadtxt(data_path, dtype=np.float, delimiter=delimiter, usecols=[0, 1, 2])
    users = list(set(raw_data[:, 0].astype(np.int)))
    users.sort()
    user_dict = {k: i for i, k in enumerate(users)}
    items = list(set(raw_data[:, 1].astype(np.int)))
    items.sort()
    item_dict = {k: i for i, k in enumerate(items)}
    for i in range(len(raw_data)):
        raw_data[i, 0] = user_dict[raw_data[i, 0]]
        raw_data[i, 1] = item_dict[raw_data[i, 1]]
    return raw_data


def build_user_item_matrix(ratings, n_user, n_item):
    data = ratings[:, 2]
    row_index = ratings[:, 0]
    col_index = ratings[:, 1]
    shape = (n_user, n_item)
    return sparse.csr_matrix((data, (row_index, col_index)), shape=shape)


def RMSE(estimation, truth):
    truth_coo = truth.tocoo()
    row_idx = truth_coo.row
    col_idx = truth_coo.col
    data = truth.data
    pred = np.zeros(shape=data.shape)
    for i in range(len(data)):
        pred[i] = estimation[row_idx[i], col_idx[i]]
    sse = np.sum(np.square(data - pred))
    return np.sqrt(np.divide(sse, len(data)))


def RMSE_with_ttest(estimation, old_estimation, truth):
    truth_coo = truth.tocoo()
    row_idx = truth_coo.row
    col_idx = truth_coo.col
    data = truth_coo.data
    pred_dis = np.zeros(shape=data.shape)
    old_pred_dis = np.zeros(shape=data.shape)
    for i in range(len(data)):
        pred_dis[i] = abs(estimation[row_idx[i], col_idx[i]] - data[i])
        old_pred_dis[i] = abs(old_estimation[row_idx[i], col_idx[i]] - data[i])
    _, p_value = ttest_rel(pred_dis, old_pred_dis)
    sse = np.sum(np.square(pred_dis))
    return np.sqrt(np.divide(sse, len(data))), p_value
