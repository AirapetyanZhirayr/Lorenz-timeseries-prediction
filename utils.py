import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from scipy.special import gamma as Г
from sklearn.neighbors import KDTree
k=11
import torch
import torch.nn as nn
def generate_patterns(betta=None, L=4, K_max=10):
    '''
    :param betta: доля случайно возвращаеммых паттернов длины L-1
    :param L-1: длина паттернов
    :param K_max: максимальное расстояние между зубцами паттерна
    :return: patterns: все паттерны длины L-1 для всевозможных расстояний до K_max
             patterns_depth: ширина окна для каждого паттерна
    '''
    # k_range: возможные расстояния между зубцами
    k_range = np.arange(1, K_max+1)
    # pos_k_range: возможные шаги между зубцами для каждой позиции
    pos_k_range = [k_range]*(L-1)
    patterns = np.array([*map(lambda x: x.flatten(), np.meshgrid(*pos_k_range))]).T
    if betta:
        idx = np.arange(len(patterns))
        # sliced_idx: betta*100% случайных индексов
        sliced_idx = np.sort(np.random.choice(idx,size = int(betta*len(idx)), replace=False))
        patterns = patterns[sliced_idx]

    patterns_depth = np.sum(patterns, axis=1)+1

    return patterns, patterns_depth

def MAPE(y_pred, y_true):
    '''
    :param y_pred: спрогнозированные значения
    :param y_true: действительные значения
    :return: средняя абсолютная ошибка в процентах
    '''
    mask = ~np.isnan(y_pred)
    return np.mean(np.abs( (y_true[mask] - y_pred[mask])/y_true[mask] )) * 100


def RMSE(y_pred, y_true):
    '''
    :param y_pred: спрогнозированные значения
    :param y_true: действительные значения
    :return: корень из среднеквадратичной ошибки MSE
    '''
    mask = ~np.isnan(y_pred)
    return np.sqrt(np.mean((y_pred[mask] - y_true[mask])**2))


def get_truncated_z(truncated_y, truncated_pattern):
    '''
    :param truncated_z: участок ряда, где к нулевому элементу
    нужно приложить данный паттерн(шаблон)
    :param truncated_pattern: данный паттерн(шаблон)
    :return: обрезанный по данному паттерну z-вектор,
    где следующую точку необрезанного z-вектора нужно предсказать
    '''
    # idx: индексы обрезанного z-вектора по данному паттерну
    idx = np.concatenate(([0], truncated_pattern.cumsum()))
    if np.isnan(truncated_y[idx]).any():
        return np.nan
    return truncated_y[idx]


def predict_exemplar(truncated_z, motifs, threshold):
    '''
    :param truncated_z: обрезанный по некоторому паттерну z-вектор,
    где след. точку соотв. необрезанного z-вектора нужно предсказать
    :param motifs: мотивы некоторого паттерна
    :param threshold: порог для расстояний между мотивами и z-вектором,
    отделяющий 'адекватные' прогнозы от 'неадекватных'
    :return: pattern_predicts: прогнозы некоторо паттерна(шаблона)
             pattern_dists: расстояния между обрезанным z-вектором и
             обрезанными мотивами некоторого паттерна, меньшие чем
             заданный порого threshold
    '''
    # truncated_motifs: обрезанные мотивы некоторого паттерна
    truncated_motifs = motifs[:, :-1]
    # all_distances: расстояния от z-вектора до всех обрезанных мотивов
    all_distances = np.sqrt(np.sum(((truncated_motifs - truncated_z)**2),axis=1))
    # dist_mask: бит-маска для расстояний, меньших заданного порого threshold
    dist_mask = (all_distances<threshold)
    pattern_predicts = motifs[dist_mask][:,-1]
    pattern_dists = all_distances[dist_mask]

    return pattern_predicts, pattern_dists


def unify_predicts(possible_predicts, dists):
    '''
    :param possible_predicts:
    :param dists:
    :return:
    '''
    if len(possible_predicts)==0:
        return np.nan

    clusterer = DBSCAN(eps=0.04)
    clusterer.fit(possible_predicts[:,None])
    main_idx = clusterer.core_sample_indices_
    unified_predict = np.mean(possible_predicts[main_idx])
    return unified_predict


def transform(x, k, template):
    # start = time.time()
    n, N, template_diam, z = get_z(x, template)
    #     #=========
    #     z = (z - np.min(z, axis=1)[:,None])/(np.max(z, axis=1) - np.min(z, axis=1))[:,None]
    #     #=========
    d_k, ind = get_d_k(z, k)
    sorted_idx = np.argsort(d_k)
    z = z[sorted_idx]
    d_k, ind = get_d_k(z, k)
    p = get_p(d_k, n, N)
    ind_mask = ind < np.arange(N)[:, None]
    neighbours = [ind[i][ind_mask[i]] for i in range(len(ind))]
    #     neighbours = [set(ind[i][ind_mask[i]]) for i in range(len(ind))]

    #     print(f'Время исполнения: {time.time() - start :.2f}')
    return N, z, neighbours, p, template_diam, ind


def get_d_k(z, k):
    tree = KDTree(z)
    dist, neighbours = tree.query(z, k+1)
    # срезаем себя как соседа
    # возвращаем расстояние то k-го соседа
    return dist[:,-1], neighbours[:,1:]

def get_p(d_k, n, N):
    p = k/(V(d_k, n)*N)
#     p = (p - p.min())/(p.max()-p.min())
#     p = p/p.sum()
    return p

def V(d_k, n):
    C_n =(np.pi**(n/2))/Г((n/2)+1)
    return C_n*(d_k**n)

def get_z(x, template):
    n = len(template)+1
    template_diam = template.sum()+ 1
#     print('template_diam', template_diam)
    N = len(x) - template_diam + 1
    z = np.empty((N, n))
    start_idx = np.concatenate(([0], (template).cumsum()))
    for i in range(N):
        idx = start_idx + i
        z[i] = x[idx]
    return n, N, template_diam, z


def get_cendtroids(z, labels):
    classes = np.unique(labels)
    classes = classes[classes != 0]
    z_centroids = np.zeros(shape=(len(classes), len(z[0])))
    for i in range(len(classes)):
        _cl = classes[i]
        z_centroids[i] = np.mean(z[labels == _cl], axis=0)

    return z_centroids



def forced_predict(h_max, test_size, depth_max, patterns, patterns_depth, y_test, patterns_motifs):
    all_y_pred = []
    rmse_res = np.zeros(shape=(h_max,))
    mape_res = np.zeros(shape=(h_max,))
    n_NP_res = np.zeros(shape=(h_max,))
    h_range = np.arange(1, h_max + 1)

    for l in tqdm(range(len(h_range))):
        h = h_range[l]
        n_NP = 0
        y_pred = 1000 * np.ones(shape=(test_size,))
        for i in range(test_size):
            available_y = y_test[:(i + h_max + depth_max - h - 1)]
            for inter_i in range(i + h_max + depth_max - h - 1, i + h_max + depth_max - 1):
                possible_predicts = np.array([])
                dists = np.array([])
                for j in range(len(patterns)):
                    pattern = patterns[j]
                    depth = patterns_depth[j]

                    if inter_i - depth + 1 < 0:
                        print('something_wrong')
                    motifs = patterns_motifs[j]
                    truncated_pattern = pattern[:-1]
                    truncated_z = get_truncated_z(available_y[inter_i - depth + 1:inter_i], truncated_pattern)

                    if not np.isnan(truncated_z).any():
                        pattern_predicts, pattern_dists = predict_exemplar(truncated_z, motifs, threshold=0.05)
                        possible_predicts = np.append(possible_predicts, pattern_predicts)
                        dists = np.append(dists, pattern_dists)

                    weights = 1 / dists; weights = weights / weights.sum()
                    unified_predict = np.sum(possible_predicts * weights)
                    available_y = np.append(available_y, unified_predict)
            y_pred[i] = available_y[-1]
        all_y_pred.append(y_pred)
        rmse_res[h - 1] = RMSE(y_pred, y_test[h_max + depth_max - 2:])
        mape_res[h - 1] = MAPE(y_pred, y_test[h_max + depth_max - 2:])
        n_NP_res[h - 1] = n_NP
    forced = dict()
    forced['rmse'] = rmse_res
    forced['mape'] = mape_res
    forced['non_pred'] = n_NP_res
    forced['h_step'] = all_y_pred

    return forced


def daemon_predict(predict_threshold, h_max, test_size, depth_max, patterns, patterns_depth, y_test, patterns_motifs):
    X = []
    y = []
    h_range = np.arange(1, h_max + 1)
    for l in tqdm(range(len(h_range))):
        h = h_range[l]
        y_pred = 1000 * np.ones(shape=(test_size,))
        for i in range(test_size):
            available_y = y_test[:(i + h_max + depth_max - h - 1)]
            for inter_i in range(i + h_max + depth_max - h - 1, i + h_max + depth_max - 1):
                possible_predicts = np.array([])
                dists = np.array([])
                for j in range(len(patterns)):
                    pattern = patterns[j]
                    depth = patterns_depth[j]
                    if inter_i - depth + 1 < 0:
                        print('something_wrong')
                    motifs = patterns_motifs[j]
                    truncated_pattern = pattern[:-1]
                    truncated_z = get_truncated_z(available_y[inter_i - depth + 1:inter_i], truncated_pattern)

                    if not np.isnan(truncated_z).any():
                        pattern_predicts, pattern_dists = predict_exemplar(truncated_z, motifs, threshold=0.05)
                        possible_predicts = np.append(possible_predicts, pattern_predicts)
                        dists = np.append(dists, pattern_dists)
                weights = 1 / dists;  weights = weights / weights.sum();
                unified_predict = np.sum(possible_predicts * weights)
                X.append(possible_predicts)

                if np.abs(unified_predict - y_test[inter_i]) > predict_threshold:
                    unified_predict = np.nan
                    y.append(1)
                else:
                    y.append(0)

                available_y = np.append(available_y, unified_predict)
            y_pred[i] = available_y[-1]

    return X, y

def predict_with_np(max_predict_length, model, h_max, test_size, depth_max, patterns, patterns_depth, y_test, patterns_motifs):
    all_y_pred = []
    rmse_res = np.zeros(shape=(h_max,))
    mape_res = np.zeros(shape=(h_max,))
    n_NP_res = np.zeros(shape=(h_max,))
    h_range = np.arange(1, h_max + 1)
    for l in tqdm(range(len(h_range))):
        h = h_range[l]
        n_NP = 0
        y_pred = 1000 * np.ones(shape=(test_size,))
        for i in range(test_size):
            available_y = y_test[:(i + h_max + depth_max - h - 1)]
            for inter_i in range(i + h_max + depth_max - h - 1, i + h_max + depth_max - 1):
                possible_predicts = np.array([])
                dists = np.array([])
                for j in range(len(patterns)):
                    pattern = patterns[j]
                    depth = patterns_depth[j]
                    if inter_i - depth + 1 < 0:
                        print('something_wrong')
                    motifs = patterns_motifs[j]
                    truncated_pattern = pattern[:-1]
                    truncated_z = get_truncated_z(available_y[inter_i - depth + 1:inter_i], truncated_pattern)

                    if not np.isnan(truncated_z).any():
                        pattern_predicts, pattern_dists = predict_exemplar(truncated_z, motifs, threshold=0.05)
                        possible_predicts = np.append(possible_predicts, pattern_predicts)
                        dists = np.append(dists, pattern_dists)


                possible_predicts = possible_predicts[:max_predict_length]
                dists = dists[:max_predict_length]
                el = np.concatenate((np.zeros(max_predict_length - len(possible_predicts)), possible_predicts))
                # el = torch.FloatTensor(scaler.fit_transform(el[:, None]).T)
                el = torch.FloatTensor(el)
                model.eval()
                with torch.no_grad():
                    non_predictable = bool(int(torch.round(torch.sigmoid(model(el))).numpy().squeeze()))

                weights = 1 / dists;
                weights = weights / weights.sum()
                unified_predict = np.sum(possible_predicts * weights)

                if non_predictable:
                    unified_predict = np.nan

                available_y = np.append(available_y, unified_predict)
            if np.isnan(available_y[-1]):
                n_NP += 1
            y_pred[i] = available_y[-1]

        all_y_pred.append(y_pred)
        rmse_res[h - 1] = RMSE(y_pred, y_test[h_max + depth_max - 2:])
        mape_res[h - 1] = MAPE(y_pred, y_test[h_max + depth_max - 2:])
        n_NP_res[h - 1] = n_NP

    with_np_points = dict()
    with_np_points['rmse'] = rmse_res
    with_np_points['mape'] = mape_res
    with_np_points['non_pred'] = n_NP_res
    with_np_points['h_step'] = all_y_pred
    return with_np_points






