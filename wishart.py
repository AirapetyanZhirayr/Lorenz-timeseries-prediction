import numpy as np
import time


def wishart(N, z, p, neighbours, h, ind):
    # start = time.time()

    w = np.zeros(N)
    formed = set()
    for i in (range(N)):
        l = np.unique(w[neighbours[i]])
        #         l = np.unique([w[j] for j in range(i) if i in ind[j]])

        if len(l) == 0:  # если вершина является изолированной, присвоить к новому кластеру
            w[i] = i + 1
        elif len(l) == 1:  # если вершина связана только с одним кластером
            l = l[0]
            if l in formed:  # если связанный кластер сформирован, присвоить шум
                w[i] = 0
            else:  # иначе, присвоить номер связанного кластера
                w[i] = l
        else:  # если вершина связана больше чем с одним кластером
            l_min = np.min(l)

            if set(l).issubset(formed):  # если все эти кластеры сформированы, присвоить шум
                w[i] = 0
            else:
                # significant - бит-маска для значимости
                significant = np.array([is_significant(p[w == _l], h) for _l in l])
                # _z - число значимых кластеров
                _z = significant.sum()
                if _z > 1 or l_min == 0:  # если есть фоновый кластер или значимых больше одного
                    formed.update(l[significant])  # добавляем значимые в сформированные

                    for j in l[~significant]: w[w == j] = 0; w[i] = 0;  # незначимые и саму вершину добавляем в шум

                elif _z <= 1 and l_min > 0:  # если нет шума и значимых не больше одного
                    for j in l: w[w == j] = l_min; w[i] = l_min  # соединяем всех в 1 кластер

    #     print(f'Время исполнения: {time.time() - start :.2f}')
    return w

def is_significant(p, h):
#     return np.abs(np.max(p) - np.min(p)) >= h
    return p.ptp() >= h