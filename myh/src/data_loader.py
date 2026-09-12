"""数据读取：四份附件统一成 numpy 数组，0 基时间索引。"""
import numpy as np
import pandas as pd

import myh.src.config as config


def _load_attach1():
    df = pd.read_excel(config.ATTACH1, header=None)
    data = df.iloc[1:145, 1:4].to_numpy(dtype=float)
    return data[:, 0], data[:, 1], data[:, 2]   # 电价, 负载, 光伏预测功率


def _load_attach2():
    load = pd.read_excel(config.ATTACH2, header=None, sheet_name=0)
    pv = pd.read_excel(config.ATTACH2, header=None, sheet_name=1)
    return (load.iloc[1:, 1:145].to_numpy(dtype=float),
            pv.iloc[1:, 1:145].to_numpy(dtype=float))


def _load_attach3():
    df = pd.read_excel(config.ATTACH3, header=None)
    df[0] = df[0].ffill()
    vals = df.iloc[1:, 2:26].to_numpy(dtype=float)
    return vals.reshape(config.N_DAYS, 4, 24)


def _load_attach4():
    df = pd.read_excel(config.ATTACH4, header=None)
    return df.iloc[1:, 1:145].to_numpy(dtype=float)


class Data:
    def __init__(self):
        self.price_fixed, self.load_mean, self.pv_typical = _load_attach1()
        self.load_actual, self.pv_actual = _load_attach2()
        self.pv_forecast = _load_attach3()
        self.price_varying = _load_attach4()


_DATA = None


def get_data():
    global _DATA
    if _DATA is None:
        _DATA = Data()
    return _DATA
