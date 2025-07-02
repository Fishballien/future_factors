# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 14:23:32 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import pandas as pd
import numpy as np


from utils.speedutils import timeit


# %%
def imb01(bid, ask):
    """
    计算 imbalance (bid - ask) / (bid + ask)，当 bid + ask == 0 时返回 NaN。
    兼容 DataFrame 和 Series 类型的输入。

    Parameters:
        bid (pd.DataFrame or pd.Series): bid 数据。
        ask (pd.DataFrame or pd.Series): ask 数据。

    Returns:
        pd.DataFrame or pd.Series: 计算后的 imbalance 数据，与输入类型一致。
    """
    # 计算 bid + ask
    sum_bid_ask = bid + ask
     
    # 使用 np.where 进行条件判断
    imbalance = np.where(
        sum_bid_ask == 0,
        np.nan,
        (bid - ask) / sum_bid_ask
    )
    
    # 根据输入类型返回对应类型的结果
    if isinstance(bid, pd.DataFrame):
        return pd.DataFrame(imbalance, index=bid.index, columns=bid.columns)
    elif isinstance(bid, pd.Series):
        return pd.Series(imbalance, index=bid.index, name=bid.name)
    else:
        raise TypeError("Inputs must be pandas DataFrame or Series.")
        
        
def imb02(bid_factor, ask_factor):
    """
    计算 imbalance 公式 abs(bid_factor - ask_factor) / (abs(bid_factor) + abs(ask_factor))，
    当 abs(bid_factor) + abs(ask_factor) == 0 时返回 NaN。
    兼容 DataFrame 和 Series 类型的输入。

    Parameters:
        bid_factor (pd.DataFrame or pd.Series): bid 数据。
        ask_factor (pd.DataFrame or pd.Series): ask 数据。

    Returns:
        pd.DataFrame or pd.Series: 计算后的 imbalance 数据，与输入类型一致。
    """
    # 计算 abs(bid_factor) + abs(ask_factor)
    abs_sum = np.abs(bid_factor) + np.abs(ask_factor)

    # 使用 np.where 进行条件判断
    imbalance = np.where(
        abs_sum == 0,
        np.nan,
        np.abs(bid_factor - ask_factor) / abs_sum
    )

    # 根据输入类型返回对应类型的结果
    if isinstance(bid_factor, pd.DataFrame):
        return pd.DataFrame(imbalance, index=bid_factor.index, columns=bid_factor.columns)
    elif isinstance(bid_factor, pd.Series):
        return pd.Series(imbalance, index=bid_factor.index, name=bid_factor.name)
    else:
        raise TypeError("Inputs must be pandas DataFrame or Series.")


def imb03(bid_factor, ask_factor):
    """
    计算 imbalance 公式 ask_factor / bid_factor，
    当 bid_factor == 0 时返回 NaN。
    兼容 DataFrame 和 Series 类型的输入。

    Parameters:
        bid_factor (pd.DataFrame or pd.Series): bid 数据。
        ask_factor (pd.DataFrame or pd.Series): ask 数据。

    Returns:
        pd.DataFrame or pd.Series: 计算后的 imbalance 数据，与输入类型一致。
    """
    # 使用 np.where 进行条件判断
    imbalance = np.where(
        bid_factor == 0,
        np.nan,
        ask_factor / bid_factor
    )

    # 根据输入类型返回对应类型的结果
    if isinstance(bid_factor, pd.DataFrame):
        return pd.DataFrame(imbalance, index=bid_factor.index, columns=bid_factor.columns)
    elif isinstance(bid_factor, pd.Series):
        return pd.Series(imbalance, index=bid_factor.index, name=bid_factor.name)
    else:
        raise TypeError("Inputs must be pandas DataFrame or Series.")
        
        
def imb04(bid_factor, ask_factor):
    """
    计算 imbalance 公式 (ask_factor - bid_factor) / max(ask_factor, bid_factor)，
    当 max(ask_factor, bid_factor) == 0 时返回 NaN。
    兼容 DataFrame 和 Series 类型的输入。

    Parameters:
        bid_factor (pd.DataFrame or pd.Series): bid 数据。
        ask_factor (pd.DataFrame or pd.Series): ask 数据。

    Returns:
        pd.DataFrame or pd.Series: 计算后的 imbalance 数据，与输入类型一致。
    """
    # 计算 max(ask_factor, bid_factor)
    max_value = np.maximum(ask_factor, bid_factor)

    # 使用 np.where 进行条件判断
    imbalance = np.where(
        max_value == 0,
        np.nan,
        (ask_factor - bid_factor) / max_value
    )

    # 根据输入类型返回对应类型的结果
    if isinstance(bid_factor, pd.DataFrame):
        return pd.DataFrame(imbalance, index=bid_factor.index, columns=bid_factor.columns)
    elif isinstance(bid_factor, pd.Series):
        return pd.Series(imbalance, index=bid_factor.index, name=bid_factor.name)
    else:
        raise TypeError("Inputs must be pandas DataFrame or Series.")


def imb05(bid_factor, ask_factor):
    """
    计算 imbalance 公式 (|ask_factor| - |bid_factor|) / (|ask_factor| + |bid_factor|)，
    当 |ask_factor| + |bid_factor| == 0 时返回 NaN。
    兼容 DataFrame 和 Series 类型的输入。

    Parameters:
        bid_factor (pd.DataFrame or pd.Series): bid 数据。
        ask_factor (pd.DataFrame or pd.Series): ask 数据。

    Returns:
        pd.DataFrame or pd.Series: 计算后的 imbalance 数据，与输入类型一致。
    """
    # 计算 |ask_factor| 和 |bid_factor| 的和
    abs_sum = np.abs(ask_factor) + np.abs(bid_factor)

    # 使用 np.where 进行条件判断
    imbalance = np.where(
        abs_sum == 0,
        np.nan,
        (np.abs(ask_factor) - np.abs(bid_factor)) / abs_sum
    )

    # 根据输入类型返回对应类型的结果
    if isinstance(bid_factor, pd.DataFrame):
        return pd.DataFrame(imbalance, index=bid_factor.index, columns=bid_factor.columns)
    elif isinstance(bid_factor, pd.Series):
        return pd.Series(imbalance, index=bid_factor.index, name=bid_factor.name)
    else:
        raise TypeError("Inputs must be pandas DataFrame or Series.")


def imb06(bid_factor, ask_factor):
    """
    计算 imbalance 公式 ask_factor - bid_factor。
    兼容 DataFrame 和 Series 类型的输入。

    Parameters:
        bid_factor (pd.DataFrame or pd.Series): bid 数据。
        ask_factor (pd.DataFrame or pd.Series): ask 数据。

    Returns:
        pd.DataFrame or pd.Series: 计算后的 imbalance 数据，与输入类型一致。
    """
    # 计算 ask_factor - bid_factor
    imbalance = ask_factor - bid_factor

    # 根据输入类型返回对应类型的结果
    if isinstance(bid_factor, pd.DataFrame):
        return pd.DataFrame(imbalance, index=bid_factor.index, columns=bid_factor.columns)
    elif isinstance(bid_factor, pd.Series):
        return pd.Series(imbalance, index=bid_factor.index, name=bid_factor.name)
    else:
        raise TypeError("Inputs must be pandas DataFrame or Series.")


def imb07(bid_factor, ask_factor):
    """
    计算 imbalance 公式 (ask_factor - bid_factor) / (|ask_factor| + |bid_factor|)，
    当 |ask_factor| + |bid_factor| == 0 时返回 NaN。
    兼容 DataFrame 和 Series 类型的输入。

    Parameters:
        bid_factor (pd.DataFrame or pd.Series): bid 数据。
        ask_factor (pd.DataFrame or pd.Series): ask 数据。

    Returns:
        pd.DataFrame or pd.Series: 计算后的 imbalance 数据，与输入类型一致。
    """
    # 计算 |ask_factor| + |bid_factor|
    abs_sum = np.abs(ask_factor) + np.abs(bid_factor)

    # 使用 np.where 进行条件判断
    imbalance = np.where(
        abs_sum == 0,
        np.nan,
        (ask_factor - bid_factor) / abs_sum
    )

    # 根据输入类型返回对应类型的结果
    if isinstance(bid_factor, pd.DataFrame):
        return pd.DataFrame(imbalance, index=bid_factor.index, columns=bid_factor.columns)
    elif isinstance(bid_factor, pd.Series):
        return pd.Series(imbalance, index=bid_factor.index, name=bid_factor.name)
    else:
        raise TypeError("Inputs must be pandas DataFrame or Series.")


def imb08(bid_factor, ask_factor):
    """
    计算 imbalance 公式 (ask_factor + bid_factor) / 2。
    兼容 DataFrame 和 Series 类型的输入。

    Parameters:
        bid_factor (pd.DataFrame or pd.Series): bid 数据。
        ask_factor (pd.DataFrame or pd.Series): ask 数据。

    Returns:
        pd.DataFrame or pd.Series: 计算后的 imbalance 数据，与输入类型一致。
    """
    # 计算 (ask_factor + bid_factor) / 2
    imbalance = (ask_factor + bid_factor) / 2

    # 根据输入类型返回对应类型的结果
    if isinstance(bid_factor, pd.DataFrame):
        return pd.DataFrame(imbalance, index=bid_factor.index, columns=bid_factor.columns)
    elif isinstance(bid_factor, pd.Series):
        return pd.Series(imbalance, index=bid_factor.index, name=bid_factor.name)
    else:
        raise TypeError("Inputs must be pandas DataFrame or Series.")
        
        
def imb09(numer_bid, numer_ask, denom_bid, denom_ask):
    """
    计算子集imbalance公式：(numer_bid - numer_ask) / (denom_bid + denom_ask)
    
    其中：
    - 分子：使用自定义权重聚合的bid和ask的差值（可以是任何你筛选的股票子集）
    - 分母：使用基准权重聚合的bid和ask的总和
    
    当 denom_bid + denom_ask == 0 时返回 NaN。
    兼容 DataFrame 和 Series 类型的输入。

    Parameters:
        numer_bid (pd.DataFrame or pd.Series): 自定义权重聚合的bid数据
        numer_ask (pd.DataFrame or pd.Series): 自定义权重聚合的ask数据  
        denom_bid (pd.DataFrame or pd.Series): 基准权重聚合的bid数据
        denom_ask (pd.DataFrame or pd.Series): 基准权重聚合的ask数据

    Returns:
        pd.DataFrame or pd.Series: 计算后的子集imbalance数据，与输入类型一致
    """
    # 计算分子：自定义权重的bid和ask差值
    numerator = numer_bid - numer_ask
    
    # 计算分母：基准权重的bid和ask总和
    denominator = denom_bid + denom_ask
     
    # 使用 np.where 进行条件判断
    imbalance = np.where(
        denominator == 0,
        np.nan,
        numerator / denominator
    )
    
    # 根据输入类型返回对应类型的结果
    if isinstance(numer_bid, pd.DataFrame):
        return pd.DataFrame(imbalance, index=numer_bid.index, columns=numer_bid.columns)
    elif isinstance(numer_bid, pd.Series):
        return pd.Series(imbalance, index=numer_bid.index, name=numer_bid.name)
    else:
        raise TypeError("Inputs must be pandas DataFrame or Series.")
        
        
def imb10(bid_factor, ask_factor):
    """
    计算 imbalance 公式 bid_factor - ask_factor（imb06的反向版本）。
    兼容 DataFrame 和 Series 类型的输入。

    Parameters:
        bid_factor (pd.DataFrame or pd.Series): bid 数据。
        ask_factor (pd.DataFrame or pd.Series): ask 数据。

    Returns:
        pd.DataFrame or pd.Series: 计算后的 imbalance 数据，与输入类型一致。
    """
    # 计算 bid_factor - ask_factor
    imbalance = bid_factor - ask_factor

    # 根据输入类型返回对应类型的结果
    if isinstance(bid_factor, pd.DataFrame):
        return pd.DataFrame(imbalance, index=bid_factor.index, columns=bid_factor.columns)
    elif isinstance(bid_factor, pd.Series):
        return pd.Series(imbalance, index=bid_factor.index, name=bid_factor.name)
    else:
        raise TypeError("Inputs must be pandas DataFrame or Series.")
        
        
def add(bid_factor, ask_factor):
    """
    计算 imbalance 公式 bid_factor - ask_factor（imb06的反向版本）。
    兼容 DataFrame 和 Series 类型的输入。

    Parameters:
        bid_factor (pd.DataFrame or pd.Series): bid 数据。
        ask_factor (pd.DataFrame or pd.Series): ask 数据。

    Returns:
        pd.DataFrame or pd.Series: 计算后的 imbalance 数据，与输入类型一致。
    """
    # 计算 bid_factor - ask_factor
    imbalance = bid_factor + ask_factor

    # 根据输入类型返回对应类型的结果
    if isinstance(bid_factor, pd.DataFrame):
        return pd.DataFrame(imbalance, index=bid_factor.index, columns=bid_factor.columns)
    elif isinstance(bid_factor, pd.Series):
        return pd.Series(imbalance, index=bid_factor.index, name=bid_factor.name)
    else:
        raise TypeError("Inputs must be pandas DataFrame or Series.")

