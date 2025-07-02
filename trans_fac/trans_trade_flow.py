# -*- coding: utf-8 -*-
"""
Created on Wed Jul 02 2025
@author: Assistant

期货主买主卖量因子计算模块
读取合并好的trade flow数据，进行平滑处理后计算imbalance因子
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import traceback

# 添加项目路径
file_path = Path(__file__).resolve()
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))

from operators.ts_intraday import intraSma, intraTEwma
from operators.fundamental import imb01, imb02, imb03, imb04, imb05, imb06, imb07, imb08, imb09, imb10, imb01_rob


def load_trade_flow_data(merged_data_dir):
    """
    加载合并好的trade flow数据
    
    Parameters:
    -----------
    merged_data_dir : Path
        合并数据的目录路径
        
    Returns:
    --------
    tuple: (act_buy_amount, act_sell_amount) DataFrame对象
    """
    print("📊 加载trade flow数据...")
    
    # 加载主买量和主卖量数据
    buy_path = merged_data_dir / 'act_buy_amount.parquet'
    sell_path = merged_data_dir / 'act_sell_amount.parquet'
    
    if not buy_path.exists():
        raise FileNotFoundError(f"未找到主买量数据文件: {buy_path}")
    
    if not sell_path.exists():
        raise FileNotFoundError(f"未找到主卖量数据文件: {sell_path}")
    
    act_buy_amount = pd.read_parquet(buy_path)
    act_sell_amount = pd.read_parquet(sell_path)
    
    print(f"✅ 数据加载完成")
    print(f"   📈 主买量数据形状: {act_buy_amount.shape}")
    print(f"   📉 主卖量数据形状: {act_sell_amount.shape}")
    print(f"   📅 时间范围: {act_buy_amount.index[0]} 至 {act_buy_amount.index[-1]}")
    print(f"   🏷️  期货品种: {list(act_buy_amount.columns)}")
    
    return act_buy_amount, act_sell_amount


def apply_smoothing(data, smooth_params):
    """
    对数据应用平滑处理
    
    Parameters:
    -----------
    data : pd.DataFrame
        输入数据
    smooth_params : dict
        平滑参数配置
        
    Returns:
    --------
    dict: 平滑后的数据字典，键为平滑方法名
    """
    smoothed_data = {}
    
    print(f"🔄 开始数据平滑处理...")
    
    # intraSma 平滑
    if 'intraSma' in smooth_params:
        print(f"   📊 应用 intraSma...")
        for window in tqdm(smooth_params['intraSma'], desc="intraSma"):
            key = f"intraSma_{window}"
            smoothed_data[key] = intraSma(data, window=window)
    
    # intraTEwma 平滑
    if 'intraTEwma' in smooth_params:
        print(f"   📈 应用 intraTEwma...")
        for config in tqdm(smooth_params['intraTEwma'], desc="intraTEwma"):
            span = config['span']
            freq = config['freq']
            key = f"intraTEwma_span{span}_freq{freq}"
            smoothed_data[key] = intraTEwma(data, span=span, freq=freq)
    
    return smoothed_data


def calculate_imbalance_factors(buy_data_dict, sell_data_dict, imb_methods):
    """
    计算imbalance因子
    
    Parameters:
    -----------
    buy_data_dict : dict
        平滑后的买量数据字典
    sell_data_dict : dict
        平滑后的卖量数据字典
    imb_methods : list
        要使用的imbalance计算方法列表
        
    Returns:
    --------
    dict: 计算出的imbalance因子字典
    """
    imb_factors = {}
    
    # 定义imbalance方法映射
    imb_method_map = {
        'imb01': imb01,
        'imb02': imb02,
        'imb03': imb03,
        'imb04': imb04,
        'imb05': imb05,
        'imb06': imb06,
        'imb07': imb07,
        'imb08': imb08,
        'imb09': imb09,
        'imb10': imb10,
        'imb01_rob': imb01_rob,
    }
    
    print(f"🧮 开始计算imbalance因子...")
    
    # 确保两个数据字典有相同的键
    common_keys = set(buy_data_dict.keys()) & set(sell_data_dict.keys())
    
    for smooth_key in tqdm(common_keys, desc="平滑方法"):
        buy_data = buy_data_dict[smooth_key]
        sell_data = sell_data_dict[smooth_key]
        
        for imb_method in imb_methods:
            if imb_method not in imb_method_map:
                print(f"⚠️  未知的imbalance方法: {imb_method}")
                continue
            
            # try:
            factor_key = f"{smooth_key}_{imb_method}"
            
            # 根据不同的imb方法调用相应函数
            if imb_method == 'imb09':
                # imb09需要4个参数，这里使用买卖数据作为分子和分母
                imb_result = imb_method_map[imb_method](buy_data, sell_data, buy_data + sell_data, buy_data + sell_data)
            else:
                # 其他方法使用2个参数
                imb_result = imb_method_map[imb_method](buy_data, sell_data)
            
            imb_factors[factor_key] = imb_result
                
            # except Exception as e:
            #     print(f"❌ 计算 {factor_key} 时出错: {str(e)}")
            #     continue
    
    return imb_factors


def save_factors(factors_dict, save_dir, prefix="trade_flow"):
    """
    保存计算出的因子
    
    Parameters:
    -----------
    factors_dict : dict
        因子数据字典
    save_dir : Path
        保存目录
    prefix : str
        文件名前缀
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 保存因子数据到 {save_dir}...")
    
    for factor_name, factor_data in tqdm(factors_dict.items(), desc="保存因子"):
        filename = f"{prefix}_{factor_name}.parquet"
        filepath = save_dir / filename
        
        try:
            factor_data.to_parquet(filepath)
        except Exception as e:
            print(f"❌ 保存 {factor_name} 时出错: {str(e)}")
            continue
    
    print(f"✅ 因子保存完成，共保存 {len(factors_dict)} 个因子")


def main(merged_data_dir, save_dir, config):
    """
    主函数
    
    Parameters:
    -----------
    merged_data_dir : Path
        合并数据目录
    save_dir : Path
        因子保存目录
    config : dict
        配置参数
    """
    # try:
    # 1. 加载数据
    act_buy_amount, act_sell_amount = load_trade_flow_data(merged_data_dir)
    
    # 2. 数据平滑
    print("\n🔄 处理主买量数据...")
    buy_smoothed = apply_smoothing(act_buy_amount, config['smooth_params'])
    
    print("\n🔄 处理主卖量数据...")
    sell_smoothed = apply_smoothing(act_sell_amount, config['smooth_params'])
    
    # 3. 计算imbalance因子
    print("\n🧮 计算imbalance因子...")
    imb_factors = calculate_imbalance_factors(
        buy_smoothed, 
        sell_smoothed, 
        config['imb_methods']
    )
    
    # 4. 保存因子
    print(f"\n💾 保存因子...")
    save_factors(imb_factors, save_dir, prefix="trade_flow")
    
    print(f"\n🎉 任务完成!")
    print(f"   📂 保存目录: {save_dir}")
    print(f"   📊 因子数量: {len(imb_factors)}")
        
    # except Exception as e:
    #     print(f"❌ 执行过程中出现错误: {str(e)}")
    #     traceback.print_exc()


# %% 主程序
if __name__ == '__main__':
    # 配置参数
    config = {
        # 平滑处理参数
        'smooth_params': {
            'intraSma': [5, 10, 15, 30, 60],  # 不同的滑动窗口大小
            'intraTEwma': [
                {'span': 10, 'freq': '1min'},
                {'span': 20, 'freq': '1min'},
                {'span': 30, 'freq': '1min'},
                {'span': 60, 'freq': '1min'},
                {'span': 120, 'freq': '1min'},
            ]
        },
        
        # imbalance计算方法
        'imb_methods': ['imb01']
    }
    
    # 路径配置
    merged_data_dir = Path('/mnt/Data/xintang/future_factors/trade_flow_merged')
    save_dir = Path('/mnt/Data/xintang/index_factors/trade_flow/v0')
    
    # 执行主函数
    main(merged_data_dir, save_dir, config)