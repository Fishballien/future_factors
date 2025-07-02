# -*- coding: utf-8 -*-
"""
Created on Wed Jul 02 2025
@author: Assistant

期货主买主卖量数据合并模块
将逐期货逐日的数据整理成按feature分组的parquet文件
行index: 所有天所有时间戳按顺序
列: 不同的期货品种
"""

import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import traceback

# 添加项目路径
file_path = Path(__file__).resolve()
project_dir = file_path.parents[1]
sys.path.append(str(project_dir))

from utils.timeutils import parse_time_string, get_a_share_intraday_time_series


def collect_all_timestamps(zhuli_dir, fut_list, params):
    """
    收集所有可能的时间戳，用于创建完整的时间索引
    """
    interval = params.get('interval', '1min')
    keep_periods = params.get('keep_periods', {
        'morning': ('09:31:00', '11:30:00'),
        'afternoon': ('13:01:00', '15:00:00')
    })
    
    all_timestamps = set()
    
    # 收集所有日期
    all_dates = set()
    for fut in fut_list:
        zhuli_path = zhuli_dir / f'{fut}.parquet'
        if zhuli_path.exists():
            zhuli_data = pd.read_parquet(zhuli_path)
            dates = zhuli_data['date'].astype(str).unique()
            all_dates.update(dates)
    
    # 为每个日期生成时间戳
    interval_timedelta = {'seconds': parse_time_string(interval)}
    
    for date in sorted(all_dates):
        date_in_dt = datetime.strptime(date, '%Y%m%d')
        day_timestamps = get_a_share_intraday_time_series(
            date_in_dt, 
            interval_timedelta, 
            trading_periods=keep_periods
        )
        all_timestamps.update(day_timestamps)
    
    # 转换为排序的DatetimeIndex
    full_index = pd.DatetimeIndex(sorted(all_timestamps))
    
    return full_index


def merge_feature_data(feature_name, raw_data_dir, fut_list, full_index, output_path):
    """
    合并单个feature的数据
    """
    # 初始化结果DataFrame
    result_df = pd.DataFrame(index=full_index, columns=fut_list, dtype=float)
    result_df = result_df.fillna(0.0)
    
    # 逐个期货品种处理
    for fut in tqdm(fut_list, desc=f"处理 {feature_name}"):
        fut_dir = raw_data_dir / fut
        
        if not fut_dir.exists():
            continue
        
        # 收集该期货的所有数据文件
        parquet_files = list(fut_dir.glob('*.parquet'))
        
        if not parquet_files:
            continue
        
        # 读取并合并该期货的所有日期数据
        fut_data_list = []
        
        for file_path in parquet_files:
            try:
                daily_data = pd.read_parquet(file_path)
                
                # 确保包含目标feature
                if feature_name not in daily_data.columns:
                    continue
                
                # 提取目标feature的数据
                feature_series = daily_data[feature_name].copy()
                fut_data_list.append(feature_series)
                
            except Exception as e:
                continue
        
        if fut_data_list:
            # 合并该期货的所有数据
            fut_combined = pd.concat(fut_data_list, axis=0)
            fut_combined = fut_combined.sort_index()
            
            # 去重（保留最后一个值，以防有重复时间戳）
            fut_combined = fut_combined[~fut_combined.index.duplicated(keep='last')]
            
            # 将数据对齐到完整索引
            result_df[fut] = fut_combined.reindex(full_index, fill_value=0.0)
    
    # 保存结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(output_path)


def merge_all_trade_flow_data(raw_data_dir, zhuli_dir, merged_save_dir, fut_list, params):
    """
    合并所有期货主买主卖量数据
    """
    # 收集完整的时间索引
    full_index = collect_all_timestamps(zhuli_dir, fut_list, params)
    
    # 定义要合并的features
    features = ['act_buy_amount', 'act_sell_amount']
    
    # 逐个feature进行合并
    for feature_name in features:
        output_path = merged_save_dir / f'{feature_name}.parquet'
        
        merge_feature_data(
            feature_name=feature_name,
            raw_data_dir=raw_data_dir,
            fut_list=fut_list,
            full_index=full_index,
            output_path=output_path
        )


# %% 主函数
if __name__ == '__main__':
    # 配置参数
    fut_list = ['IC', 'IF', 'IH', 'IM']
    zhuli_dir = Path('/mnt/nfs/30.132_xt_data1/future_zhuli')
    raw_data_dir = Path('/mnt/Data/xintang/future_factors/trade_flow_raw')  # trade_flow_mp.py的save_dir
    merged_save_dir = Path('/mnt/Data/xintang/future_factors/trade_flow_merged')  # 新的保存目录
    
    params = {
        'interval': '1min',
        'keep_periods': {
            'morning': ('09:31:00', '11:30:00'),
            'afternoon': ('13:01:00', '15:00:00')
        }
    }
    
    # 执行数据合并
    merge_all_trade_flow_data(
        raw_data_dir=raw_data_dir,
        zhuli_dir=zhuli_dir,
        merged_save_dir=merged_save_dir,
        fut_list=fut_list,
        params=params
    )
    
    print('期货主买主卖量数据合并完成！')