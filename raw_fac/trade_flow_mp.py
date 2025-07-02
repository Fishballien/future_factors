# -*- coding: utf-8 -*-
"""
Created on Mon May 12 13:43:24 2025
@author: Xintang Zheng

期货主买主卖量计算模块 - 并行版本
仿照TWAP计算逻辑，计算每个期货品种的主买主卖金额
使用concurrent.futures实现并行计算

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝
"""

# %% imports
import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
from functools import partial
from tqdm import tqdm
import traceback
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from itertools import product
import time

# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[0]
sys.path.append(str(project_dir))

# %%
from utils.timeutils import parse_time_string, get_a_share_intraday_time_series


# %% 计算单个期货单日主买主卖量的函数
def calc_order_flow_per_fut_per_day(date, data_all, instru_id, interval='1min', keep_periods=None):
    """
    计算单个期货品种单日的主买主卖金额
    
    参数:
    date (str): 日期，格式为'YYYYMMDD'
    data_all (pd.DataFrame): 包含所有期货数据的DataFrame
    instru_id (str): 目标合约的InstruID，如'IC2401'
    interval (str): 聚合间隔，如'1min'
    keep_periods (dict): 保留的交易时段，如{'morning': ('09:31:00', '11:30:00'), 'afternoon': ('13:01:00', '15:00:00')}
    
    返回:
    pd.DataFrame: 包含主买主卖金额的DataFrame
    """
    if keep_periods is None:
        keep_periods = {
            'morning': ('09:31:00', '11:30:00'),
            'afternoon': ('13:01:00', '15:00:00')
        }
    
    date_in_dt = datetime.strptime(date, '%Y%m%d')
    interval_timedelta = {'seconds': parse_time_string(interval)}
    keep_ts = get_a_share_intraday_time_series(date_in_dt, interval_timedelta, 
                                               trading_periods=keep_periods)
    
    # 初始化结果DataFrame
    res = pd.DataFrame(index=keep_ts, columns=['act_buy_amount', 'act_sell_amount'])
    res['act_buy_amount'] = 0.0
    res['act_sell_amount'] = 0.0
    
    try:
        # 筛选指定合约的数据
        data = data_all[data_all['InstruID'] == instru_id].copy()
        
        if data.empty:
            print(f'No data found for {instru_id} on {date}')
            return res
        
        # 计算VWAP和中间价
        data['vwap'] = data['Turnover'] / data['Volume'] / 200
        data['midprice'] = (data['BidPrice1'] + data['AskPrice1']) / 2
        data['midprice_diff'] = data['midprice'].diff()
        data['vwap_lastmpc_diff'] = (data['vwap'] - data['midprice']).shift(1)
        
        # 计算交易方向
        data['trade_direction'] = 0
        
        # 第一优先级：midprice变化方向
        data.loc[data['midprice_diff'] > 0, 'trade_direction'] = 1   # midprice上升，主买
        data.loc[data['midprice_diff'] < 0, 'trade_direction'] = -1  # midprice下降，主卖
        
        # 第二优先级：midprice不变时，比较vwap和midprice
        midprice_unchanged = (data['midprice_diff'] == 0) | data['midprice_diff'].isna()
        data.loc[midprice_unchanged & (data['vwap'] > data['midprice']), 'trade_direction'] = 1   # vwap > midprice，主买
        data.loc[midprice_unchanged & (data['vwap'] < data['midprice']), 'trade_direction'] = -1  # vwap < midprice，主卖
        
        # 第三优先级：vwap == midprice时，延续上一tick方向
        vwap_eq_midprice = midprice_unchanged & (data['vwap'] == data['midprice'])
        data.loc[vwap_eq_midprice, 'trade_direction'] = data['trade_direction'].shift(1).fillna(0)
        
        # 计算每个tick的主买和主卖金额
        data['act_buy_amount'] = 0.0
        data['act_sell_amount'] = 0.0
        data.loc[data['trade_direction'] == 1, 'act_buy_amount'] = data.loc[data['trade_direction'] == 1, 'Turnover']
        data.loc[data['trade_direction'] == -1, 'act_sell_amount'] = data.loc[data['trade_direction'] == -1, 'Turnover']
        
        # 创建时间索引
        data['DateTime'] = pd.to_datetime(data['TradDay'].astype(str) + ' ' + data['UpdateTime'].astype(str))
        data.set_index('DateTime', inplace=True)
        
        # 按指定间隔聚合
        minute_data = data.resample(interval).agg({
            'act_buy_amount': 'sum',
            'act_sell_amount': 'sum'
        })
        
        # 重新索引到目标时间序列
        output = minute_data.reindex(index=keep_ts)
        output = output.fillna(0.0)  # 将NaN填充为0
        
        res.loc[:, 'act_buy_amount'] = output['act_buy_amount']
        res.loc[:, 'act_sell_amount'] = output['act_sell_amount']
        
    except Exception as e:
        traceback.print_exc()
        print(f'Error processing {instru_id} on {date}: {str(e)}')
        if date > '20250101':
            raise Exception(f'Processing error: {date}, {instru_id}')
    
    return res


# %% 处理单个任务的函数
def process_single_task(task_params):
    """
    处理单个任务的函数，用于并行计算
    
    参数:
    task_params (tuple): 包含任务参数的元组
    
    返回:
    dict: 任务结果
    """
    fut, date, curr_trade, data_base_path, save_dir, interval, keep_periods, use_cache = task_params
    
    instru_id = f'{fut}{curr_trade}'
    fut_save_dir = save_dir / fut
    cache_path = fut_save_dir / f'{date}.parquet'
    
    # 检查是否已有缓存文件
    if use_cache and cache_path.exists():
        return {
            'fut': fut,
            'date': date,
            'status': 'cached',
            'message': f'Cache exists for {instru_id} on {date}'
        }
    
    try:
        # 构建数据文件路径
        data_path = f'{data_base_path}/{date}/mdl_21_1_0.csv'
        
        # 读取当日数据
        try:
            data_all = pd.read_csv(data_path)
        except Exception as e:
            return {
                'fut': fut,
                'date': date,
                'status': 'error',
                'message': f'无法读取数据文件 {data_path}: {str(e)}'
            }
        
        # 计算当日主买主卖量
        result = calc_order_flow_per_fut_per_day(
            date=date,
            data_all=data_all,
            instru_id=instru_id,
            interval=interval,
            keep_periods=keep_periods
        )
        
        # 确保保存目录存在
        fut_save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存结果
        result.to_parquet(cache_path)
        
        return {
            'fut': fut,
            'date': date,
            'status': 'success',
            'message': f'Successfully processed {instru_id} on {date}'
        }
        
    except Exception as e:
        error_msg = f'处理 {fut} {date} 时出错: {str(e)}'
        if date > '20250101':
            return {
                'fut': fut,
                'date': date,
                'status': 'critical_error',
                'message': error_msg
            }
        return {
            'fut': fut,
            'date': date,
            'status': 'error',
            'message': error_msg
        }


# %% 并行计算所有期货品种的主买主卖量
def calc_order_flow_for_all_parallel(fut_list, zhuli_dir, data_base_path, save_dir, params, 
                                    use_cache=True, max_workers=None, executor_type='process'):
    """
    并行计算所有期货品种的主买主卖量
    
    参数:
    fut_list (list): 期货品种列表，如['IC', 'IF', 'IH', 'IM']
    zhuli_dir (Path): 主力合约数据目录
    data_base_path (str): 数据基础路径，如'http://172.16.30.3/future-data/tonglian-data/msg_backup'
    save_dir (Path): 保存目录
    params (dict): 参数字典，包含interval和keep_periods
    use_cache (bool): 是否使用缓存
    max_workers (int): 最大并行工作进程数，None表示使用CPU核心数
    executor_type (str): 执行器类型，'process' 或 'thread'
    
    返回:
    None
    """
    interval = params.get('interval', '1min')
    keep_periods = params.get('keep_periods', {
        'morning': ('09:31:00', '11:30:00'),
        'afternoon': ('13:01:00', '15:00:00')
    })
    
    # 确保保存目录存在
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有任务
    all_tasks = []
    
    for fut in fut_list:
        print(f'准备 {fut} 的任务...')
        
        # 读取主力合约数据
        zhuli_path = zhuli_dir / f'{fut}.parquet'
        if not zhuli_path.exists():
            print(f'主力合约文件不存在: {zhuli_path}')
            continue
            
        zhuli_data = pd.read_parquet(zhuli_path)
        
        for idx in zhuli_data.index:
            date = str(zhuli_data.loc[idx, 'date'])
            curr_trade = zhuli_data.loc[idx, 'curr_trade']
            
            task_params = (
                fut, date, curr_trade, data_base_path, save_dir,
                interval, keep_periods, use_cache
            )
            all_tasks.append(task_params)
    
    print(f'总共准备了 {len(all_tasks)} 个任务')
    
    # 设置最大工作进程数
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(all_tasks))
    
    print(f'使用 {executor_type} 执行器，最大工作进程数: {max_workers}')
    
    # 选择执行器类型
    ExecutorClass = ProcessPoolExecutor if executor_type == 'process' else ThreadPoolExecutor
    
    # 统计结果
    results = {
        'success': 0,
        'cached': 0,
        'error': 0,
        'critical_error': 0
    }
    
    start_time = time.time()
    
    # 执行并行计算
    with ExecutorClass(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(process_single_task, task): task for task in all_tasks}
        
        # 使用tqdm显示进度
        with tqdm(total=len(all_tasks), desc='处理任务') as pbar:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                
                try:
                    result = future.result()
                    status = result['status']
                    results[status] += 1
                    
                    # 如果是严重错误，抛出异常
                    if status == 'critical_error':
                        print(f"严重错误: {result['message']}")
                        raise Exception(result['message'])
                    
                    # 更新进度条描述
                    pbar.set_postfix({
                        '成功': results['success'],
                        '缓存': results['cached'],
                        '错误': results['error']
                    })
                    
                except Exception as e:
                    fut, date = task[0], task[1]
                    print(f'任务执行异常 {fut} {date}: {str(e)}')
                    results['error'] += 1
                
                pbar.update(1)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 打印统计结果
    print(f'\n处理完成！总耗时: {elapsed_time:.2f} 秒')
    print(f'成功处理: {results["success"]} 个任务')
    print(f'使用缓存: {results["cached"]} 个任务')
    print(f'处理错误: {results["error"]} 个任务')
    print(f'严重错误: {results["critical_error"]} 个任务')
    print(f'总任务数: {len(all_tasks)} 个')


# %% 兼容性函数：保持原有接口
def calc_order_flow_for_all(fut_list, zhuli_dir, data_base_path, save_dir, params, use_cache=True):
    """
    计算所有期货品种的主买主卖量（兼容性函数，调用并行版本）
    """
    calc_order_flow_for_all_parallel(
        fut_list=fut_list,
        zhuli_dir=zhuli_dir,
        data_base_path=data_base_path,
        save_dir=save_dir,
        params=params,
        use_cache=use_cache,
        max_workers=None,
        executor_type='process'
    )


# %% 主函数
if __name__ == '__main__':
    # 配置参数
    fut_list = ['IC', 'IF', 'IH', 'IM']
    zhuli_dir = Path('/mnt/nfs/30.132_xt_data1/future_zhuli')
    data_base_path = 'http://172.16.30.3/future-data/tonglian-data/msg_backup'
    save_dir = Path('/mnt/Data/xintang/future_factors/trade_flow_raw')
    
    params = {
        'interval': '1min',
        'keep_periods': {
            'morning': ('09:31:00', '11:30:00'),
            'afternoon': ('13:01:00', '15:00:00')
        }
    }
    
    # 执行并行计算
    calc_order_flow_for_all_parallel(
        fut_list=fut_list,
        zhuli_dir=zhuli_dir,
        data_base_path=data_base_path,
        save_dir=save_dir,
        params=params,
        use_cache=True,
        max_workers=None,  # 自动使用CPU核心数
        executor_type='process'  # 使用进程池，也可以选择'thread'
    )
    
    print('所有期货品种的主买主卖量并行计算完成！')