# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 10:19:45 2024

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import numpy as np
from datetime import datetime, timedelta
import re


# %%
def parse_time_string(time_string):
    """
    解析格式为 "xxdxxhxxminxxs" 的时间字符串并转换为总秒数。

    参数:
    time_string (str): 表示时间间隔的字符串，如 "1d2h30min45s"。

    返回:
    int: 转换后的总秒数。

    异常:
    ValueError: 如果时间字符串格式无效。
    """
    # 正则模式支持 d（天），h（小时），min（分钟），s（秒）
    pattern = re.compile(r'(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)min)?(?:(\d+)s)?')
    match = pattern.fullmatch(time_string)
    
    if not match:
        raise ValueError("Invalid time string format")
    
    # 将天、小时、分钟、秒提取并转换为整数
    days = int(match.group(1)) if match.group(1) else 0
    hours = int(match.group(2)) if match.group(2) else 0
    mins = int(match.group(3)) if match.group(3) else 0
    secs = int(match.group(4)) if match.group(4) else 0
    
    # 转换为总秒数
    total_seconds = days * 24 * 60 * 60 + hours * 60 * 60 + mins * 60 + secs
    return total_seconds


def add_time(time_str, interval_params, minus=False):
    """
    给定一个时间字符串（格式为'HH:MM'或'HH:MM:SS'）和时间间隔参数，返回调整后的时间字符串。
    
    :param time_str: 初始时间字符串（'HH:MM'或'HH:MM:SS'）
    :param interval_params: 时间间隔参数，字典形式，如 {'seconds': 10} 或 {'minutes': 5}
    :return: 调整后的时间字符串（'HH:MM:SS'格式）
    """
    time_format = '%H:%M:%S' if len(time_str.split(':')) == 3 else '%H:%M'
    time_obj = datetime.strptime(time_str, time_format)
    interval = timedelta(**interval_params)
    if minus:
        interval *= -1
    adjusted_time = (time_obj + interval).strftime('%H:%M:%S')
    return adjusted_time
        
        
def get_a_share_intraday_time_series(date: datetime, interval_params, trading_periods=None):
    """
    生成A股市场交易时间内的等间隔时间序列。
    
    :param date: 日期 (datetime对象)
    :param interval_params: 时间间隔参数 (字典形式，如 {'seconds': 1} 或 {'minutes': 1})
    :param trading_periods: 交易时段参数 (字典形式，包含 'morning' 和 'afternoon' 键，每个键的值为 (start_time, end_time))
        默认为 A 股市场常见的交易时段，支持精确到秒:
            'morning': ('09:30:00', '11:30:00')
            'afternoon': ('13:00:00', '15:00:00')
    :return: numpy数组，包含当天交易时间内的时间戳序列 (毫秒级)
    """
    # 设置默认的交易时段
    default_trading_periods = {
        'morning': ('09:30:00', '11:30:00'),
        'afternoon': ('13:00:00', '15:00:00')
    }
    
    # 使用用户提供的交易时段，如果未提供则使用默认值
    trading_periods = trading_periods or default_trading_periods

    # 自动化计算交易时段的开始和结束时间，支持到秒
    morning_start_str = trading_periods['morning'][0]
    morning_end_str = trading_periods['morning'][1]
    afternoon_start_str = trading_periods['afternoon'][0]
    afternoon_end_str = trading_periods['afternoon'][1]
    
    # 将时间字符串转换为 datetime 对象
    morning_start = datetime.strptime(f"{date.strftime('%Y-%m-%d')} {morning_start_str}", '%Y-%m-%d %H:%M:%S')
    morning_end = datetime.strptime(f"{date.strftime('%Y-%m-%d')} {morning_end_str}", '%Y-%m-%d %H:%M:%S')
    afternoon_start = datetime.strptime(f"{date.strftime('%Y-%m-%d')} {afternoon_start_str}", '%Y-%m-%d %H:%M:%S')
    afternoon_end = datetime.strptime(f"{date.strftime('%Y-%m-%d')} {afternoon_end_str}", '%Y-%m-%d %H:%M:%S')

    interval = timedelta(**interval_params)
    
    # 生成上午交易时间序列
    morning_series = (np.arange(morning_start, morning_end + interval, 
                               interval).astype('i8') // 1e3).astype('i8')
    
    # 生成下午交易时间序列
    afternoon_series = (np.arange(afternoon_start, afternoon_end + interval, 
                                 interval).astype('i8') // 1e3).astype('i8')

    # 合并上午和下午时间序列
    time_series = np.concatenate([morning_series, afternoon_series]).view('datetime64[ms]')
    
    return time_series