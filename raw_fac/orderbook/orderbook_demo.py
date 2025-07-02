# -*- coding: utf-8 -*-
"""
Created on Mon May 12 13:43:24 2025

@author: Xintang Zheng

星星: ★ ☆ ✪ ✩ 🌟 ⭐ ✨ 🌠 💫 ⭐️
勾勾叉叉: ✓ ✔ ✕ ✖ ✅ ❎
报警啦: ⚠ ⓘ ℹ ☣
箭头: ➔ ➜ ➙ ➤ ➥ ↩ ↪
emoji: 🔔 ⏳ ⏰ 🔒 🔓 🛑 🚫 ❗ ❓ ❌ ⭕ 🚀 🔥 💧 💡 🎵 🎶 🧭 📅 🤔 🧮 🔢 📊 📈 📉 🧠 📝

"""
# %% imports
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime


# %% add sys path
file_path = Path(__file__).resolve()
file_dir = file_path.parents[0]
project_dir = file_path.parents[0]
sys.path.append(str(project_dir))


# %%
from utils.timeutils import parse_time_string, get_a_share_intraday_time_series


# %%
path = 'http://172.16.30.3/future-data/tonglian-data/msg_backup/20231213/mdl_21_1_0.csv'
data_all = pd.read_csv(path)
data = data_all[data_all['InstruID']=='IC2401']

interval = '1min'
keep_periods = {
    'morning': ('09:31:00', '11:30:00'),
    'afternoon': ('13:01:00', '15:00:00')
}
date = '20231213'
date_in_dt = datetime.strptime(date, '%Y%m%d')


# %%
# 计算bid_amount (买方金额)
data['bid_amount'] = (
    data['BidPrice1'] * data['BidVolume1'] +
    data['BidPrice2'] * data['BidVolume2'] +
    data['BidPrice3'] * data['BidVolume3'] +
    data['BidPrice4'] * data['BidVolume4'] +
    data['BidPrice5'] * data['BidVolume5']
)

# 计算ask_amount (卖方金额)
data['ask_amount'] = (
    data['AskPrice1'] * data['AskVolume1'] +
    data['AskPrice2'] * data['AskVolume2'] +
    data['AskPrice3'] * data['AskVolume3'] +
    data['AskPrice4'] * data['AskVolume4'] +
    data['AskPrice5'] * data['AskVolume5']
)

# 按分钟聚合
data['DateTime'] = pd.to_datetime(data['TradDay'].astype(str) + ' ' + data['UpdateTime'].astype(str))
data.set_index('DateTime', inplace=True)
minute_data = data.resample('1min', closed='right', label='right').agg({
    'bid_amount': 'sum',
    'ask_amount': 'sum'
})


interval_timedelta = {'seconds': parse_time_string(interval)}
keep_ts = get_a_share_intraday_time_series(date_in_dt, interval_timedelta, 
                                           trading_periods=keep_periods)
output = minute_data.reindex(index=keep_ts)
    