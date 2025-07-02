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
data['vwap'] = data['Turnover'] / data['Volume'] / 200
data['midprice'] = (data['BidPrice1'] + data['AskPrice1']) / 2
data['midprice_diff'] = data['midprice'].diff()
data['vwap_lastmpc_diff'] = (data['vwap'] - data['midprice']).shift(1)
data['if_mpc_gt_last'] = data['midprice_diff'] > 0
data['if_mpc_lt_last'] = data['midprice_diff'] < 0
data['if_vwap_gt_last_mpc'] = data['vwap_lastmpc_diff'] > 0
data['if_vwap_lt_last_mpc'] = data['vwap_lastmpc_diff'] < 0

# 初始化trade_direction
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

# 按分钟聚合
data['DateTime'] = pd.to_datetime(data['TradDay'].astype(str) + ' ' + data['UpdateTime'].astype(str))
data.set_index('DateTime', inplace=True)
minute_data = data.resample('1min').agg({
    'act_buy_amount': 'sum',
    'act_sell_amount': 'sum'
})


interval_timedelta = {'seconds': parse_time_string(interval)}
keep_ts = get_a_share_intraday_time_series(date_in_dt, interval_timedelta, 
                                           trading_periods=keep_periods)
output = minute_data.reindex(index=keep_ts)
    