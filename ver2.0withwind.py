import pandas as pd
import pandas_ta as ta
import numpy as np
from scipy.signal import find_peaks
import mplfinance as mpf
from matplotlib import font_manager
import os
from datetime import datetime, timedelta, time
# 导入WindPy库
from WindPy import w

# ==============================================================================
# 全局设置
# ==============================================================================
# 定义要分析的证券代码，上证指数和中证1000指数["000001.SH", "000852.SH"]
SECURITY_CODES = ["000001.SH"]
# 本地数据存储路径
DATA_DIR = "hist_data"
# 创建存储数据的文件夹
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


# ** 核心修正：预定义所有分钟级别的时间桶 **
def _generate_bins_from_start_times(start_times, end_of_day_time=time(15, 0)):
    """辅助函数：根据开始时间列表生成(开始, 结束)时间对。"""
    bins = []
    if not start_times:
        return bins
    for i in range(len(start_times) - 1):
        bins.append((start_times[i], start_times[i + 1]))
    # 最后一个桶的结束时间固定为15:00
    bins.append((start_times[-1], end_of_day_time))
    return bins


# 定义每个周期的K线开始时间点
TIME_BINS_START_POINTS = {
    '3T': [time(h, m) for h in (9, 10, 11, 13, 14) for m in range(0, 60, 3) if
           not (h == 11 and m > 27) and not (h == 9 and m < 30)],
    '5T': [time(h, m) for h in (9, 10, 11, 13, 14) for m in range(0, 60, 5) if
           not (h == 11 and m > 25) and not (h == 9 and m < 30)],
    '15T': [time(9, 30), time(9, 45), time(10, 0), time(10, 15), time(10, 30), time(10, 45), time(11, 0), time(11, 15),
            time(13, 0), time(13, 15), time(13, 30), time(13, 45), time(14, 0), time(14, 15), time(14, 30),
            time(14, 45)],
    '30T': [time(9, 30), time(10, 0), time(10, 30), time(11, 0), time(13, 0), time(13, 30), time(14, 0), time(14, 30)],
    '60T': [time(9, 30), time(10, 30), time(13, 0), time(14, 0)],
    '120T': [time(9, 30), time(13, 0)],
}
# 根据开始时间点生成(开始, 结束)的时间区间
INTRA_DAY_BINS = {k: _generate_bins_from_start_times(sorted(list(set(v)))) for k, v in TIME_BINS_START_POINTS.items()}


# ==============================================================================
# Wind数据获取与处理模块
# ==============================================================================

def format_wind_data(wind_data):
    """
    将从Wind API获取的原始数据转换并格式化为我们程序所需的DataFrame格式。
    """
    if wind_data.ErrorCode != 0:
        print(f"Wind API数据获取错误，错误代码: {wind_data.ErrorCode}, 错误信息: {wind_data.Data[0][0]}")
        return None

    df = pd.DataFrame(wind_data.Data, index=wind_data.Fields, columns=pd.to_datetime(wind_data.Times)).T

    rename_map = {"OPEN": "open", "HIGH": "high", "LOW": "low", "CLOSE": "close"}
    df.rename(columns=rename_map, inplace=True)

    df.index.name = 'bob'
    df.reset_index(inplace=True)

    required_cols = ['bob', 'open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        print("错误：从Wind获取的数据经转换后，缺少必要字段。")
        return None

    return df[required_cols]


def fetch_and_save_wind_data(security_code, force_update=False):
    """
    从Wind获取指定证券的日线和1分钟线历史数据，并保存到本地。
    """
    print(f"\n===== 正在为 {security_code} 处理数据 =====")

    daily_file = os.path.join(DATA_DIR, f"{security_code}_daily.parquet")
    minute_file = os.path.join(DATA_DIR, f"{security_code}_1min.parquet")

    # --- 1. 获取日线数据 ---
    if os.path.exists(daily_file) and not force_update:
        print(f"发现本地日线数据，直接加载: {daily_file}")
        df_daily = pd.read_parquet(daily_file)
    else:
        print("正在从Wind下载日线数据...")
        start_date = "1990-01-01"
        today_date = datetime.now().strftime("%Y-%m-%d")
        wind_data_d = w.wsd(security_code, "open,high,low,close", start_date, today_date, "")
        df_daily = format_wind_data(wind_data_d)
        if df_daily is not None:
            df_daily.to_parquet(daily_file)
            print(f"日线数据已下载并保存到: {daily_file}")

    # --- 2. 获取分钟线数据 (采用分段循环策略) ---
    if os.path.exists(minute_file) and not force_update:
        print(f"发现本地分钟线数据，直接加载: {minute_file}")
        df_minute = pd.read_parquet(minute_file)
    else:
        print("正在从Wind下载分钟线数据（此过程可能需要较长时间）...")
        all_minute_chunks = []
        current_date = datetime.now()
        ten_years_ago = current_date - timedelta(days=365 * 10)

        while current_date > ten_years_ago:
            end_of_month = current_date.strftime("%Y-%m-%d 15:00:00")
            start_of_month = (current_date.replace(day=1)).strftime("%Y-%m-%d 09:30:00")
            print(f"  正在下载 {start_of_month} 至 {end_of_month} 的分钟数据...")

            wind_data_m = w.wsi(security_code, "open,high,low,close", start_of_month, end_of_month, "barSize=1")
            chunk_df = format_wind_data(wind_data_m)

            if chunk_df is not None and not chunk_df.empty:
                all_minute_chunks.append(chunk_df)

            current_date = current_date.replace(day=1) - timedelta(days=1)

        if all_minute_chunks:
            df_minute = pd.concat(all_minute_chunks).drop_duplicates(subset=['bob']).sort_values(by='bob')
            df_minute.to_parquet(minute_file)
            print(f"分钟线数据已下载并保存到: {minute_file}")
        else:
            print(f"未能获取到 {security_code} 的分钟线数据。")
            df_minute = None

    return df_daily, df_minute


# ==============================================================================
# 分析与计算模块 - 核心逻辑重构
# ==============================================================================

def resample_ohlc_by_grouping(df, rule):
    """
    根据日历周期（周、月、季）进行重采样，能够正确处理节假日。
    """
    logic = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    resampled_df = df.groupby(pd.Grouper(freq=rule)).agg(logic)
    return resampled_df.dropna(how='all')


def resample_by_count(df, count):
    """
    将数据每N行合并成一根K线，用于生成双日线、双周线。
    """
    resampled_data = []
    for i in range(0, len(df), count):
        chunk = df.iloc[i: i + count]
        if len(chunk) < count: continue

        resampled_data.append({
            'bob': chunk.index[0],
            'open': chunk['open'].iloc[0],
            'high': chunk['high'].max(),
            'low': chunk['low'].min(),
            'close': chunk['close'].iloc[-1]
        })

    if not resampled_data: return pd.DataFrame()
    return pd.DataFrame(resampled_data).set_index('bob')


def resample_intraday_by_bins(df_1m, freq_rule):
    """通过精确的、预定义的时间桶进行分钟线重采样。"""
    all_resampled_bars = []
    df_1m_indexed = df_1m.set_index('bob')

    time_bins_for_freq = INTRA_DAY_BINS.get(freq_rule)
    if not time_bins_for_freq:
        raise ValueError(f"未找到频率 '{freq_rule}' 的预定义时间桶。")

    for day_str, df_day in df_1m_indexed.groupby(df_1m_indexed.index.date):
        day_date = pd.to_datetime(day_str)

        for start_t, end_t in time_bins_for_freq:
            start_dt = datetime.combine(day_date, start_t)
            end_dt = datetime.combine(day_date, end_t)

            mask = (df_day.index >= start_dt) & (df_day.index < end_dt)
            if end_t == time(15, 0):  # 对15:00的收盘点进行闭区间处理
                mask = (df_day.index >= start_dt) & (df_day.index <= end_dt)

            chunk = df_day[mask]

            if not chunk.empty:
                all_resampled_bars.append({
                    'bob': start_dt,  # K线的开始时间为桶的左边界
                    'open': chunk['open'].iloc[0],
                    'high': chunk['high'].max(),
                    'low': chunk['low'].min(),
                    'close': chunk['close'].iloc[-1]
                })

    if not all_resampled_bars: return pd.DataFrame()
    return pd.DataFrame(all_resampled_bars).set_index('bob')


def process_last_minute_bar(df_1m):
    """
    处理1分钟数据，将每个交易日14:57到15:00的数据合并成一根K线，
    以解决尾盘集合竞价导致的数据不连续问题。
    """
    if df_1m is None or df_1m.empty:
        return df_1m

    print("正在对1分钟数据进行尾盘连续性处理...")

    df_1m_copy = df_1m.copy()
    df_1m_copy['bob'] = pd.to_datetime(df_1m_copy['bob'])
    df_1m_copy.set_index('bob', inplace=True)

    processed_days = []

    # 按天分组处理
    for day, day_df in df_1m_copy.groupby(df_1m_copy.index.date):
        day_str = day.strftime('%Y-%m-%d')

        time_1457 = pd.to_datetime(f"{day_str} 14:57:00")
        time_1500 = pd.to_datetime(f"{day_str} 15:00:00")

        # 获取当天14:57之前的数据
        before_end_session = day_df[day_df.index < time_1457]

        # 获取当天最后交易时段（14:57及之后）的数据
        end_session_data = day_df[day_df.index >= time_1457]

        # 仅当最后时段有数据时才进行合并
        if not end_session_data.empty and time_1457 in end_session_data.index:
            # 合成新的K线
            open_price = end_session_data.loc[time_1457, 'open']
            high_price = end_session_data['high'].max()
            low_price = end_session_data['low'].min()
            # 收盘价以15:00为准，若无则取能取到的最后一根
            close_price = end_session_data.loc[time_1500, 'close'] if time_1500 in end_session_data.index else \
            end_session_data['close'].iloc[-1]

            # 创建合并后的K线，时间戳设为14:57，代表这个K线时段的开始
            combined_bar = pd.DataFrame([{
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price
            }], index=[time_1457])

            # 拼接当天14:57前的数据和合并后的K线
            processed_day_df = pd.concat([before_end_session, combined_bar])
            processed_days.append(processed_day_df)
        else:
            # 如果不满足合并条件，则保留当天原始数据
            processed_days.append(day_df)

    if not processed_days:
        return pd.DataFrame()

    # 将所有处理完的交易日数据重新合并成一个DataFrame
    final_df = pd.concat(processed_days)
    final_df.reset_index(inplace=True)

    # ============================ BUG修复 ============================
    # 修正：当拼接不同来源的DataFrame时，索引名称可能会丢失。
    # reset_index() 会将无名索引转换为名为 'index' 的列。
    # 我们需要确保该列被重命名为 'bob'，以供后续函数使用。
    if 'index' in final_df.columns and 'bob' not in final_df.columns:
        final_df.rename(columns={'index': 'bob'}, inplace=True)
    # ========================== 修复结束 ===========================

    print("尾盘连续性处理完成。")
    return final_df


def calculate_indicators(df, level_name):
    """
    为给定的DataFrame计算MACD和均线。
    """
    if df.empty or len(df) < 55:
        print(f"提示：级别 '{level_name}' 数据量为 {len(df)}，少于55根K线，无法计算基本指标，将跳过此级别。")
        return pd.DataFrame()

    # 数据连续性问题已在预处理阶段解决，此处无需特殊处理，所有级别一视同仁
    macd_data = ta.macd(df['close'], fast=12, slow=26, signal=9, adjust=False)

    if macd_data is not None and not macd_data.empty:
        df['DIF'] = macd_data['MACD_12_26_9']
        df['DEA'] = macd_data['MACDs_12_26_9']
        df['MACD'] = macd_data['MACDh_12_26_9']

    df['BOLL'] = ta.sma(df['close'], length=21)
    df['55MA'] = ta.sma(df['close'], length=55)

    if len(df) >= 233:
        df['233MA'] = ta.sma(df['close'], length=233)
    else:
        print(f"提示：级别 '{level_name}' 数据量为 {len(df)}，不足以计算233MA。")

    return df


def prepare_all_dataframes(data_1m, data_d):
    """
    主函数：准备所有时间周期的数据。
    """
    # 新增步骤：对1分钟数据进行尾盘连续性处理
    if data_1m is not None and not data_1m.empty:
        data_1m = process_last_minute_bar(data_1m)

    print("正在准备所有时间周期的数据并计算指标...")
    dataframes = {}
    if data_1m is not None and not data_1m.empty:
        # **重要**：此处必须用set_index('bob')，后续所有计算都基于DatetimeIndex
        df_1m_indexed = data_1m.set_index('bob')
        dataframes['1'] = calculate_indicators(df_1m_indexed.copy(), '1')
        dataframes['3'] = calculate_indicators(resample_intraday_by_bins(data_1m, '3T'), '3')
        dataframes['5'] = calculate_indicators(resample_intraday_by_bins(data_1m, '5T'), '5')
        dataframes['15'] = calculate_indicators(resample_intraday_by_bins(data_1m, '15T'), '15')
        dataframes['30'] = calculate_indicators(resample_intraday_by_bins(data_1m, '30T'), '30')
        dataframes['60'] = calculate_indicators(resample_intraday_by_bins(data_1m, '60T'), '60')
        dataframes['120'] = calculate_indicators(resample_intraday_by_bins(data_1m, '120T'), '120')

    if data_d is not None and not data_d.empty:
        df_d_indexed = data_d.set_index('bob').sort_index()
        dataframes['1d'] = calculate_indicators(df_d_indexed.copy(), '1d')
        dataframes['2d'] = calculate_indicators(resample_by_count(df_d_indexed, 2), '2d')

        df_w = resample_ohlc_by_grouping(df_d_indexed, 'W-FRI')
        dataframes['1w'] = calculate_indicators(df_w, '1w')
        dataframes['2w'] = calculate_indicators(resample_by_count(df_w, 2), '2w')
        dataframes['m'] = calculate_indicators(resample_ohlc_by_grouping(df_d_indexed, 'M'), 'm')
        dataframes['s'] = calculate_indicators(resample_ohlc_by_grouping(df_d_indexed, 'Q'), 's')

    print("数据准备完成。")
    return dataframes


def get_level(current_level, offset, all_dfs):
    """
    根据当前级别和偏移量，返回目标级别名称。
    """
    seq1 = ['1', '3', '15', '60', '1d', '1w', 'm', 's']
    seq2 = ['5', '30', '120', '2d', '2w']

    current_seq, current_idx = (None, -1)
    if current_level in seq1:
        current_seq, current_idx = seq1, seq1.index(current_level)
    elif current_level in seq2:
        current_seq, current_idx = seq2, seq2.index(current_level)
    else:
        return '0'

    target_idx = current_idx + offset
    if 0 <= target_idx < len(current_seq):
        target_level_name = current_seq[target_idx]
        if target_level_name in all_dfs and not all_dfs[target_level_name].empty:
            return target_level_name
    return '0'


def check_divergence(df, multiplier=0.65):
    """
    判断最新的K线是否存在MACD背离。
    """
    if df.empty or len(df) < 20: return 0
    cross_signal = np.sign(df['DIF'] - df['DEA'])
    cross_indices_rev = []
    for i in range(len(df) - 1, 0, -1):
        if cross_signal.iloc[i] * cross_signal.iloc[i - 1] < 0:
            cross_indices_rev.append(df.index[i])
            if len(cross_indices_rev) >= 5: break

    cross_indices = pd.DatetimeIndex(reversed(cross_indices_rev))
    if len(cross_indices) < 3: return 0

    a_idx, b_idx, c_idx = cross_indices[-1], cross_indices[-2], cross_indices[-3]
    b_loc = df.index.get_loc(b_idx)

    if b_loc - df.index.get_loc(c_idx) < 5:
        if len(cross_indices) >= 5:
            c_idx = cross_indices[-5]
        else:
            return 0
    if df.index.get_loc(a_idx) - b_loc < 5: return 0

    dif_at_b = df.loc[b_idx, 'DIF']
    price_range_1, price_range_2 = df.loc[c_idx:b_idx], df.loc[b_idx:]
    dif_range_1, dif_range_2 = df.loc[c_idx:b_idx, 'DIF'], df.loc[a_idx:, 'DIF']

    if dif_at_b > 0:
        if price_range_2['high'].max() > price_range_1[
            'high'].max() and dif_range_1.max() * multiplier > dif_range_2.max():
            return 1
    elif dif_at_b < 0:
        if price_range_2['low'].min() < price_range_1['low'].min() and abs(dif_range_1.min()) * multiplier > abs(
                dif_range_2.min()):
            return -1
    return 0


def check_strong_weak_cross(level_name, df, all_dfs):
    """
    识别极强、极弱、金叉、死叉状态及其有效性。
    """
    results = {'strong': False, 'weak': False, 'golden_cross': False, 'death_cross': False}
    if df.empty or len(df) < 5: return results

    last, prev, p_prev = df.iloc[-1], df.iloc[-2], df.iloc[-3]
    status = {'type': None, 'start_idx': None}

    if last['DIF'] > 0 > last['DEA'] and (prev['DIF'] <= 0 or prev['DEA'] >= 0):
        status = {'type': 'strong', 'start_idx': df.index[-1]}
    elif last['DEA'] > 0 > last['DIF'] and (prev['DEA'] <= 0 or prev['DIF'] >= 0):
        status = {'type': 'weak', 'start_idx': df.index[-1]}
    elif last['DIF'] > last['DEA'] > 0:
        if prev['DEA'] > prev['DIF'] > 0:
            status = {'type': 'golden_cross', 'start_idx': df.index[-1]}
        elif prev['DIF'] > prev['DEA'] > 0 and p_prev['DEA'] > p_prev['DIF'] > 0:
            status = {'type': 'golden_cross', 'start_idx': df.index[-2]}
    elif 0 > last['DEA'] > last['DIF']:
        if 0 > prev['DIF'] > prev['DEA']:
            status = {'type': 'death_cross', 'start_idx': df.index[-1]}
        elif 0 > prev['DEA'] > prev['DIF'] and 0 > p_prev['DIF'] > p_prev['DEA']:
            status = {'type': 'death_cross', 'start_idx': df.index[-2]}

    if not status['type']: return results

    is_valid = True

    def check_sequential_failure(df_sub, condition_A, condition_B):
        if len(df_sub) < 6 or not condition_A.any(): return False
        first_A_idx = condition_A.idxmax()
        return condition_B.loc[first_A_idx:].any()

    offset = -1 if status['type'] in ['strong', 'weak'] else -3
    target_level = get_level(level_name, offset, all_dfs)
    if target_level != '0':
        df_sub = all_dfs[target_level].loc[status['start_idx']:]
        above = (df_sub['close'] > df_sub['BOLL']).rolling(3).sum() >= 3
        below = (df_sub['close'] < df_sub['BOLL']).rolling(3).sum() >= 3
        failure = False
        if status['type'] in ['strong', 'golden_cross']:
            failure = check_sequential_failure(df_sub, above, below)
        elif status['type'] in ['weak', 'death_cross']:
            failure = check_sequential_failure(df_sub, below, above)
        if failure: is_valid = False

    if is_valid: results[status['type']] = True
    return results


def check_wave_pattern(df, window=5):
    """
    识别上升或下跌浪型是否完整。
    """
    if df.empty or len(df) < window * 6: return 0
    high_indices, _ = find_peaks(df['high'], distance=window)
    low_indices, _ = find_peaks(-df['low'], distance=window)
    if len(high_indices) < 3 or len(low_indices) < 3: return 0

    recent_highs, recent_lows = df.index[high_indices[-3:]], df.index[low_indices[-3:]]
    highs, lows = df.loc[recent_highs, 'high'], df.loc[recent_lows, 'low']
    macds_high, macds_low = df.loc[recent_highs, 'MACD'], df.loc[recent_lows, 'MACD']
    is_valid_macd = (macds_high > 0).all() and (macds_low < 0).all()

    if highs.is_monotonic_increasing and lows.is_monotonic_increasing and is_valid_macd: return 1
    if highs.is_monotonic_decreasing and lows.is_monotonic_decreasing and is_valid_macd: return -1
    return 0


def run_analysis(all_dfs, security_code):
    """
    对所有时间周期的数据执行完整的分析流程。
    """
    print(f"\n===== 开始对 {security_code} 进行分析 =====")
    print("正在为所有级别计算特征(背离、浪型、强弱)...")
    for level, df in all_dfs.items():
        if not df.empty:
            last_idx = df.iloc[-1].name
            df.loc[last_idx, 'divergence'] = check_divergence(df)
            df.loc[last_idx, 'wave'] = check_wave_pattern(df)
            cross_status = check_strong_weak_cross(level, df, all_dfs)
            for k, v in cross_status.items(): df.loc[last_idx, k] = v

    print("正在执行核心分析逻辑...")
    analysis_results = {}
    all_levels_sorted = ['1', '3', '5', '15', '30', '60', '120', '1d', '2d', '1w', '2w', 'm', 's']

    for level in all_levels_sorted:
        if level not in all_dfs:
            analysis_results[level] = f"{level}: 未生成该级别数据"
            continue
        df = all_dfs[level]
        if df.empty:
            analysis_results[level] = f"{level}: 数据量不足，无法生成指标"
            continue
        if 'divergence' not in df.columns:
            analysis_results[level] = f"{level}: 指标已生成，但特征计算失败"
            continue

        last_row = df.iloc[-1]
        output = []
        if last_row.get('strong'): output.append("出现极强")
        if last_row.get('weak'): output.append("出现极弱")
        if last_row.get('golden_cross'): output.append("出现金叉")
        if last_row.get('death_cross'): output.append("出现死叉")

        if '55MA' not in last_row:
            output.append("55MA数据不足")
        elif last_row['close'] > last_row['55MA']:
            output.append("55MA上方")
            if last_row['divergence'] == 1:
                output.append("顶背离")
                level_p2 = get_level(level, 2, all_dfs)
                if level_p2 != '0' and (
                        all_dfs[level_p2].iloc[-1].get('golden_cross') or all_dfs[level_p2].iloc[-1].get('strong')):
                    output.append("背离因高级别极强/金叉失效")
                level_m1 = get_level(level, -1, all_dfs)
                if level_m1 != '0':
                    df_m1 = all_dfs[level_m1]
                    if len(df_m1) >= 3 and (df_m1['close'].tail(3) > df_m1['BOLL'].tail(3)).all():
                        output.append("-1级别boll支撑的主涨段")
                    elif len(df_m1) >= 3 and (df_m1['close'].tail(3) < df_m1['BOLL'].tail(3)).all():
                        output.append("跌破boll确认背离生效，看空")
                    else:
                        output.append("无特殊情况，背离生效，看空")
            elif last_row['wave'] == 1:
                output.append("无背离")
                level_m1 = get_level(level, -1, all_dfs)
                if level_m1 != '0':
                    if all_dfs[level_m1].iloc[-1].get('divergence') == 1:
                        output.append("本级别上升浪型完整，次级别有顶背离，看空")
                    else:
                        output.append("本级别上升浪型完整但次级别无顶背离，看多")
            else:
                output.append("浪型不完整且无背离，看多")
        else:
            output.append("55MA下方")
            if last_row['divergence'] == -1:
                output.append("底背离")
                level_p2 = get_level(level, 2, all_dfs)
                if level_p2 != '0' and (
                        all_dfs[level_p2].iloc[-1].get('death_cross') or all_dfs[level_p2].iloc[-1].get('weak')):
                    output.append("背离因高级别极弱/死叉失效")
                level_m1 = get_level(level, -1, all_dfs)
                if level_m1 != '0':
                    df_m1 = all_dfs[level_m1]
                    if len(df_m1) >= 3 and (df_m1['close'].tail(3) < df_m1['BOLL'].tail(3)).all():
                        output.append("-1级别boll压制的主跌段")
                    elif len(df_m1) >= 3 and (df_m1['close'].tail(3) > df_m1['BOLL'].tail(3)).all():
                        output.append("上穿boll确认背离生效，看多")
                    else:
                        output.append("无特殊情况，背离生效，看多")
            elif last_row['wave'] == -1:
                output.append("无背离")
                level_m1 = get_level(level, -1, all_dfs)
                if level_m1 != '0':
                    if all_dfs[level_m1].iloc[-1].get('divergence') == -1:
                        output.append("本级别下跌浪型完整，次级别有底背离，看多")
                    else:
                        output.append("本级别下跌浪型完整但次级别无底背离，看空")
            else:
                output.append("浪型不完整且无背离，看空")
        analysis_results[level] = f"{level}: " + "; ".join(output)

    print("\n" + f"===== {security_code} 分析结果 =====")
    seq1_print, seq2_print = ['1', '3', '15', '60', '1d', '1w', 'm', 's'], ['5', '30', '120', '2d', '2w']
    for level in seq1_print: print(analysis_results.get(level, f"{level}: 未执行分析"))
    print()
    for level in seq2_print: print(analysis_results.get(level, f"{level}: 未执行分析"))


def plot_kline_chart(df_kline, security_code, level_name):
    """
    绘制指定级别的K线图，包含均线和MACD附图。
    """
    if df_kline.empty or len(df_kline) < 50:
        print(f"\n{security_code} 的 {level_name} 级别数据不足，无法绘制图表。")
        return

    print(f"\n正在为 {security_code} 绘制 {level_name} 级别图表...")

    font_path, rc_params = None, {'axes.unicode_minus': False}
    if os.name == 'nt':
        font_dir = 'C:/Windows/Fonts'
        font_options = ['simhei.ttf', 'msyh.ttc', 'kaiti.ttf', 'Deng.ttf']
        for font_name in font_options:
            path = os.path.join(font_dir, font_name)
            if os.path.exists(path):
                font_path = path
                break

    if font_path:
        print(f"提示：找到并使用中文字体 -> {font_path}")
        prop = font_manager.FontProperties(fname=font_path)
        rc_params['font.family'] = prop.get_name()
    else:
        print("警告：未能在 C:/Windows/Fonts/ 找到指定的中文字体，图表标题可能无法正常显示。")

    s = mpf.make_mpf_style(base_mpf_style='charles', rc=rc_params)
    df_plot = df_kline.tail(2000).copy()
    macd_colors = ['#ff4d4d' if v > 0 else '#43cf7c' for v in df_plot['MACD']]
    ap = [
        mpf.make_addplot(df_plot[['DIF', 'DEA']], panel=1, ylabel='MACD'),
        mpf.make_addplot(df_plot['MACD'], type='bar', color=macd_colors, panel=1)
    ]
    moving_averages = [ma for ma in [21, 55, 233] if
                       f'{ma}MA' in df_plot.columns or (ma == 21 and 'BOLL' in df_plot.columns)]

    chart_title = f'\n{security_code} {level_name} K线图与技术指标'
    chart_filename = f"{security_code}_{level_name.replace(' ', '')}_chart.png"

    mpf.plot(df_plot, type='candle', style=s, title=chart_title,
             ylabel='价格', mav=tuple(moving_averages) if moving_averages else None,
             addplot=ap, panel_ratios=(3, 1), figratio=(16, 9), volume=False,
             savefig=chart_filename)
    print(f"图表已保存为 {chart_filename}")

    print(f"\n===== {security_code} ({level_name}) 最近5根K线的MACD指标数值 =====")
    print(df_plot[['close', 'DIF', 'DEA', 'MACD']].tail(5).round(4))


# ==============================================================================
# 5. 主程序入口
# ==============================================================================
if __name__ == '__main__':
    try:
        w.start()
        print("Wind API 连接成功。")
    except Exception as e:
        print(f"Wind API 连接失败，请确保已安装Wind并登录。错误: {e}")
        exit()

    for code in SECURITY_CODES:
        daily_data, minute_data = fetch_and_save_wind_data(code, force_update=False)
        if daily_data is None:
            print(f"无法获取 {code} 的日线数据，跳过分析。")
            continue
        all_dfs = prepare_all_dataframes(minute_data, daily_data)
        run_analysis(all_dfs, code)

        # 绘制处理后的1分钟图
        if '30' in all_dfs and not all_dfs['30'].empty:
            plot_kline_chart(all_dfs['2w'], code, level_name='2周')

    w.stop()
    print("\nWind API 已断开连接。程序运行结束。")
