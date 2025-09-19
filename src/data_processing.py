# src/data_processing.py

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt

# --- config.pyから設定値をまとめてインポート ---
# 👈 必要な設定値をすべて読み込む
from .config import (
    DATA_DIR,
    SAMPLING_RATE,
    RAW_CHANNEL_NAMES,
    EEG_CHANNELS_TO_FILTER,
    FILTER_LOWCUT,
    FILTER_HIGHCUT,
    FILTER_ORDER
)

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    データにバンドパスフィルタを適用する関数。
    (この関数自体に変更はありません)
    """
    data = np.asarray(data, dtype=np.float64)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def process_eeg_to_df(data: np.ndarray) -> pd.DataFrame:
    """
    Numpy配列を受け取り、ラベル付けとフィルタリング処理をしたDataFrameを返す。
    """
    # configから読み込んだチャンネル名リストを使用
    df = pd.DataFrame(data.T, columns=RAW_CHANNEL_NAMES) # 👈 変更

    # configから読み込んだサンプリングレートを使用
    df['Time_s'] = df.index / SAMPLING_RATE # 👈 変更
    
    # --- フィルタ処理 ---
    # configから読み込んだチャンネルリストをループ
    for channel in EEG_CHANNELS_TO_FILTER: # 👈 変更
        filtered_col_name = f'{channel}_filtered'
        # configから読み込んだフィルタパラメータを使用
        df[filtered_col_name] = bandpass_filter( # 👈 変更
            data=df[channel],
            lowcut=FILTER_LOWCUT,
            highcut=FILTER_HIGHCUT,
            fs=SAMPLING_RATE,
            order=FILTER_ORDER
        )
    
    # --- 刺激ラベルの付与 (ここは変更なし) ---
    is_stim_on = df['Stimulus'] > 5
    stim_starts = is_stim_on & ~is_stim_on.shift(1).fillna(False)
    event_ids = stim_starts.cumsum()
    df['Event_ID'] = event_ids.where(is_stim_on, 0)

    conditions = [
        (df['Event_ID'] > 0) & (df['Event_ID'] <= 600),
        (df['Event_ID'] > 600) & (df['Event_ID'] <= 1200),
        (df['Event_ID'] > 1200) & (df['Event_ID'] <= 1800)
    ]
    choices = ['Light', 'Sound', 'Light+Sound']
    df['Stimulus_Type'] = np.select(conditions, choices, default='No_Stimulus')
    
    return df

def create_processed_file(filename: str, processed_dir: Path):
    """
    単一の.npyファイルを読み込み、処理してParquet形式で保存する関数。
    """
    # 出力パスを生成 (例: wt262avs.adicht_rec2.parquet)
    output_path = processed_dir / f"{Path(filename).stem}.parquet"
    
    # すでに処理済みファイルが存在する場合はスキップ
    if output_path.exists():
        print(f"'{output_path.name}' は既に存在するためスキップします。")
        return

    # 1. データの読み込み
    print(f"'{filename}' を処理中...")
    file_path = DATA_DIR / filename
    raw_data = np.load(file_path)
    
    # 2. DataFrameに変換してラベル付け
    processed_df = process_eeg_to_df(raw_data)
    
    # 3. Parquet形式で保存
    processed_df.to_parquet(output_path)
    print(f"  -> '{output_path.name}' として保存しました。")