# src/data_processing.py

import numpy as np
import pandas as pd
from pathlib import Path
import os
from .config import DATA_DIR

def process_eeg_to_df(data: np.ndarray) -> pd.DataFrame:
    """
    Numpy配列を受け取り、ラベル付けされたDataFrameを返す内部処理用の関数。
    """
    # 2. チャネル名の設定
    channel_names = ['PFC', 'PPC', 'A1', 'V1', 'Stimulus']
    df = pd.DataFrame(data.T, columns=channel_names)

    # --- ここから追加 ---
    # Time_s列を追加 (サンプリングレートを1000Hzと仮定)
    sampling_rate = 1000
    df['Time_s'] = df.index / sampling_rate
    # --- ここまで追加 ---
    
    # 3. 刺激ラベルの付与
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