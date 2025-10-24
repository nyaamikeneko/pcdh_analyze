import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt
import re  # 👈 正規表現ライブラリをインポート

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
    (この関数自体に変更はありません)
    """
    # configから読み込んだチャンネル名リストを使用
    df = pd.DataFrame(data.T, columns=RAW_CHANNEL_NAMES) 

    # configから読み込んだサンプリングレートを使用
    df['Time_s'] = df.index / SAMPLING_RATE 
    
    # --- フィルタ処理 ---
    # configから読み込んだチャンネルリストをループ
    for channel in EEG_CHANNELS_TO_FILTER: 
        filtered_col_name = f'{channel}_filtered'
        # configから読み込んだフィルタパラメータを使用
        df[filtered_col_name] = bandpass_filter( 
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

def _extract_metadata_from_filename(filename_stem: str):
    """
    ファイル名のステムから Subject_ID と Genotype を抽出するヘルパー関数。
    """
    # Subject_ID (2-3桁の数字) を抽出
    id_match = re.search(r'(\d{2,3})', filename_stem)
    subject_id = id_match.group(1) if id_match else "Unknown_ID"
    
    # Genotype (wt, het, homo) を抽出 (大文字小文字を区別しない)
    geno_match = re.search(r'(wt|het|homo)', filename_stem, re.IGNORECASE)
    
    genotype = "Unknown_Geno"
    if geno_match:
        raw_geno = geno_match.group(1).lower()
        # 表記を統一 (wt -> WT, het -> Het, homo -> Homo)
        if raw_geno == 'wt':
            genotype = 'WT'
        elif raw_geno == 'het':
            genotype = 'Het'
        elif raw_geno == 'homo':
            genotype = 'Homo'
            
    return subject_id, genotype

def create_processed_file(filename: str, processed_dir: Path):
    """
    単一の.npyファイルを読み込み、処理してParquet形式で保存する関数。
    (👈 メタデータを追加するよう修正)
    """
    filename_stem = Path(filename).stem
    
    # 出力パスを生成 (例: 203wt~.parquet)
    output_path = processed_dir / f"{filename_stem}.parquet"
    
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
    
    # 3. 👈 ファイル名からメタデータ（IDとGenotype）を抽出
    subject_id, genotype = _extract_metadata_from_filename(filename_stem)
    
    # 4. 👈 DataFrameにメタデータを列として追加
    processed_df['Subject_ID'] = subject_id
    processed_df['Genotype'] = genotype
    
    # 5. Parquet形式で保存
    processed_df.to_parquet(output_path)
    print(f"  -> '{output_path.name}' として保存しました。(ID: {subject_id}, Genotype: {genotype})")