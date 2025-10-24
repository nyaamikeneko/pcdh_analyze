import logging
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt
import re

# 設定をインポート
from .config import (
    DATA_DIR,
    PROCESSED_DIR,
    SAMPLING_RATE,
    RAW_CHANNEL_NAMES,
    EEG_CHANNELS_TO_FILTER,
    FILTER_LOWCUT,
    FILTER_HIGHCUT,
    FILTER_ORDER,
    STIMULUS_THRESHOLD,
    EVENT_RANGES,
)

# ロガー設定
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """データにバンドパスフィルタを適用する関数。"""
    data = np.asarray(data, dtype=np.float64)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def process_eeg_to_df(data: np.ndarray,
                      sampling_rate: int = None,
                      stimulus_threshold: float = None,
                      event_ranges: dict = None) -> pd.DataFrame:
    """
    Numpy配列を受け取り、ラベル付けとフィルタリング処理をした DataFrame を返す。

    Args:
        data: numpy array shaped (n_samples, n_channels) or (n_channels, n_samples) depending on source.
        sampling_rate: サンプリング周波数。省略時は `config.SAMPLING_RATE` を使用。
        stimulus_threshold: 刺激検出の閾値。省略時は `config.STIMULUS_THRESHOLD` を使用。
        event_ranges: イベント名->(start,end) の辞書。省略時は `config.EVENT_RANGES` を使用。

    Returns:
        pd.DataFrame: フィルタリングとメタデータ列が付加されたデータフレーム。
    """
    if sampling_rate is None:
        sampling_rate = SAMPLING_RATE
    if stimulus_threshold is None:
        stimulus_threshold = STIMULUS_THRESHOLD
    if event_ranges is None:
        event_ranges = EVENT_RANGES

    # 入力の形状に合わせる: 期待は (n_samples, n_channels) の場合が多いので、
    arr = np.asarray(data)
    if arr.ndim == 2 and arr.shape[0] == len(RAW_CHANNEL_NAMES):
        # (n_channels, n_samples) -> transpose
        arr = arr.T

    df = pd.DataFrame(arr, columns=RAW_CHANNEL_NAMES)

    # Time_s を確実に作る
    df['Time_s'] = df.index / float(sampling_rate)

    # --- フィルタ処理 ---
    for channel in EEG_CHANNELS_TO_FILTER:
        if channel not in df.columns:
            logger.warning("Channel %s not found in data columns", channel)
            continue
        filtered_col_name = f'{channel}_filtered'
        df[filtered_col_name] = bandpass_filter(
            data=df[channel].values,
            lowcut=FILTER_LOWCUT,
            highcut=FILTER_HIGHCUT,
            fs=sampling_rate,
            order=FILTER_ORDER
        )

    # --- 刺激ラベルの付与 ---
    if 'Stimulus' not in df.columns:
        raise KeyError("Input data must contain a 'Stimulus' column for event detection")

    is_stim_on = df['Stimulus'] > stimulus_threshold
    stim_starts = is_stim_on & ~is_stim_on.shift(1).fillna(False)
    event_ids = stim_starts.cumsum()
    df['Event_ID'] = event_ids.where(is_stim_on, 0).astype(int)

    # Stimulus_Type を event_ranges を使って割り当て
    df['Stimulus_Type'] = 'No_Stimulus'
    for name, (a, b) in event_ranges.items():
        mask = (df['Event_ID'] > 0) & (df['Event_ID'] >= a) & (df['Event_ID'] <= b)
        df.loc[mask, 'Stimulus_Type'] = name

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

def create_processed_file(filename: str, processed_dir: Path = None, overwrite: bool = False):
    """
    単一の.npyファイルを読み込み、処理してParquet形式で保存する関数。
    (👈 メタデータを追加するよう修正)
    """
    if processed_dir is None:
        processed_dir = PROCESSED_DIR

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    filename_stem = Path(filename).stem
    output_path = processed_dir / f"{filename_stem}.parquet"

    if output_path.exists() and not overwrite:
        logger.info("%s already exists; skipping (set overwrite=True to replace)", output_path.name)
        return

    file_path = DATA_DIR / filename
    if not file_path.exists():
        logger.error("Input file not found: %s", file_path)
        raise FileNotFoundError(file_path)

    logger.info("Processing '%s' -> '%s'", file_path.name, output_path.name)

    # safe load (support .npy, .npz maybe)
    try:
        raw_data = np.load(file_path, allow_pickle=False)
    except Exception as exc:
        logger.exception("Failed to load file %s: %s", file_path, exc)
        raise

    # Process
    processed_df = process_eeg_to_df(raw_data)

    # Extract metadata from filename and attach
    subject_id, genotype = _extract_metadata_from_filename(filename_stem)
    processed_df['Subject_ID'] = subject_id
    processed_df['Genotype'] = genotype

    # Save
    try:
        processed_df.to_parquet(output_path)
        logger.info("Saved processed data to %s (ID: %s, Genotype: %s)", output_path.name, subject_id, genotype)
    except Exception as exc:
        logger.exception("Failed to save parquet file %s: %s", output_path, exc)
        raise