# src/erp_utils.py
import numpy as np
import pandas as pd
from typing import List, Dict, Iterable, Tuple, Optional

def calculate_erp(
    df: pd.DataFrame,
    event_ids: Iterable[int],
    channels: List[str],
    tmin_sec: float,
    tmax_sec: float,
    baseline_sec: Tuple[float, float],
    sampling_rate: int
) -> Dict[str, np.ndarray]:
    """
    単一のデータフレームから、指定されたイベントのERPを計算する。

    Args:
        df (pd.DataFrame): 個体のデータフレーム
        event_ids (Iterable[int]): 対象とするイベントIDのイテラブル (rangeなど)
        channels (List[str]): 計算対象のチャンネル名のリスト
        tmin_sec (float): エポック開始時間 (秒, 刺激開始時点=0)
        tmax_sec (float): エポック終了時間 (秒)
        baseline_sec (Tuple[float, float]): ベースライン期間 (秒)
        sampling_rate (int): サンプリング周波数 (Hz)

    Returns:
        Dict[str, np.ndarray]: {channel_name: erp_waveform} の辞書
    """
    
    # 時間をサンプル数に変換
    tmin_samples = int(tmin_sec * sampling_rate)
    tmax_samples = int(tmax_sec * sampling_rate)
    total_samples = tmax_samples - tmin_samples
    
    # ベースライン期間をエポック内のインデックスに変換
    baseline_start_idx = int(baseline_sec[0] * sampling_rate) - tmin_samples
    baseline_end_idx = int(baseline_sec[1] * sampling_rate) - tmin_samples
    baseline_indices = np.arange(baseline_start_idx, baseline_end_idx)

    epochs_dict = {ch: [] for ch in channels}
    
    # 対象イベントIDの刺激開始インデックスを高速に検索
    valid_event_mask = df['Event_ID'].isin(event_ids)
    # 刺激開始時点（Event_IDが変わる最初のインデックス）のみを抽出
    onset_indices = df.index[valid_event_mask & (df['Event_ID'].diff() != 0)]

    for stim_onset_idx in onset_indices:
        start_cut_idx = stim_onset_idx + tmin_samples
        end_cut_idx = stim_onset_idx + tmax_samples
        
        # データ範囲外の場合はスキップ
        if start_cut_idx < 0 or end_cut_idx >= len(df):
            continue
            
        for channel in channels:
            # .values を使って NumPy 配列として切り出す
            epoch = df[channel].iloc[start_cut_idx:end_cut_idx].values
            
            if len(epoch) == total_samples:
                # ベースライン補正
                baseline_value = np.mean(epoch[baseline_indices])
                epoch_corrected = epoch - baseline_value
                epochs_dict[channel].append(epoch_corrected)

    # 試行平均（ERP）を計算
    erp_dict = {}
    for channel in channels:
        if epochs_dict[channel]:
            epochs_matrix = np.array(epochs_dict[channel])
            erp_dict[channel] = np.mean(epochs_matrix, axis=0)
        else:
            # データがない場合はNaNで埋める
            erp_dict[channel] = np.full(total_samples, np.nan)
            
    return erp_dict