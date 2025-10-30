# src/plot_utils.py
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib # 日本語表示のため
from typing import Iterable, Optional

from .config import EEG_CHANNELS_TO_FILTER


def plot_erp_with_dynamic_window(df, start_id, end_id, pre_stim_ms=200, post_offset_ms=50, title="", eeg_channels: Optional[Iterable[str]] = None):
    """
    加算平均波形をプロットする。表示範囲は刺激終了後、指定時間まで動的に調整される。
    
    Args:
        df (pd.DataFrame): データフレーム
        start_id (int): 開始イベントID
        end_id (int): 終了イベントID
        pre_stim_ms (int): 刺激前の描画期間 (ms)
        post_offset_ms (int): 刺激 "終了後" の描画期間 (ms)
        title (str): グラフ全体のタイトル
    """
    # チャンネルリストを引数経由で受け取り、指定がなければ config の設定を使う
    if eeg_channels is None:
        eeg_channels = list(EEG_CHANNELS_TO_FILTER)
    
    # ステップ1: まず平均刺激持続時間を計算
    stim_durations_ms = []
    valid_event_ids = []
    for event_id in range(start_id, end_id + 1):
        event_indices = df.index[df['Event_ID'] == event_id]
        if not event_indices.empty:
            duration_ms = event_indices[-1] - event_indices[0]
            stim_durations_ms.append(duration_ms)
            valid_event_ids.append(event_id)

    if not stim_durations_ms:
        print(f"イベントID {start_id}-{end_id} の有効なデータが見つかりませんでした。")
        return
        
    avg_duration_ms = int(np.mean(stim_durations_ms))
    
    # ステップ2: 描画範囲を決定
    plot_post_ms = avg_duration_ms + post_offset_ms

    # ステップ3: エポック化（データ切り出し）
    epochs_dict = {ch: [] for ch in eeg_channels}
    for event_id in valid_event_ids:
        stim_onset_idx = df.index[df['Event_ID'] == event_id][0]
        
        start_cut_idx = stim_onset_idx - pre_stim_ms
        end_cut_idx = stim_onset_idx + plot_post_ms
        
        if start_cut_idx < 0 or end_cut_idx >= len(df):
            continue
            
        for channel in eeg_channels:
            epoch = df[channel].iloc[start_cut_idx:end_cut_idx].values
            if len(epoch) == pre_stim_ms + plot_post_ms:
                epochs_dict[channel].append(epoch)

    # ステップ4: グラフ描画
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(15, 12), sharex=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    time_axis = np.linspace(-pre_stim_ms / 1000, plot_post_ms / 1000, pre_stim_ms + plot_post_ms)
    avg_duration_s = avg_duration_ms / 1000.0

    for i, channel in enumerate(eeg_channels):
        ax = axes[i]
        
        if not epochs_dict[channel]:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center')
            continue
            
        epochs_matrix = np.array(epochs_dict[channel])
        mean_waveform = np.mean(epochs_matrix, axis=0)
        std_waveform = np.std(epochs_matrix, axis=0)
        
        ax.plot(time_axis, mean_waveform, label='Mean', color=colors[i], linewidth=2)
        ax.fill_between(time_axis, mean_waveform - std_waveform, mean_waveform + std_waveform, color=colors[i], alpha=0.2)
        
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Stimulus Onset')
        ax.axvline(x=avg_duration_s, color='blue', linestyle='--', linewidth=1.5, label='Stimulus Offset')
        
        ax.set_ylabel(f'{channel}\nAmplitude')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        if i == 0:
            ax.legend(loc='upper left')

    axes[-1].set_xlabel('刺激開始からの時間 (秒)')
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()