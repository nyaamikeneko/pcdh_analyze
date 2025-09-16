import matplotlib.pyplot as plt
# import japanize_matplotlib # .pyファイル側では不要なことが多い

def plot_event_waveform(df, event_id, padding_ms=200):
    """
    指定されたイベントIDの前後区間を含めてプロットする。
    Stimulus信号は第2Y軸に元のスケールで表示する。
    """
    # イベントIDに該当する区間のデータを抽出
    event_df = df[df['Event_ID'] == event_id]
    
    if event_df.empty:
        print(f"イベントID {event_id} は見つかりませんでした。")
        return
        
    stim_type = event_df['Stimulus_Type'].iloc[0]
    
    padding_points = int(padding_ms)
    event_start_idx = event_df.index[0]
    event_end_idx = event_df.index[-1]
    plot_start_idx = max(0, event_start_idx - padding_points)
    plot_end_idx = min(len(df) - 1, event_end_idx + padding_points)
    plot_df = df.loc[plot_start_idx:plot_end_idx]
    
    # グラフ描画
    fig, ax1 = plt.subplots(figsize=(15, 6))
    
    # X軸のオフセット表記を無効にする
    ax1.get_xaxis().get_major_formatter().set_useOffset(False)

    # EEGチャネルのプロット (左のY軸)
    eeg_channels = ['PFC', 'PPC', 'A1', 'V1']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd'] 
    
    for i, channel in enumerate(eeg_channels):
        ax1.plot(plot_df['Time_s'], plot_df[channel], label=channel, color=colors[i], alpha=0.9)
    
    ax1.set_xlabel('時間 (秒)')
    ax1.set_ylabel('EEG Amplitude', color='black')
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Stimulus信号を第2Y軸でプロット
    ax2 = ax1.twinx()
    ax2.plot(plot_df['Time_s'], plot_df['Stimulus'], label='Stimulus (Raw)', color='red', linestyle='--')
    ax2.set_ylabel('Stimulus Intensity', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(-1, max(df['Stimulus'].max(), 10))

    # 仕上げ
    plt.title(f'イベントID: {event_id} ({stim_type}) の波形（前後{padding_ms}msを含む）')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    fig.tight_layout()
    plt.show()