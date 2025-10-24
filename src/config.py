# src/config.py
from pathlib import Path

# --- プロジェクト全体の基準パス設定 ---
PROJECT_ROOT = Path(__file__).parent.parent 

# --- パス設定 ---
DATA_DIR = Path("G:/マイドライブ/ALL_EEG_pdch15") 
PROCESSED_DIR = Path("G:/マイドライブ/ALL_EEG_pdch15/processed") # 処理後のデータはローカルに保存でOK

# --- データ仕様に関する設定 ---
SAMPLING_RATE = 1000
RAW_CHANNEL_NAMES = ['PFC', 'PPC', 'A1', 'V1', 'Stimulus']
EEG_CHANNELS_TO_FILTER = ['PFC', 'PPC', 'A1', 'V1']

# --- バンドパスフィルタのパラメータ設定 ---
FILTER_LOWCUT = 1.0
FILTER_HIGHCUT = 40.0
FILTER_ORDER = 4