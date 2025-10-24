# src/config.py
from pathlib import Path
import os

# --- プロジェクト全体の基準パス設定 ---
PROJECT_ROOT = Path(__file__).parent.parent

# --- パス設定 ---
# 環境変数があれば優先して使い、なければプロジェクト内の data フォルダを参照
DATA_DIR = Path(os.environ.get("EEG_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
PROCESSED_DIR = Path(os.environ.get("EEG_PROCESSED_DIR", PROJECT_ROOT / "data" / "processed"))
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --- データ仕様に関する設定 ---
SAMPLING_RATE = int(os.environ.get("EEG_SAMPLING_RATE", 1000))
RAW_CHANNEL_NAMES = ['PFC', 'PPC', 'A1', 'V1', 'Stimulus']
EEG_CHANNELS_TO_FILTER = ['PFC', 'PPC', 'A1', 'V1']

# --- バンドパスフィルタのパラメータ設定 ---
FILTER_LOWCUT = float(os.environ.get("EEG_FILTER_LOWCUT", 1.0))
FILTER_HIGHCUT = float(os.environ.get("EEG_FILTER_HIGHCUT", 40.0))
FILTER_ORDER = int(os.environ.get("EEG_FILTER_ORDER", 4))

# --- 刺激／イベント関連のデフォルト設定 ---
# 刺激検出の閾値 (デフォルト: Stimulus > 5 を想定していた箇所に対応)
STIMULUS_THRESHOLD = float(os.environ.get("EEG_STIMULUS_THRESHOLD", 5.0))

# イベントIDの範囲を名前で定義しておく (柔軟に変更できるように辞書で管理)
EVENT_RANGES = {
	'Light': (1, 600),
	'Sound': (601, 1200),
	'Light+Sound': (1201, 1800),
}

# --- 解析パラメータ ---
FREQS = list(range(4, 40))  # 4Hz〜39Hz
TMIN = -0.5
TMAX = 1.5
BASELINE = (-0.4, -0.1)

# MNE / 並列化等のデフォルト
N_JOBS = int(os.environ.get("EEG_N_JOBS", -1))

# Plot / display defaults
DEFAULT_FIGSIZE = (8, 6)

# 周波数帯の名前定義（可読性のためのラベルと周波数レンジ）
FREQ_BANDS = {
	'Theta (4-8 Hz)': (4, 8),
	'Alpha (8-13 Hz)': (8, 13),
	'Beta (13-30 Hz)': (13, 30),
	'Gamma (30-45 Hz)': (30, 45)
}