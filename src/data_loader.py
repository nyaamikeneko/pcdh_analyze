import numpy as np
from .config import DATA_DIR # .は同じ階層にあるファイルという意味

def load_eeg_file(filename: str):
    """指定されたファイル名の脳波データを読み込み、脳波と刺激に分離して返す"""
    file_path = DATA_DIR / filename
    data = np.load(file_path)
    eeg = data[:4, :]
    stim = data[4, :]
    return eeg, stim