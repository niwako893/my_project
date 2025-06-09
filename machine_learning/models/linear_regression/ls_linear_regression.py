# ===========================================

# --- ライブラリのインポート（必要に応じて追加） ---
import math
import random
import time
import datetime
import sys
import os
import string
import numpy as np   #計算
import matplotlib.pyplot as plt  #グラフ
import matplotlib as mpl
import pandas as pd #データ処理
import seaborn as sns #グラフ
import sklearn #機械学習
import pyperclip #クリップボード
from itertools import product #組み合わせ
import sympy as sp #数式処理,論理式
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils")))
from plot_utils import plot_regression_line

# フォント設定（日本語が表示されるように）
mpl.rcParams['font.family'] = 'MS Gothic'  # 例：Windowsの場合
# mpl.rcParams['font.family'] = 'IPAexGothic'  # 例：Linux/インストール済みの日本語フォント



df = pd.read_csv("data/study_hours/score.csv")
tdf = pd.read_csv("data/study_hours/score.csv")
df = df.dropna()  # 欠損を含む行をすべて削除


scores= df["Scores"]
hours = df["Hours"]

#二乗和誤差の傾きと切片を計算する
def linear_function(x, y):
    numerator = np.sum((x - np.mean(x)) * (y - np.mean(y)))
    denominator = np.sum((x - np.mean(x)) ** 2)

    slope = numerator / denominator
    intercept = np.mean(y) - slope * np.mean(x)
    return slope, intercept

def cal_sse(x, y, slope, intercept):
    y_pred = slope * x + intercept
    sse = np.sum((y_pred - y) ** 2)
    return sse
    



# --- メイン関数 ---
def main():
    a, b = linear_function(hours, scores)
    y_pred = a * hours + b  # 回帰直線の予測値を計算

    # SSEの計算
    sse = cal_sse(hours, scores, a, b)
    print(f"SSE（二乗和誤差）: {sse:.2f}")

    plot_regression_line(
        x=hours,
        y=scores,
        y_pred=y_pred,
        xlabel="StudyHours",
        ylabel="Score",
        title="成績と学習時間の関係",
        xlim=(0, 10),
        ylim=(0, 150)
                )
    


# --- 実行ブロック ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"エラーが発生しました: {e}")
