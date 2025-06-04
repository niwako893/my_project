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
import pandas as pd #データ処理
import seaborn as sns #グラフ
import sklearn #機械学習
import pyperclip #クリップボード
from itertools import product #組み合わせ
import sympy as sp #数式処理,論理式
import pandas as pd


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  # ← machine_learning まで戻る
from utils.plot_utils import plot_regression_line

data1 = pd.read_csv("../../data/archive/Original_data_with_more_rows.csv")
df = pd.read_csv("../../data/archive/Expanded_data_with_more_features.csv")
df = df.dropna()  # 欠損を含む行をすべて削除


mapping = {
    "< 5": 2.5,
    "5 - 10": 7.5,
    "> 10": 12.5
}

# "WklyStudyHours"列の値を数値に変換
df["WklyStudyHours_num"] = df["WklyStudyHours"].map(mapping)

math_score = df["MathScore"]
reading_score = df["ReadingScore"]
study_hours = df["WklyStudyHours_num"]

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
    


#**i for i in range(self.M)	X^0, X^1, X^2, ..., X^(M-1) のリストを作る
#np.array([...]).T	配列を2次元化し、転置して (データ数, M) の形に
#X.flatten()	入力 X を1次元に変換（例：列ベクトル → フラットな配列）

# --- メイン関数 ---
def main():
    a, b = linear_function(study_hours, math_score)
    y_pred = a * study_hours + b  # 回帰直線の予測値を計算

    # SSEの計算
    sse = cal_sse(study_hours, math_score, a, b)
    print(f"SSE（二乗和誤差）: {sse:.2f}")

    plot_regression_line(
        x=study_hours,
        y=math_score,
        y_pred=y_pred,
        xlabel="WklyStudyHours",
        ylabel="MathScore",
        title="数学の成績と学習時間の関係",
        xlim=(0, 15),
        ylim=(50, 90)
                )
    


# --- 実行ブロック ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"エラーが発生しました: {e}")
