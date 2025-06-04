#sklearnの線形回帰モデルを使った最小二乗法の実装

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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))  
from utils.plot_utils import plot_regression_line

# フォント設定（日本語が表示されるように）
mpl.rcParams['font.family'] = 'MS Gothic'  # 例：Windowsの場合
# mpl.rcParams['font.family'] = 'IPAexGothic'  # 例：Linux/インストール済みの日本語フォント

#linear_regrssionのフォルダ内のパスで実行する

#--------------------------------------------------------

df = pd.read_csv("../../data/study_hours/score.csv")
tdf = pd.read_csv("../../data/study_hours/score.csv")

df = df.dropna()  # 欠損を含む行をすべて削除


#1次元配列
scores= df["Scores"]
hours = df["Hours"]

x = hours.values.reshape(-1, 1)  # (n_samples, 1)
y = scores.values  # (n_samples,) → 1次元でOK

# 線形回帰モデル作成・学習
model = LinearRegression()
model.fit(x, y)


# --- メイン関数 ---
def main():
    y_pred = model.predict(x)


    plot_regression_line(
        x=x,
        y=y,
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
