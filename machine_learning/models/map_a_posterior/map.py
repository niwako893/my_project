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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))) 
from utils.plot_utils import plot_classification_report_bar
from utils.plot_utils import plot_confusion_matrix_heatmap
mpl.rcParams['font.family'] = 'MS Gothic' 

df = pd.read_csv("../../data/diabetes/diabetes.csv")
df = df.dropna()  # 欠損を含む行をすべて削除


def main():
    # データ読み込みと前処理
    df = pd.read_csv("../../data/diabetes/diabetes.csv")
    df = df.dropna()

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=0
    )

    # MAP推定 ≒ Ridge分類器で学習
    model = RidgeClassifier(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    

    # 分析結果の表示
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    plot_classification_report_bar(
    report_dict,
    model_name="RidgeClassifier",
    class_labels=["0", "1"],
    title="分類評価指標",
    xlabel="モデル予測",
    ylabel="実際の診断結果"
    )

    plot_confusion_matrix_heatmap(
    y_test,
    y_pred,
    labels=["非糖尿病", "糖尿病"],
    title="混同行列（RidgeClassifier）",
    xlabel="モデル予測",
    ylabel="実際の診断結果"
    )



if __name__ == "__main__":
    try:
        main()
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
