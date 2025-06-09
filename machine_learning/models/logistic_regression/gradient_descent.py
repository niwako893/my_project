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

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../utils")))
from plot_utils import plot_regression_line
from plot_utils import plot_classification_report_bar
from plot_utils import plot_confusion_matrix_heatmap



df = pd.read_csv("data/survey lung cancer.csv")
df = df.dropna()  # 欠損を含む行をすべて削除

df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'NO': 0, 'YES': 1})
df['GENDER'] = df['GENDER'].map({'M': 0, 'F': 1})

#yes:1 no:0   M:0 F:1

# 特徴量とラベルに分離
X = df.drop(columns=['LUNG_CANCER'])
y = df['LUNG_CANCER'].values.reshape(-1, 1) 
#values.reshape(-1 (自動で行数を決定), 1 (列数を1に固定))


# 標準化(平均0, 標準偏差1に変換)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#全ての列に1を追加してバイアス項を作成(#np.onesでfloat型の1.を作成)
X_bias = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])


#目的: ある入力をK個のクラスに分類したい

#シグモイド関数
#予測値を0~1に変換するための関数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#エントロピー損失関数
#データが分類にどれくらい使えるか(大きい方が確実に使える)

def compute_loss(y_true, y_pred):
    epsilon = 1e-8  # logのゼロ除け
    return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

# yn+1 = yn - η∇E

# X: 全ての特徴量, y: cancer, lr: 学習率, epochs: エポック数(学習回数)

def gradient_descent(X, y, lr=0.1, epochs=1000):
    n_samples, n_features = X.shape
    weights = np.zeros((n_features, 1))  # 重みベクトル float型の0.

    losses = []  # 損失記録
    accuracies = [] 

    for epoch in range(epochs):
        z = X @ weights  # "全てのsampledata15個の"線形結合 （2つのベクトルの方向がどれくらい同じか計算したい）
        y_pred = sigmoid(z)

        error = y_pred - y #予測値との誤差ベクトル

        grad = X.T @ error / n_samples  # 誤差ベクトルと特徴量の内積の平均値を計算
                                        #損失関数の微分

        weights = weights - lr * grad  # 重みの更新
        loss = compute_loss(y, y_pred)
        losses.append(loss)


        y_pred_label = (y_pred >= 0.5).astype(int)
        acc = np.mean(y_pred_label == y)
        accuracies.append(acc)        

        #ログ
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")

    return weights, losses,accuracies
        
def predict(X, weights):
    print(x)
    z = X @ weights
    y_prob = sigmoid(z)  # 確率
    y_label = (y_prob >= 0.5).astype(int)  
    return y_label, y_prob

def logistic_model(X_input, weights):
    return sigmoid(X_input @ weights)

                                        
def main():
    weights, losses ,accuracies = gradient_descent(X_bias, y, lr=0.1, epochs=1000)

    # 予測
    z = X_bias @ weights
    y_pred_prob = sigmoid(z)        
    y_pred_label = (y_pred_prob >= 0.5).astype(int)

    # 分類精度
    #y_pred_prob    がんの予想確率(0.0~1.0の小数)
    # y_pred_label  0.5異常でがん (=1) 未満でがんじゃない (=0) と分類

    accuracy = np.mean(y_pred_label == y)
    print(f"\n分類精度: {accuracy * 100:.2f}%")


    friend_sample_values = [0, 20, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1]
    friend_sample = np.array(friend_sample_values)
    friend_sample_scaled = scaler.transform(friend_sample.reshape(1, -1))   # 標準化
    friend_sample_bias = np.hstack([np.ones((1, 1)), friend_sample_scaled]) # バイアス項追加

    friend_prob = logistic_model(friend_sample_bias, weights)
    friend_label = (friend_prob >= 0.5).astype(int)

    print(f"\n[診断結果]")
    print(f"肺がんの確率: {friend_prob[0][0] * 100:.2f}%")
    print(f"予測ラベル（1=がん）: {friend_label[0][0]}")
    print(pd.DataFrame(weights.T, columns=["バイアス"] + X.columns.tolist()))
    print("z =", friend_sample_bias @ weights)

    



    plt.plot(accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



    #損失関数
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 混同行列ヒートマップ
    plot_confusion_matrix_heatmap(
        y_true=y,
        y_pred=y_pred_label,
        labels=["肺がん", "肺がんじゃない"],
        title="混同行列",
        xlabel="モデル予測",
        ylabel="実際の診断結果"
    )

# 分類レポートを dict で取得
    report_dict = classification_report(y, y_pred_label, output_dict=True)


# 評価指標バーグラフ
    plot_classification_report_bar(
        report_dict=report_dict,
        class_labels=["0", "1"],  # または ["肺がん", "肺がんじゃない"]
        model_name="ロジスティック回帰",
        title="分類評価指標"
    )

# --- 実行ブロック ---
if __name__ == "__main__":
    try:
        main()


    except Exception as e:
        print(f"エラーが発生しました: {e}")