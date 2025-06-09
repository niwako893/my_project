import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns #グラフ


mpl.rcParams['font.family'] = 'MS Gothic'  # 例：Windowsの場合


def plot_regression_line(x, y, y_pred, xlabel="", ylabel="", title="", xlim=None, ylim=None):
    plt.figure(figsize=(8, 6))

    # 散布図（実測値）
    plt.scatter(x, y, color="blue", alpha=0.6, label="実データ")

    # 回帰直線
    plt.plot(x, y_pred, color="red", linewidth=2, label="回帰直線")

    # 軸ラベル・タイトル
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)

    # 軸の範囲（指定があれば）
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    # 説明テキストを図中に表示
    x_text = min(x) + (max(x) - min(x)) * 0.05
    y_text = max(y) - (max(y) - min(y)) * 0.1
    plt.text(x_text, y_text,
             "● 青い点：実データ\n― 赤い線：回帰直線",
             fontsize=10, color="black",
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close() 


def plot_confusion_matrix_heatmap(y_true, y_pred, labels=None, title="混同行列", xlabel="予測ラベル", ylabel="実際のラベル"):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    cm = confusion_matrix(y_true, y_pred)
    if labels is None:
        labels = np.unique(y_true)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.close()




def plot_classification_report_bar(
    report_dict,
    model_name,
    class_labels=None,
    title="分類評価指標",
    xlabel="予測",
    ylabel="実際のラベル"
    ):
    """
    classification_reportのdictを受け取り、precision・recall・f1-scoreをクラスごとに棒グラフ表示する

    Parameters:
    - report_dict: classification_report(..., output_dict=True) で得られた辞書
    - class_labels: クラスラベル（指定しなければ自動で取得）
    - title: グラフのタイトル
    - model_name: モデル名の表示
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if class_labels is None:
        class_labels = [label for label in report_dict.keys()
                        if label not in ("accuracy", "macro avg", "weighted avg")]

    metrics = ['precision', 'recall', 'f1-score']
    data = {metric: [report_dict[label][metric] for label in class_labels] for metric in metrics}

    x = np.arange(len(class_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, data[metric], width, label=metric)

    ax.set_xlabel("クラスラベル")
    ax.set_ylabel("スコア")
    ax.set_title(f"{title} ({model_name})")
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_labels)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close()
