

    

    # 分析結果の表示
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    plot_classification_report_bar(plot_classification_report_bar(report_dict, class_labels=None, title="分類評価指標", title="混同行列",model_name, xlabel="予測ラベル", ylabel="実際のラベル"):

    plot_confusion_matrix_heatmap(
    y_test,
    y_pred,
    labels=["非糖尿病", "糖尿病"],
    title="混同行列（RidgeClassifier）",
    xlabel="モデル予測",
    ylabel="実際の診断結果"
    )