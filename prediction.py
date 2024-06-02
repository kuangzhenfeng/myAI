import numpy as np
from module.model_manager import ModelManager
from module.dataset_preprocess import actions

def model_prediction():
    # 创建模型管理器
    manager = ModelManager()

    # 加载保存的模型
    manager.load_model("model/model.keras")

    # 使用模型进行预测
    X_pre, y_pre, _, _ = manager.load_data("dataset/prediction.csv")
    predictions = manager.predict_action(X_pre)

    # 输出所有相对比
    print("Predictions\tActual Labels")
    correct_count = 0
    for i in range(len(predictions)):
        prediction = predictions[i]
        actual_label = np.argmax(y_pre[i])
        print(f"{actions[prediction]}\t\t{actions[actual_label]}")
        if prediction == actual_label:
            correct_count += 1

    # 输出错误项
    print("\nIncorrect Predictions:")
    for i in range(len(predictions)):
        prediction = predictions[i]
        actual_label = np.argmax(y_pre[i])
        if prediction != actual_label:
            print(f"Prediction: {actions[prediction]}, Actual: {actions[actual_label]}")

    # 计算并输出正确率
    accuracy = correct_count / len(predictions)
    print(f"\nAccuracy: {accuracy}")

if __name__ == "__main__":
    model_prediction()