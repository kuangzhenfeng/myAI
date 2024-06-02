from module.model_manager import ModelManager

def model_train():
    # 创建模型管理器
    manager = ModelManager()

    # 加载数据
    X_train, y_train, X_val, y_val = manager.load_data("dataset/train.csv")

    # 创建模型
    manager.model = manager.create_model(manager.X_dim, manager.y_dim)

    # 训练模型
    manager.train_model(X_train, y_train, X_val, y_val, epochs=100, patience=3, save_path="model/model.keras")

if __name__ == "__main__":
    model_train()