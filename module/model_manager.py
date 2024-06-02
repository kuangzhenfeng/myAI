import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from module.dataset_generator import generate_data
from module.dataset_preprocess import preprocess_data

class ModelManager:
    def __init__(self):
        self.model = None
        self.X_dim = None
        self.y_dim = None

    def create_model(self, X_dim, y_dim):
        """
        创建神经网络模型

        参数:
        X_dim: 输入维度
        y_dim: 输出纬度

        返回:
        model: 创建的神经网络模型
        """
        input_layer = layers.Input(shape=(X_dim,))
        hidden_layer_1 = layers.Dense(64, activation='relu')(input_layer)
        hidden_layer_2 = layers.Dense(32, activation='relu')(hidden_layer_1)
        hidden_layer_3 = layers.Dense(16, activation='relu')(hidden_layer_2)
        output_layer = layers.Dense(y_dim, activation='softmax')(hidden_layer_3)
        model = models.Model(inputs=input_layer, outputs=output_layer)
        return model

    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, patience=3, save_path="model/model.keras"):
        """
        训练神经网络模型

        参数:
        X_train: 训练集输入特征
        y_train: 训练集标签
        X_val: 验证集输入特征
        y_val: 验证集标签
        epochs: 最大训练轮数
        patience: EarlyStopping的等待轮数
        save_path: 模型保存路径
        """
        # 设置 EarlyStopping 回调
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

        # 保存模型
        self.model.save(save_path)
        print("Model saved successfully.")

    def load_model(self, model_path):
        """
        加载模型

        参数:
        model_path: 模型文件路径
        """
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")

    def predict_action(self, input_data):
        """
        使用模型预测动作

        参数:
        input_data: 输入数据

        返回:
        actions: 预测的动作
        """
        predictions = self.model.predict(input_data)
        actions = np.argmax(predictions, axis=1)
        return actions

    def load_data(self, file_path):
        """
        加载训练集和验证集数据

        参数:
        file_path: 数据文件路径

        返回:
        X_train: 训练集输入特征
        y_train: 训练集标签
        X_val: 验证集输入特征
        y_val: 验证集标签
        """
        # 预处理数据
        X_train, y_train, X_val, y_val, self.X_dim, self.y_dim = preprocess_data(file_path)

        # 生成训练集
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=self.y_dim)

        # 生成验证集
        y_val = tf.keras.utils.to_categorical(y_val, num_classes=self.y_dim)

        return X_train, y_train, X_val, y_val

    def load_random_data(self, train_size=1000, val_size=200):
        """
        加载随机生成的训练集和验证集数据

        参数:
        train_size: 训练集大小
        val_size: 验证集大小

        返回:
        X_train: 训练集输入特征
        y_train: 训练集标签
        X_val: 验证集输入特征
        y_val: 验证集标签
        """
        # 生成训练集
        X_train, y_train = generate_data(train_size)
        y_train = tf.keras.utils.to_categorical(y_train)

        # 生成验证集
        X_val, y_val = generate_data(val_size)
        y_val = tf.keras.utils.to_categorical(y_val)

        self.X_dim = X_train.shape[1]
        self.y_dim = y_train.shape[1]

        return X_train, y_train, X_val, y_val

