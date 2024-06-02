# 建立模型，操作推荐，使用MLP或RNN
# 入参：高度，速度，避障，仿地，任务模式，平台工作模式，航段，航点总数，当前执行航点，sn(hash)，时间戳
# 出参：操作（暂停，继续，迫降，返航，指点飞行，设置高度，设置速度，设置避障，设置仿地）

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from dataset_generator import generate_data

def create_model(input_shape):
    """
    创建神经网络模型
    
    参数:
    input_shape: 输入数据的形状
    
    返回:
    model: 创建的神经网络模型
    """
    input_layer = layers.Input(shape=input_shape)
    hidden_layer_1 = layers.Dense(64, activation='relu')(input_layer)
    hidden_layer_2 = layers.Dense(32, activation='relu')(hidden_layer_1)
    hidden_layer_3 = layers.Dense(16, activation='relu')(hidden_layer_2)
    output_layer = layers.Dense(9, activation='softmax')(hidden_layer_3)
    model = models.Model(inputs=input_layer, outputs=output_layer)
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, patience=3):
    """
    训练神经网络模型
    
    参数:
    model: 神经网络模型
    X_train: 训练集输入特征
    y_train: 训练集标签
    X_val: 验证集输入特征
    y_val: 验证集标签
    epochs: 最大训练轮数
    patience: EarlyStopping的等待轮数
    """
    # 设置 EarlyStopping 回调
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

def load_data():
    """
    加载训练集和验证集数据
    
    返回:
    X_train: 训练集输入特征
    y_train: 训练集标签
    X_val: 验证集输入特征
    y_val: 验证集标签
    """
    # 生成训练集
    X_train, y_train = generate_data(1000)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=9)

    # 生成验证集
    X_val, y_val = generate_data(200)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=9)

    return X_train, y_train, X_val, y_val

def predict_action(model, input_data):
    """
    使用模型预测动作
    
    参数:
    model: 训练好的神经网络模型
    input_data: 输入数据
    
    返回:
    actions: 预测的动作
    """
    predictions = model.predict(input_data)
    actions = np.argmax(predictions, axis=1)
    return actions

if __name__ == "__main__":
    # 加载数据
    X_train, y_train, X_val, y_val = load_data()

    # 获取输入数据维度
    input_dimension = X_train.shape[1]

    # 创建模型
    input_shape = (input_dimension,)
    model = create_model(input_shape)

    # 训练模型
    train_model(model, X_train, y_train, X_val, y_val, 100, 3)

    # 使用模型进行预测
    input_data = np.random.randn(1, input_dimension)  # 输入数据示例
    action = predict_action(model, input_data)
    print(f'Predicted Action: {action}')
