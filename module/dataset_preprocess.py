import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

actions = ['load', 'start', 'pause', 'resume', 'land', 'goHome', 'speedHeightOption', 'sprayOption', 'spreadOption', 'emptyingControl', 'OaEnableOption', 'HeightSourceOption']


def preprocess_data(file_path):
    """
    对数据进行预处理
    
    参数:
    file_path: 数据文件路径
    
    返回:
    X_train: 训练集输入特征
    y_train: 训练集标签
    X_val: 验证集输入特征
    y_val: 验证集标签
    X_dim: 输入维度
    y_dim: 输出纬度
    """
    # 读取数据
    data = pd.read_csv(file_path)

    # 数据处理
    tps2_data = pd.json_normalize(data["tps2_data"].apply(json.loads))

    # 提取所需特征
    X = pd.DataFrame()  # 创建特征 DataFrame
    # X["time"] = tps2_data["system.time"]
    X["height"] = tps2_data["motioninfo.height"].fillna(0)
    X["speed"] = tps2_data["motioninfo.speed"].fillna(0)
    X["oa_enable"] = tps2_data["motioninfo.oa_enable"].fillna(0).apply(lambda x: 1 if x else 0)
    X["height_source"] = tps2_data["motioninfo.height_source"].fillna(0)
    X["mission_mode"] = tps2_data["system.mission_mode"].fillna(0)
    X["platform_mode"] = tps2_data["system.platform_mode"].fillna(0)
    X["wp_index"] = tps2_data["motionstatus.wp_index"].fillna(0)
    # X["flag"] = tps2_data["motionstatus.current_wp.flag"]        # current_wp是数组，待解决
    X["wp_segment"] = tps2_data["motionstatus.wp_segment"].fillna(0)
    X["work_count"] = tps2_data["motionstatus.work_count"].fillna(0)

    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 提取标签
    y = [next((actions.index(act) for act in actions if act in action), -1) for action in data["app_cmd"]]

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_dim = X_train.shape[1]
    y_dim = len(actions)

    return X_train, y_train, X_val, y_val, X_dim, y_dim

# 使用示例
if __name__ == "__main__":
    file_path = "dataset/train.csv"
    X_train, y_train, X_val, y_val, X_dim, y_dim = preprocess_data(file_path)
    print(X_train)
    print(y_train)
    print(f"X_dim:{X_dim}")
    print(f"y_dim:{y_dim}")
