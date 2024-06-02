import numpy as np
from module.model_manager import ModelManager
from module.dataset_preprocess import actions
from module.redis_manager import RedisManager
from config import REDIS_PASSWORD

if __name__ == "__main__":
    # 创建redis管理器
    redis_manager = RedisManager(host='127.0.0.1', port=6379, password=REDIS_PASSWORD)

    # 获取数据
    data = redis_manager.get('my_key')
    print("Retrieved data:", data)