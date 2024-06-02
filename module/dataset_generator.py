import numpy as np
import random

# 定义字符串到数字的映射
action_map = {
    'Pause': 0,
    'Resume': 1,
    'Land': 2,
    'Return': 3,
    'Guide': 4,
    'Set Height': 5,
    'Set Speed': 6,
    'Set Obstacle': 7,
    'Set Terrain': 8
}

def generate_data(num_samples):
    X = []
    y = []
    for _ in range(num_samples):
        height = random.uniform(0, 100)
        speed = random.uniform(0, 10)
        obstacle = random.choice([True, False])
        terrain = random.uniform(0, 1)
        task_mode = random.choice([1, 2, 3])
        platform_mode = random.choice([1, 2, 3])
        segment = random.randint(0, 10)
        waypoint_count = random.randint(0, 20)
        current_waypoint = random.randint(0, waypoint_count)
        sn_hash = random.randint(0, 10000)
        timestamp = random.randint(0, 1000000)
        action_str = random.choice(['Pause', 'Resume', 'Land', 'Return', 'Guide', 'Set Height', 'Set Speed', 'Set Obstacle', 'Set Terrain'])
        action = action_map[action_str]  # 将字符串转换为数字
        X.append([height, speed, obstacle, terrain, task_mode, platform_mode, segment, waypoint_count, current_waypoint, sn_hash, timestamp])
        y.append(action)
    return np.array(X), np.array(y)
