import redis

class RedisManager:
    def __init__(self, host='localhost', port=6379, password=None):
        self.host = host
        self.port = port
        self.password = password
        self.connection = None

    def connect(self):
        self.connection = redis.Redis(host=self.host, port=self.port, password=self.password)

    def set(self, key, value):
        if not self.connection:
            self.connect()
        self.connection.set(key, value)

    def get(self, key):
        if not self.connection:
            self.connect()
        return self.connection.get(key)

    def delete(self, key):
        if not self.connection:
            self.connect()
        self.connection.delete(key)

    def disconnect(self):
        if self.connection:
            self.connection.close()
            self.connection = None

# 示例用法
if __name__ == "__main__":
    # 创建redis管理器
    redis_manager = RedisManager(host='your_redis_host', port=6379, password='your_redis_password')

    # 设置数据
    redis_manager.set('my_key', 'my_value')

    # 获取数据
    data = redis_manager.get('my_key')
    print("Retrieved data:", data)

    # 删除数据
    redis_manager.delete('my_key')

    # 断开连接
    redis_manager.disconnect()
