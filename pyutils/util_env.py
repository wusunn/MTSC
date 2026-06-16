import os


class UtilEnv:
    envs: dict = None

    @staticmethod
    def get_env_dict() -> dict:
        """获取当前环境变量列表"""
        return dict(os.environ)

    @staticmethod
    def get_value(key, default=None):
        if UtilEnv.envs is None:
            UtilEnv.envs = UtilEnv.get_env_dict()
        return UtilEnv.envs.get(key, default)
