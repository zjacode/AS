from typing import Dict

class SimulationTask:
    def __init__(self, user_id: str, item_id: str):
        """
        Simulation Task for the SimulationAgent.
        Args:
            time: The time parameter to limit InteractionTool behavior.
            item: The item receiving the simulated review.
        """
        self.user_id = user_id
        self.item_id = item_id

    def to_dict(self) -> Dict[str, str]:
        """
        Convert the task to a dictionary.
        Returns:
            dict: The task in dictionary format.
        """
        return {
            "description": """This is a simulation task. 
            You are a simulation agent that simulates a user's rating and review with an item. 
            There is a user with id and an item with id. """,
            "user_id": self.user_id,
            "item_id": self.item_id
        }




# SimulationTask 用于表示模拟任务，通常由模拟代理执行，以模拟用户对某个项目的评分和评论。
# 该任务关注的是用户如何与特定项目互动，包括评分和评论的生成。
# 主要特点
# 用户 ID (user_id): 识别进行模拟的用户。
# 项目 ID (item_id): 被模拟的项目的标识符。这有助于模拟代理生成与该项目相关的反馈。
# 方法
# to_dict(): 将模拟任务转换为字典格式，便于记录和使用。返回的字典包含任务描述、用户 ID 和项目 ID。


