from typing import Any, Dict, List, Optional

class RecommendationTask:
    def __init__(self, user_id: str,  
                 candidate_category: str,
                 candidate_list: List[str],
                 loc: List[float]):
        """
        Recommendation Task for the RecommendationAgent.
        Args:
            user_id: The ID of the user requesting recommendations.
            candidate_category: The category of the candidate items.
            candidate_list: List of candidate item IDs.
            loc: User's location as [latitude, longitude]. If is [-1, -1], the user is not in a specific location.
        """
        self.user_id = user_id
        self.candidate_category = candidate_category
        self.candidate_list = candidate_list
        self.loc = loc

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the task to a dictionary.
        Returns:
            dict: The task in dictionary format.
        """
        return {
            "description": """This is a recommendation task. 
            You are a recommendation agent that recommends items to users. 
            There is a user with id and a list of items with category and ids. 
            The location of the user is set only if it is not [-1, -1].""",
            "user_id": self.user_id,
            "candidate_category": self.candidate_category,
            "candidate_list": self.candidate_list,
            "loc": self.loc
        }
    

# RecommendationTask 用于表示推荐任务，通常由推荐代理执行，以便向用户推荐合适的项目或商品。
# 该任务包含用户的详细信息、候选项目的类别及其 ID 列表，以及用户的位置。
# 主要特点
# 用户 ID (user_id): 识别请求推荐的用户。
# 候选项目类别 (candidate_category): 指明候选项目的分类。这可以帮助推荐代理在特定领域内进行筛选和推荐。
# 候选项目列表 (candidate_list): 包含可以推荐给用户的项目 ID 列表。这个列表是推荐过程中使用的基础数据。
# 用户位置 (loc): 用户的地理位置，通常用于提供基于位置的推荐。如果位置被设置为 [-1, -1]，表示用户没有特定的位置信息。
# 方法
# to_dict(): 将任务信息转换为字典格式，方便传输或记录。返回的字典包括任务描述、用户 ID、候选类别、候选列表及用户位置。