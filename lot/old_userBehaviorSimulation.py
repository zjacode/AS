# import sys
# print(sys.path)

from websocietysimulator.agent import SimulationAgent
from websocietysimulator.llm import LLMBase
from websocietysimulator.agent.modules.planning_modules import PlanningBase
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
from websocietysimulator.agent.modules.memory_modules import MemoryDILU

import logging

user_id = "AF6HG5ZKZR4N2BTWJ7FGMXZO3JEQ"
item_id = "B09LLZN3V3"

class PlanningBaseline(PlanningBase):
    """Inherit from PlanningBase"""

    print(PlanningBase)
    
    def __init__(self, llm):
        """Initialize the planning module"""
        super().__init__(llm=llm)


    def __call__(self, task_description):
        """Override the parent class's __call__ method"""
        self.plan = [
            {
                'description': 'First I need to find user information',
                'reasoning instruction': 'None',
                'tool use instruction': {task_description['user_id']}
            },
            {
                'description': 'Next, I need to find business information',
                'reasoning instruction': 'None',
                'tool use instruction': {task_description['item_id']}
            }
        ]
        print(self.plan)
        return self.plan    
       

  #  返回一个包含两个字典的列表

class ReasoningBaseline(ReasoningBase):
    """Inherit from ReasoningBase"""

    def __init__(self, profile_type_prompt, llm):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)

    def __call__(self, task_description: str):
        """Override the parent class's __call__ method"""
        prompt = '''
{task_description}'''
        prompt = prompt.format(task_description=task_description)

        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.0,
            max_tokens=1000
        )

        return reasoning_result
    print("4")

class MySimulationAgent(SimulationAgent):
    """Participant's implementation of SimulationAgent."""
    
    def __init__(self, llm: LLMBase):
        """Initialize MySimulationAgent"""
        super().__init__(llm=llm)
        self.planning = PlanningBaseline(llm=self.llm)
        self.reasoning = ReasoningBaseline(profile_type_prompt='', llm=self.llm)
        self.memory = MemoryDILU(llm=self.llm)

    def workflow(self):
        
        print("5")

        """
        Simulate user behavior
        Returns:
            tuple: (star (float), useful (float), funny (float), cool (float), review_text (str))
        """
        try:
            plan = self.planning(task_description=self.task)

            for sub_task in plan:
                if 'user' in sub_task['description']:
                    user = str(self.interaction_tool.get_user(user_id=self.task['user_id']))
                elif 'business' in sub_task['description']:
                    business = str(self.interaction_tool.get_item(item_id=self.task['item_id']))
            reviews_item = self.interaction_tool.get_reviews(item_id=self.task['item_id'])
            for review in reviews_item:
                review_text = review['text']
                self.memory(f'review: {review_text}')
            reviews_user = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
            review_similar = self.memory(f'{reviews_user[0]["text"]}')
            task_description = f'''
            You are a real human user on Yelp, a platform for crowd-sourced business reviews. Here is your Yelp profile and review history: {user}

            You need to write a review for this business: {business}

            Others have reviewed this business before: {review_similar}

            Please analyze the following aspects carefully:
            1. Based on your user profile and review style, what rating would you give this business? Remember that many users give 5-star ratings for excellent experiences that exceed expectations, and 1-star ratings for very poor experiences that fail to meet basic standards.
            2. Given the business details and your past experiences, what specific aspects would you comment on? Focus on the positive aspects that make this business stand out or negative aspects that severely impact the experience.
            3. Consider how other users might engage with your review in terms of:

            - Useful: How informative and helpful is your review?
            - Funny: Does your review have any humorous or entertaining elements?
            - Cool: Is your review particularly insightful or praiseworthy?                
                有用性: 你的评论信息量多么，帮助性如何？
                幽默感: 你的评论是否有幽默或娱乐的元素？
                好评: 你的评论是否特别有见地或值得赞扬？

            Requirements:
            - Star rating must be one of: 1.0, 2.0, 3.0, 4.0, 5.0
            - If the business meets or exceeds expectations in key areas, consider giving a 5-star rating
            - If the business fails significantly in key areas, consider giving a 1-star rating
            - Review text should be 2-4 sentences, focusing on your personal experience and emotional response
            - Useful/funny/cool counts should be non-negative integers that reflect likely user engagement
            - Maintain consistency with your historical review style and rating patterns
            - Focus on specific details about the business rather than generic comments
            - Be generous with ratings when businesses deliver quality service and products
            - Be critical when businesses fail to meet basic standards
             
              星级评分必须是以下选项之一: 1.0, 2.0, 3.0, 4.0, 5.0
              如果商家在关键领域满足或超出预期考虑给出5星评分。
              如果商家在关键领域显著失败,考虑给出1星评分。
              评论文字应为2-4句话.关注你的个人体验和情感反应。
              有用/幽默/好评的计数应为非负整数，反映用户可能的参与程度。
               保持与历史评论风格和评分模式的一致性。
              着眼于商家的具体细节，而不是通用评论。
              当商家提供优质的服务和产品时，给予慷慨的评分。

            Format your response exactly as follows:
            stars: [your rating]
            useful: [count]
            funny: [count]
            cool: [count]
            review: [your review]
            '''
            result = self.reasoning(task_description)

            try:
                stars_line = [line for line in result.split('\n') if 'stars:' in line][0]
                review_line = [line for line in result.split('\n') if 'review:' in line][0]
            except:
                print('Error:', result)

            stars = float(stars_line.split(':')[1].strip())
            review_text = review_line.split(':')[1].strip()

            if len(review_text) > 512:
                review_text = review_text[:512]

            return {
                "stars": stars,
                "review": review_text
            }
        except Exception as e:
            print(f"Error in workflow: {e}")
            return {
                "stars": 0,
                "review": ""
            } 
    
    print("6")

print("7")




# print("1")

# def workflow(self):
#         """
#         Simulate user behavior
#         Returns:
#             dict: {
#                 "stars": float,
#                 "review": str
#             }
#         """
#         try:
#             print("Starting the workflow.")

#             # 生成计划
#             print(f"Task description: {self.task}")
#             plan = self.planning(task_description=self.task)
#             print(f"Generated plan: {plan}")

#             # 执行子任务
#             for sub_task in plan:
#                 if 'user' in sub_task['description']:
#                     print("Fetching user information...")
#                     user = str(self.interaction_tool.get_user(user_id=self.task['user_id']))
#                     print(f"Retrieved user info: {user}")
#                 elif 'business' in sub_task['description']:
#                     print("Fetching business information...")
#                     business = str(self.interaction_tool.get_item(item_id=self.task['item_id']))
#                     print(f"Retrieved business info: {business}")

#             # 获取评论
#             print("Fetching reviews for the item...")
#             reviews_item = self.interaction_tool.get_reviews(item_id=self.task['item_id'])
#             for review in reviews_item:
#                 review_text = review['text']
#                 print(f"Storing item review: {review_text}")
#                 self.memory(f'review: {review_text}')

#             print("Fetching user reviews...")
#             reviews_user = self.interaction_tool.get_reviews(user_id=self.task['user_id'])
#             review_similar = self.memory(f'{reviews_user[0]["text"]}')
#             print(f"Retrieved similar user review: {review_similar}")

#             # 准备任务描述
#             task_description = f'''
#             You are a real human user on Yelp, a platform for crowd-sourced business reviews. Here is your Yelp profile and review history: {user}

#             You need to write a review for this business: {business}

#             Others have reviewed this business before: {review_similar}
#             ...
#             '''  # 省略标准文本，为简洁起见。

#             print("Requesting reasoning from LLM...")
#             result = self.reasoning(task_description)
#             print(f"Received reasoning result: {result}")

#             # 解析推理结果
#             try:
#                 stars_line = [line for line in result.split('\n') if 'stars:' in line][0]
#                 review_line = [line for line in result.split('\n') if 'review:' in line][0]
#                 print(f"Parsed stars line: {stars_line}")
#                 print(f"Parsed review line: {review_line}")
#             except IndexError:
#                 print('Error parsing result:', result)
#                 return {
#                     "stars": 0,
#                     "review": ""
#                 }

#             stars = float(stars_line.split(':')[1].strip())
#             review_text = review_line.split(':')[1].strip()
#             print(f"Final star rating: {stars}")
#             print(f"Final review text: {review_text[:60]}...")  # 只打印前60个字符

#             if len(review_text) > 512:
#                 review_text = review_text[:512]
#                 print("Truncated review text to 512 characters.")

#             return {
#                 "stars": stars,
#                 "review": review_text
#             }
#         except Exception as e:
#             print(f"Error in workflow: {e}")
#             return {
#                 "stars": 0,
#                 "review": ""
#             }

# print("2")