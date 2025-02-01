from websocietysimulator.agent.modules.planning_modules import PlanningBase

class PlanningBaseline(PlanningBase):
    """Inherit from PlanningBase"""

#   "user_id": "AF6HG5ZKZR4N2BTWJ7FGMXZO3JEQ",
#   "item_id": "B09LLZN3V3"

    def __init__(self, llm):
        """Initialize the planning module"""
        super().__init__(llm=llm)

    def __call__(self, task_description):
        """Override the parent class's __call__ method"""
        self.plan = [
            {
                'description': 'First I need to find user information',
                'reasoning instruction': 'None',
                'tool use instruction': {task_description['AF6HG5ZKZR4N2BTWJ7FGMXZO3JEQ']}
            },
            {
                'description': 'Next, I need to find business information',
                'reasoning instruction': 'None',
                'tool use instruction': {task_description['B09LLZN3V3']}
            }
        ]
        return self.plan
    
# 创建类的实例
my_instance =  PlanningBaseline(llm,any)
# 定义任务描述
task_description = {
    'AF6HG5ZKZR4N2BTWJ7FGMXZO3JEQ': '获取用户信息的工具指令',
    'B09LLZN3V3': '获取商业信息的工具指令'
}

# 调用实例
plans = my_instance(task_description)  # 这里会打印 self.plan
