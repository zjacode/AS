import json
from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent
import tiktoken
from websocietysimulator.llm import LLMBase, InfinigenceLLM
import numpy as np
from typing import List, Dict, Any
import torch
from tqdm import tqdm
import time
from websocietysimulator.tools.cache_interaction_tool import CacheInteractionTool
from datetime import datetime
import logging
import os
import lmdb

class MemoryModule:
    """记忆模块：存储和管理不同来源的数据"""
    def __init__(self):
        self.user_memory = {}  # 用户历史行为记忆
        self.item_memory = {}  # 单独又添加了商品特征记忆（zmq2.2.3）
        self.review_memory = {}  # 评论模式记忆
        self.source_models = {}  # 不同来源的模型(zmq1.1.2)
        self.user_preferences = {}  # 用户偏好缓存
        self.item_features = {}     # 商品特征缓存(zmq2.2.3)

    def add_user_memory(self, user_id: str, behavior: Dict):
        if user_id not in self.user_memory:
            self.user_memory[user_id] = []
        self.user_memory[user_id].append(behavior)

    def add_item_memory(self, item_id: str, features: Dict):
        self.item_memory[item_id] = features

    def add_review_memory(self, source: str, review_data: Dict):
        if source not in self.review_memory:
            self.review_memory[source] = []
        self.review_memory[source].append(review_data)

    def get_source_specific_data(self, source: str) -> Dict:
        return {
            'users': {k: v for k, v in self.user_memory.items() if v.get('source') == source},
            'items': {k: v for k, v in self.item_memory.items() if v.get('source') == source},
            'reviews': self.review_memory.get(source, [])
        }

    def update_user_preference(self, user_id, item_id, rating):
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {'liked': [], 'disliked': []}
        
        if rating >= 4:
            self.user_preferences[user_id]['liked'].append(item_id)
        elif rating <= 2:
            self.user_preferences[user_id]['disliked'].append(item_id)

class PlanningModule:
    """规划模块：制定推荐策略"""
    def __init__(self, llm):
        self.llm = llm

    def create_recommendation_plan(self, user_id: str, memory_module: MemoryModule) -> List[Dict]:
        user_history = memory_module.user_memory.get(user_id, [])
        
        plan_prompt = f"""基于用户 {user_id} 的历史行为，制定推荐计划：
        1. 分析用户历史偏好
        2. 确定商品匹配策略
        3. 生成推荐候选列表
        用户历史: {json.dumps(user_history, ensure_ascii=False)}
        """
        
        response = self.llm.generate(plan_prompt)
        return self._parse_plan(response)

    def _parse_plan(self, response: str) -> List[Dict]:
        # 浅浅解析LLM返回的计划
        try:
            return json.loads(response)
        except:
            return []

class ReasoningModule:
    """推理模块：执行推荐决策"""
    def __init__(self, llm):
        self.llm = llm

    def reason_recommendation(self, 
                            user_id: str, 
                            candidate_items: List[str],
                            memory_module: MemoryModule) -> str:
        user_history = memory_module.user_memory.get(user_id, [])
        items_info = {item_id: memory_module.item_memory.get(item_id, {}) 
                     for item_id in candidate_items}
        
        reasoning_prompt = f"""基于以下信息为用户 {user_id} 选择最佳商品：
        用户历史: {json.dumps(user_history, ensure_ascii=False)}
        候选商品: {json.dumps(items_info, ensure_ascii=False)}
        
        请直接返回最合适的商品ID。
        """
        
        return self.llm.generate(reasoning_prompt).strip()

class TrainingMonitor:
    """训练监控模块：监控和报告训练进度"""
    def __init__(self):
        self.metrics = {
            'total_samples': 0,
            'processed_samples': 0,
            'current_batch': 0,
            'total_batches': 0,
            'source_metrics': {},
            'current_source': '',
            'training_history': []
        }
        self.start_time = None

    def start_training(self, total_samples: int, batch_size: int, source: str):
        """开始训练监控"""
        self.start_time = time.time()
        self.metrics['total_samples'] = total_samples
        self.metrics['total_batches'] = total_samples // batch_size
        self.metrics['current_source'] = source
        print(f"\n开始训练 {source} 数据源:")
        print(f"总样本数: {total_samples}")
        print(f"总批次数: {self.metrics['total_batches']}")
        print(f"批次大小: {batch_size}")

    def update_batch_progress(self, batch_idx: int, batch_metrics: Dict, batch_size: int):
        """更新批次进度
        Args:
            batch_idx: 当前批次索引
            batch_metrics: 当前批次的训练指标
            batch_size: 批次大小
        """
        current_batch = batch_idx + 1
        self.metrics['current_batch'] = current_batch
        
        # 浅算一下实际处理的样本数
        processed_samples = min(
            current_batch * batch_size,  
            self.metrics['total_samples']  # 总样本数上限
        )
        self.metrics['processed_samples'] = processed_samples
        
        # 实际进度
        progress = processed_samples / self.metrics['total_samples']
        
        # 时间
        elapsed_time = time.time() - self.start_time
        estimated_total = elapsed_time / progress if progress > 0 else 0
        remaining_time = estimated_total - elapsed_time
        
        # 更新源数据指标嘿嘿（zmq2.2.7）
        self.metrics['source_metrics'][self.metrics['current_source']] = batch_metrics
        
        print(f"\r进度: {processed_samples}/{self.metrics['total_samples']} 样本"
              f"({progress:.1%}) | "
              f"批次: {current_batch}/{self.metrics['total_batches']} | "
              f"已用时间: {elapsed_time:.1f}s | "
              f"预计剩余: {remaining_time:.1f}s | "
              f"损失: {batch_metrics.get('loss', 'N/A'):.4f} | "
              f"准确率: {batch_metrics.get('accuracy', 'N/A'):.2%}", 
              end='')

    def log_epoch_metrics(self, epoch_metrics: Dict):
        """记录每轮训练的指标"""
        self.metrics['training_history'].append(epoch_metrics)
        print(f"\n\n{self.metrics['current_source']} 训练轮次完成:")
        for metric_name, value in epoch_metrics.items():
            print(f"{metric_name}: {value:.4f}")

    def get_training_summary(self) -> Dict:
        """获取训练总结"""
        return {
            'total_time': time.time() - self.start_time,
            'final_metrics': self.metrics['source_metrics'],
            'training_history': self.metrics['training_history']
        }

class BatchTrainer:
    """批量训练管理器"""
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.monitor = TrainingMonitor()

    def train_source_model(self, source_data: Dict, memory_module: MemoryModule, source: str):
        """针对特定来源的数据进行批量训练"""
        total_samples = len(source_data['reviews'])
        self.monitor.start_training(total_samples, self.batch_size, source)
        
        # 计算实际的总批次数（记得要考虑不能整除的情况，在1.1.2版本中就忘记了···）
        total_batches = (total_samples + self.batch_size - 1) // self.batch_size
        
        epoch_metrics = {
            'total_loss': 0,
            'total_accuracy': 0,
            'processed_batches': 0
        }
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, total_samples)  # 防止越界
            
            # 获取当前批次实际大小
            current_batch_size = batch_end - batch_start
            
            batch_data = {
                'reviews': source_data['reviews'][batch_start:batch_end],
                'users': {k: v for k, v in source_data['users'].items() 
                         if any(r['user_id'] == k for r in source_data['reviews'][batch_start:batch_end])},
                'items': {k: v for k, v in source_data['items'].items() 
                         if any(r['item_id'] == k for r in source_data['reviews'][batch_start:batch_end])}
            }
            
            # 训练批次并获取指标
            batch_metrics = self._train_batch(batch_data, memory_module, source)
            
            # 更新进度和指标（其实是为了传入实际的batch_size）
            self.monitor.update_batch_progress(
                batch_idx, 
                batch_metrics,
                current_batch_size
            )
            
            # 根据实际处理的样本数更新累积指标
            epoch_metrics['total_loss'] += batch_metrics.get('loss', 0) * current_batch_size
            epoch_metrics['total_accuracy'] += batch_metrics.get('accuracy', 0) * current_batch_size
            epoch_metrics['processed_batches'] += current_batch_size

        # 计算加权平均的轮次指标
        avg_epoch_metrics = {
            'avg_loss': epoch_metrics['total_loss'] / total_samples,
            'avg_accuracy': epoch_metrics['total_accuracy'] / total_samples
        }
        self.monitor.log_epoch_metrics(avg_epoch_metrics)
        
        return self.monitor.get_training_summary()

    def _train_batch(self, 
                    batch_data: Dict, 
                    memory_module: MemoryModule,
                    source: str):
        """训练单个批次"""
        # 这里实现具体的训练逻辑
        pass

class MyRecommendationAgent(RecommendationAgent):
    def __init__(self, llm=None, data_dir=None):
        super().__init__(llm)
        self.cache_tool = None
        self.user_preferences = {'amazon': {}, 'yelp': {}, 'goodreads': {}}
        if data_dir:
            self.cache_tool = CacheInteractionTool(data_dir)
            self.data_dir = data_dir
            # 初始化LMDB环境
            self.lmdb_dir = os.path.join(data_dir, 'lmdb')
        self.memory = MemoryModule()
        self.planning = PlanningModule(llm)
        self.reasoning = ReasoningModule(llm)
        self.trainer = BatchTrainer()
        self.training_summaries = {}
        self.setup_logging()

    def setup_logging(self):
        """配置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('rec_agent.log'),
                logging.StreamHandler()
            ]
        )

    def _get_source_from_item(self, item_info):
        """根据商品信息判断数据来源"""
        if 'asin' in item_info:
            return 'amazon'
        elif 'business_id' in item_info:
            return 'yelp'
        elif 'book_id' in item_info:
            return 'goodreads'
        return None
    
    def train_on_cached_data(self, batch_size=1000):
        """使用缓存工具训练模型"""
        try:
            if not self.cache_tool:
                return {'status': 'error', 'message': '缓存工具未初始化'}
            
            print("\n=== 开始训练缓存数据 ===")
            
            # 使用新方法获取所有用户ID
            user_ids = self.cache_tool.get_all_user_ids()
            if not user_ids:
                return {'status': 'error', 'message': '未找到用户数据'}
            
            print(f"找到 {len(user_ids)} 个用户")
            
            # 遍历每个用户获取评论
            for user_id in user_ids:
                try:
                    # 获取用户的评论
                    reviews = self.cache_tool.get_reviews(user_id)
                    if not reviews:
                        continue
                    
                    # 处理每条评论
                    for review in reviews:
                        item_id = review.get('business_id') or review.get('asin') or review.get('book_id')
                        rating = float(review.get('stars', 0))
                        
                        if not item_id or not rating:
                            continue
                            
                        # 获取商品信息
                        item_info = self.cache_tool.get_item(item_id)
                        if not item_info:
                            continue
                            
                        # 确定数据来源
                        source = None
                        if 'asin' in item_info:
                            source = 'amazon'
                        elif 'business_id' in item_info:
                            source = 'yelp'
                        elif 'book_id' in item_info:
                            source = 'goodreads'
                        
                        if not source:
                            continue
                            
                        # 更新用户偏好
                        if source not in self.user_preferences:
                            self.user_preferences[source] = {}
                            
                        if user_id not in self.user_preferences[source]:
                            self.user_preferences[source][user_id] = {
                                'categories': {},
                                'avg_rating': 0,
                                'review_count': 0
                            }
                            
                        user_prefs = self.user_preferences[source][user_id]
                        
                        # 更新类别偏好
                        categories = item_info.get('categories', [])
                        if isinstance(categories, str):
                            categories = categories.split(', ')
                        elif not isinstance(categories, list):
                            categories = []
                            
                        for category in categories:
                            if category not in user_prefs['categories']:
                                user_prefs['categories'][category] = {
                                    'count': 0,
                                    'avg_rating': 0
                                }
                                
                            cat_prefs = user_prefs['categories'][category]
                            old_count = cat_prefs['count']
                            old_avg = cat_prefs['avg_rating']
                            new_count = old_count + 1
                            new_avg = (old_avg * old_count + rating) / new_count
                            
                            cat_prefs.update({
                                'count': new_count,
                                'avg_rating': new_avg
                            })
                        
                        # 更新整体统计
                        old_count = user_prefs['review_count']
                        old_avg = user_prefs['avg_rating']
                        new_count = old_count + 1
                        new_avg = (old_avg * old_count + rating) / new_count
                        
                        user_prefs.update({
                            'review_count': new_count,
                            'avg_rating': new_avg
                        })
                        
                except Exception as e:
                    continue
            
            # 打印训练结果统计
            print("\n=== 训练完成 ===")
            for source in ['amazon', 'yelp', 'goodreads']:
                user_count = len(self.user_preferences.get(source, {}))
                print(f"\n{source} 统计信息:")
                print(f"- 用户数: {user_count}")
                if user_count > 0:
                    sample_user = next(iter(self.user_preferences[source]))
                    print(f"- 示例用户 {sample_user} 的偏好:")
                    print(json.dumps(self.user_preferences[source][sample_user], indent=2, ensure_ascii=False))
            
            return {'status': 'success', 'message': '训练完成'}
            
        except Exception as e:
            print(f"训练过程出错: {e}")
            import traceback
            print(traceback.format_exc())
            return {'status': 'error', 'message': str(e)}
            
    def workflow(self):
        """推荐工作流程"""
        try:
            user_id = self.task['user_id']
            candidate_list = self.task['candidate_list']
            
            if not candidate_list:
                return []
            
            # 确定数据来源
            source = None
            for s in ['amazon', 'yelp', 'goodreads']:
                if user_id in self.user_preferences.get(s, {}):
                    source = s
                    break
            
            if not source:
                return candidate_list
            
            # 获取用户偏好
            user_prefs = self.user_preferences[source][user_id]
            
            # 构建提示词
            prompt = f"""作为推荐系统专家，请根据以下用户信息和候选商品为用户生成个性化推荐。

用户信息 ({source} 平台):
- 评论数: {user_prefs['review_count']}
- 平均评分: {user_prefs['avg_rating']:.2f}
- 偏好类别: {json.dumps(user_prefs['categories'], ensure_ascii=False)}

候选商品:
"""
            
            # 添加候选商品信息
            for item_id in candidate_list:
                item_info = self.cache_tool.get_item(item_id)
                if item_info:
                    prompt += f"- ID: {item_id}\n"
                    prompt += f"  名称: {item_info.get('name', '')}\n"
                    prompt += f"  类别: {item_info.get('categories', [])}\n"
            
            prompt += "\n请返回一个排序后的商品ID列表，格式如下：\n[\"item_id1\", \"item_id2\", ...]"
            
            # 调用LLM生成推荐
            response = self.llm.generate(prompt)
            
            try:
                recommendations = json.loads(response)
                if isinstance(recommendations, list):
                    return recommendations
            except:
                pass
            
            return candidate_list
            
        except Exception as e:
            print(f"推荐过程出错: {e}")
            return candidate_list

    def get_user_history(self, user_id):
        # 获取用户历史数据
        return self.cache_tool.get_reviews(user_id)

    def get_item_info(self, item_id):
        # 获取商品信息
        return self.cache_tool.get_item(item_id)