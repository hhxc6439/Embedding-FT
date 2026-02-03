import os
import json
import random
import pickle
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """数据集构建配置"""
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # 样本筛选
    min_query_length: int = 10
    max_query_length: int = 100
    min_abstract_length: int = 50
    max_abstract_length: int = 500

    # 平衡性配置
    balance_ipc_distribution: bool = True
    max_samples_per_subclass: int = 1000
    min_samples_per_subclass: int = 1  # 小数据友好；同大组负例需至少2个专利

    # 负样本配置
    num_negatives_per_query: int = 3  # 每个query的负样本数

    # 划分策略（避免同一专利同时出现在 train/val/test）
    split_by_patent: bool = True  # 按专利ID划分，防止泄露

    # 其他
    seed: int = 42
    cache_dir: str = "./cache"

    def __post_init__(self):
        # 确保比例总和为1
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.001:
            logger.warning(f"数据集比例总和不为1({total:.3f})，将自动调整")
            # 按比例调整
            self.train_ratio = self.train_ratio / total
            self.val_ratio = self.val_ratio / total
            self.test_ratio = self.test_ratio / total

        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)


class DatasetBuilder:
    """数据集构建器"""

    def __init__(self, config, query_generator, negative_sampler, data_loader):
        self.config = config
        self.query_generator = query_generator
        self.negative_sampler = negative_sampler
        self.data_loader = data_loader
        self.dataset_config = DatasetConfig()

        # 设置随机种子
        random.seed(config.experiment.seed)
        np.random.seed(config.experiment.seed)

        # 数据集存储
        self.training_samples = []
        self.validation_samples = []
        self.test_samples = []

        # 统计信息
        self.stats = {
            'total_samples': 0,
            'train_samples': 0,
            'val_samples': 0,
            'test_samples': 0,
            'ipc_distribution': defaultdict(int),
            'query_length_stats': {},
            'abstract_length_stats': {}
        }

    def build_dataset(self, use_cache: bool = True, pre_generated_queries: Optional[List[Dict]] = None) -> Dict[
        str, List]:
        """构建完整数据集

        Args:
            use_cache: 是否使用缓存
            pre_generated_queries: 预先生成的查询列表（可选）
        """
        cache_path = os.path.join(self.dataset_config.cache_dir, "dataset_cache.pkl")

        # 检查缓存
        if use_cache and os.path.exists(cache_path):
            logger.info("从缓存加载数据集...")
            return self._load_from_cache(cache_path)

        logger.info("开始构建数据集...")

        # 1. 生成查询（如果未提供预先生成的查询）
        if pre_generated_queries is not None:
            queries = pre_generated_queries
            logger.info(f"使用预先生成的 {len(queries)} 个查询")
        else:
            queries = self.query_generator.generate_batch(self.data_loader.patents)
            logger.info(f"生成了 {len(queries)} 个查询")

        # 2. 按IPC小类分组
        subclass_to_queries = self._group_queries_by_subclass(queries)

        # 3. 过滤样本数不足的IPC小类
        valid_subclasses = self._filter_subclasses(subclass_to_queries)
        logger.info(f"有效IPC小类数量: {len(valid_subclasses)}")

        # 4. 构建训练样本
        all_samples = []

        for subclass, query_samples in tqdm(valid_subclasses.items(), desc="构建样本"):
            subclass_samples = self._build_samples_for_subclass(subclass, query_samples)
            all_samples.extend(subclass_samples)
            logger.debug(f"IPC小类 {subclass}: 生成了 {len(subclass_samples)} 个样本")

        logger.info(f"总共生成了 {len(all_samples)} 个样本")

        # 5. 分割数据集
        self._split_dataset(all_samples)

        # 6. 构建最终数据集
        dataset = {
            'train': self.training_samples,
            'val': self.validation_samples,
            'test': self.test_samples
        }

        # 7. 保存到缓存
        if use_cache:
            self._save_to_cache(dataset, cache_path)

        # 8. 计算统计信息
        self._calculate_statistics()

        return dataset

    def _group_queries_by_subclass(self, queries: List[Dict]) -> Dict[str, List[Dict]]:
        """按IPC小类分组查询"""
        subclass_to_queries = defaultdict(list)

        for query_sample in queries:
            subclasses = query_sample.get('ipc_subclasses', [])
            for subclass in subclasses:
                if subclass:  # 确保小类不为空
                    subclass_to_queries[subclass].append(query_sample)

        return dict(subclass_to_queries)

    def _filter_subclasses(self, subclass_to_queries: Dict[str, List]) -> Dict[str, List]:
        """过滤样本数不足的IPC小类。min_samples_per_subclass=1 时保留所有小类（小数据友好）。"""
        min_n = max(1, self.dataset_config.min_samples_per_subclass)
        valid_subclasses = {}

        for subclass, queries in subclass_to_queries.items():
            if len(queries) < min_n:
                continue
            if len(queries) > self.dataset_config.max_samples_per_subclass:
                logger.info(f"IPC小类 {subclass} 样本数过多({len(queries)})，进行采样")
                queries = random.sample(queries, self.dataset_config.max_samples_per_subclass)
            valid_subclasses[subclass] = queries

        return valid_subclasses

    # 在 dataset_builder.py 的 _build_samples_for_subclass 方法中
    # 修改为每个query创建多个负样本的训练样本

    def _build_samples_for_subclass(self, subclass: str, query_samples: List[Dict]) -> List[Dict]:
        """为特定IPC小类构建样本（修改版）"""
        samples = []

        # 限制每个query的最大样本数
        for query_sample in query_samples:
            patent_id = query_sample['patent_id']
            anchor_patent = self.data_loader.get_patent_by_id(patent_id)

            if not anchor_patent:
                logger.warning(f"无法找到专利 {patent_id}")
                continue

            # 获取负样本（多个）
            negative_patents = self.negative_sampler.get_negative_samples(
                anchor_patent,
                num_samples=self.dataset_config.num_negatives_per_query
            )

            if not negative_patents:
                logger.warning(f"专利 {patent_id} 无法找到负样本")
                continue

            # 为每个负样本创建独立的训练样本
            for neg_patent in negative_patents:
                # 构建训练样本
                training_sample = {
                    'query': query_sample['query'],
                    'query_id': f"Q_{patent_id}_{neg_patent['id']}",
                    'positive': {
                        'id': anchor_patent.id,
                        'title': anchor_patent.title,
                        'abstract': anchor_patent.abstract,
                        'ipc_subclasses': list(anchor_patent.ipc_subclasses),
                        'ipc_groups': list(anchor_patent.ipc_groups)
                    },
                    'negative': {
                        'id': neg_patent['id'],
                        'title': neg_patent['title'],
                        'abstract': neg_patent['abstract'],
                        'ipc_subclasses': neg_patent.get('ipc_subclasses', []),
                        'ipc_groups': neg_patent.get('ipc_groups', []),
                        'similarity_score': neg_patent.get('similarity_score', 0.0),
                        'negative_type': neg_patent.get('negative_type', 'unknown')
                    },
                    'metadata': {
                        'positive_ipc_subclass': subclass,
                        'negative_ipc_subclass': neg_patent.get('ipc_subclasses', [])[:1],
                        'positive_n_claims': anchor_patent.n_claims,
                        'negative_n_claims': 0,
                        'query_generation_method': self.query_generator.__class__.__name__,
                        'negative_sampling_method': self.negative_sampler.config.strategy,
                        'negative_type': neg_patent.get('negative_type', 'unknown'),
                        'similarity_score': neg_patent.get('similarity_score', 0.0)
                    }
                }

                # 验证样本质量
                if self._validate_sample(training_sample):
                    samples.append(training_sample)

        return samples

    def _validate_sample(self, sample: Dict) -> bool:
        """验证样本质量"""
        # 检查query长度
        query = sample['query']
        if len(query) < self.dataset_config.min_query_length:
            return False
        if len(query) > self.dataset_config.max_query_length:
            return False

        # 检查摘要长度
        positive_abstract = sample['positive']['abstract']
        negative_abstract = sample['negative']['abstract']

        if len(positive_abstract) < self.dataset_config.min_abstract_length:
            return False
        if len(negative_abstract) < self.dataset_config.min_abstract_length:
            return False

        # 检查是否重复
        if sample['positive']['id'] == sample['negative']['id']:
            return False

        # 检查IPC小类是否相同（应该相同）
        pos_subclasses = set(sample['positive']['ipc_subclasses'])
        neg_subclasses = set(sample['negative']['ipc_subclasses'])

        # 检查是否有交集（理想情况下应该有交集）
        if not pos_subclasses.intersection(neg_subclasses):
            logger.debug(f"正负样本IPC小类无交集: {sample['positive']['id']} vs {sample['negative']['id']}")
            # 这不一定是错误，但记录下来

        return True

    def _split_dataset(self, all_samples: List[Dict]):
        """分割数据集。若 split_by_patent=True，按正样本专利ID划分，避免同一专利出现在多折造成泄露。"""
        cfg = self.dataset_config
        if not all_samples:
            self.training_samples = []
            self.validation_samples = []
            self.test_samples = []
            return

        if getattr(cfg, "split_by_patent", True):
            # 按正样本专利ID分组
            pid_to_samples: Dict[str, List[Dict]] = defaultdict(list)
            for s in all_samples:
                pid = s["positive"]["id"]
                pid_to_samples[pid].append(s)

            pids = list(pid_to_samples.keys())
            random.shuffle(pids)
            n = len(pids)
            n_train = int(n * cfg.train_ratio)
            n_val = int(n * cfg.val_ratio)

            train_pids = set(pids[:n_train])
            val_pids = set(pids[n_train : n_train + n_val])
            test_pids = set(pids[n_train + n_val :])

            self.training_samples = [s for pid in train_pids for s in pid_to_samples[pid]]
            self.validation_samples = [s for pid in val_pids for s in pid_to_samples[pid]]
            self.test_samples = [s for pid in test_pids for s in pid_to_samples[pid]]
            random.shuffle(self.training_samples)
            random.shuffle(self.validation_samples)
            random.shuffle(self.test_samples)
        else:
            random.shuffle(all_samples)
            n_total = len(all_samples)
            n_train = int(n_total * cfg.train_ratio)
            n_val = int(n_total * cfg.val_ratio)
            self.training_samples = all_samples[:n_train]
            self.validation_samples = all_samples[n_train : n_train + n_val]
            self.test_samples = all_samples[n_train + n_val :]

        logger.info("=" * 50)
        logger.info("数据集分割结果:" + (" (按专利划分)" if getattr(cfg, "split_by_patent", True) else ""))
        logger.info(f"  训练集: {len(self.training_samples)} 个样本")
        logger.info(f"  验证集: {len(self.validation_samples)} 个样本")
        logger.info(f"  测试集: {len(self.test_samples)} 个样本")
        logger.info(f"  总计: {len(all_samples)} 个样本")
        logger.info("=" * 50)

        # 更新统计信息
        self.stats['total_samples'] = len(all_samples)
        self.stats['train_samples'] = len(self.training_samples)
        self.stats['val_samples'] = len(self.validation_samples)
        self.stats['test_samples'] = len(self.test_samples)

    def _calculate_statistics(self):
        """计算数据集统计信息"""
        logger.info("计算数据集统计信息...")

        # 计算query长度统计
        all_queries = [s['query'] for s in self.training_samples]
        if all_queries:
            query_lengths = [len(q) for q in all_queries]
            self.stats['query_length_stats'] = {
                'min': min(query_lengths),
                'max': max(query_lengths),
                'mean': np.mean(query_lengths),
                'median': np.median(query_lengths),
                'std': np.std(query_lengths)
            }

        # 计算摘要长度统计
        all_abstracts = []
        for sample in self.training_samples:
            all_abstracts.append(sample['positive']['abstract'])
            all_abstracts.append(sample['negative']['abstract'])

        if all_abstracts:
            abstract_lengths = [len(a) for a in all_abstracts]
            self.stats['abstract_length_stats'] = {
                'min': min(abstract_lengths),
                'max': max(abstract_lengths),
                'mean': np.mean(abstract_lengths),
                'median': np.median(abstract_lengths),
                'std': np.std(abstract_lengths)
            }

        # 计算IPC分布
        for sample in self.training_samples:
            for subclass in sample['positive']['ipc_subclasses']:
                self.stats['ipc_distribution'][subclass] += 1

        # 打印统计信息
        self._print_statistics()

    def _print_statistics(self):
        """打印统计信息"""
        logger.info("=" * 50)
        logger.info("数据集详细统计:")
        logger.info(f"Query长度统计:")
        for key, value in self.stats['query_length_stats'].items():
            logger.info(f"  {key}: {value:.2f}")

        logger.info(f"摘要长度统计:")
        for key, value in self.stats['abstract_length_stats'].items():
            logger.info(f"  {key}: {value:.2f}")

        logger.info(f"IPC小类分布 (Top 10):")
        sorted_ipc = sorted(self.stats['ipc_distribution'].items(),
                            key=lambda x: x[1], reverse=True)[:10]
        for ipc, count in sorted_ipc:
            logger.info(f"  {ipc}: {count} 个样本")
        logger.info("=" * 50)

    def _save_to_cache(self, dataset: Dict, cache_path: str):
        """保存数据集到缓存"""
        cache_data = {
            'dataset': dataset,
            'stats': self.stats,
            'config': self.dataset_config
        }

        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

        logger.info(f"数据集已保存到缓存: {cache_path}")

    def _load_from_cache(self, cache_path: str) -> Dict:
        """从缓存加载数据集"""
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)

        dataset = cache_data['dataset']
        self.stats = cache_data['stats']

        self.training_samples = dataset['train']
        self.validation_samples = dataset['val']
        self.test_samples = dataset['test']

        logger.info(f"从缓存加载了 {len(self.training_samples)} 个训练样本")
        logger.info(f"从缓存加载了 {len(self.validation_samples)} 个验证样本")
        logger.info(f"从缓存加载了 {len(self.test_samples)} 个测试样本")

        return dataset

    def save_dataset(self, output_dir: str, dataset_name: str = "patent_rag_dataset"):
        """保存数据集到文件"""
        os.makedirs(output_dir, exist_ok=True)

        # 保存数据集
        dataset_files = {}

        for split_name, samples in [('train', self.training_samples),
                                    ('val', self.validation_samples),
                                    ('test', self.test_samples)]:
            if not samples:
                logger.warning(f"{split_name}集为空，跳过保存")
                continue

            file_path = os.path.join(output_dir, f"{dataset_name}_{split_name}.json")

            # 保存为JSON格式
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)

            dataset_files[split_name] = file_path
            logger.info(f"{split_name}集已保存到 {file_path}")

        # 保存统计信息（转换 numpy 类型以支持 JSON 序列化）
        stats_path = os.path.join(output_dir, f"{dataset_name}_stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(_convert_stats_to_serializable(self.stats), f, ensure_ascii=False, indent=2)

        # 保存配置
        config_path = os.path.join(output_dir, f"{dataset_name}_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            config_dict = {
                'dataset_config': self.dataset_config.__dict__,
                'build_timestamp': str(pd.Timestamp.now())
            }
            json.dump(config_dict, f, ensure_ascii=False, indent=2)

        return dataset_files


def _convert_stats_to_serializable(obj):
    """将 stats 中的 numpy 类型转换为 Python 原生类型以便 JSON 序列化"""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _convert_stats_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_stats_to_serializable(x) for x in obj]
    return obj


class PatentDataset:
    """PyTorch数据集类"""

    def __init__(self, samples: List[Dict], tokenizer, max_length: int = 512,
                 query_instruction: Optional[str] = None):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.query_instruction = query_instruction or ""

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # BGE 官方推荐：query 加 instruction，passage(abstract) 不加
        query_text = (self.query_instruction + sample['query']) if self.query_instruction else sample['query']

        # Tokenize query
        query_encoding = self.tokenizer(
            query_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Tokenize positive abstract
        positive_encoding = self.tokenizer(
            sample['positive']['abstract'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Tokenize negative abstract
        negative_encoding = self.tokenizer(
            sample['negative']['abstract'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 移除batch维度（DataLoader会重新添加）
        return {
            'query_input_ids': query_encoding['input_ids'].squeeze(0),
            'query_attention_mask': query_encoding['attention_mask'].squeeze(0),
            'positive_input_ids': positive_encoding['input_ids'].squeeze(0),
            'positive_attention_mask': positive_encoding['attention_mask'].squeeze(0),
            'negative_input_ids': negative_encoding['input_ids'].squeeze(0),
            'negative_attention_mask': negative_encoding['attention_mask'].squeeze(0),
            'query_text': sample['query'],
            'positive_id': sample['positive']['id'],
            'negative_id': sample['negative']['id'],
            'positive_title': sample['positive']['title'],
            'negative_title': sample['negative']['title']
        }


def create_data_collator(tokenizer):
    """创建数据收集器"""
    from transformers import DataCollatorWithPadding

    return DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding='longest',
        max_length=512,
        pad_to_multiple_of=8,
        return_tensors='pt'
    )