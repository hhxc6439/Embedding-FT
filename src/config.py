# config.py
import os
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import json

# 项目根目录（config 所在目录的上级）
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_DEFAULT_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "embedding", "微调数据-5753")


# config.py 中更新 DataConfig
@dataclass
class DataConfig:
    """数据配置"""
    # 数据路径（默认指向微调数据目录）
    raw_data_dir: str = _DEFAULT_DATA_DIR  # 原始数据目录
    processed_data_dir: str = os.path.join(_PROJECT_ROOT, "data", "processed")  # 处理后的数据目录

    # 数据采样
    max_samples: Optional[int] = None  # 最大样本数（None表示全部）
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # IPC配置
    ipc_strategy: str = "first"  # IPC处理策略: first, all, first_g, first_h
    use_ipc_filter: bool = True  # 是否使用IPC过滤
    min_patents_per_ipc: int = 1  # 每个IPC最少专利数（1=小数据友好，同大组负例需≥2专利）
    max_groups_per_patent: int = 3  # 每个专利最多保留几个大组

    # 文本处理
    max_abstract_length: int = 300  # 摘要最大长度（词数）
    max_title_length: int = 50  # 标题最大长度


@dataclass
class QueryConfig:
    """查询生成配置"""
    use_llm: bool = True  # 是否使用LLM生成query
    llm_model: str = "gpt-3.5-turbo"  # LLM模型名称
    llm_api_key: Optional[str] = None  # API密钥
    llm_base_url: str = "https://api.openai.com/v1"  # API基础URL

    # 本地模型配置（默认用本地 LLM；提供 api key 且 --use_api_llm 时改用 API）
    local_model_enabled: bool = True
    local_model_name: str = "/home/cly/Embedding-FT/model/Qwen2.5-7B-Instruct"  # 本地模型名称或路径
    local_model_device: str = "cuda:4"
    local_model_dtype: str = "auto"
    max_new_tokens: int = 1024

    # 规则方法配置（专利RAG：摘要→query，故默认 abstract_based）
    strategy: str = "abstract_based"  # title_based, abstract_based, hybrid
    min_query_length: int = 10
    max_query_length: int = 100

    # 规则模板（abstract_based 时也会用标题作后备）
    query_templates: List[str] = field(default_factory=lambda: [
        "关于{title}的技术方案是什么？",
        "如何实现{title}？",
        "{title}的技术要点有哪些？",
        "什么是{title}？"
    ])

    # LLM query 缓存：重复训练时复用已生成的 query
    use_query_cache: bool = True
    query_cache_filename: str = "llm_queries_cache.json"


@dataclass
class NegativeSamplingConfig:
    """负样本采样配置"""
    # 策略选择
    strategy: str = "optimized_mixed"  # 新策略：optimized_mixed, same_group_priority, same_subclass_different_group, mixed, random, hard
    negatives_per_positive: int = 3  # 每个正样本的负样本数

    # 同大组负例配置（新）
    same_group_enabled: bool = True  # 是否启用同大组负例
    same_group_max_samples: int = 2  # 同大组最多取几个负例
    same_group_min_samples_required: int = 1  # 至少需要几个同大组负例才能使用此策略
    same_group_similarity_threshold: float = 0.8  # 同大组负例的相似度阈值

    # 同小类不同大组配置
    same_subclass_diff_group_enabled: bool = True  # 是否启用同小类不同大组负例
    same_subclass_diff_group_max_samples: int = 1  # 同小类不同大组最多取几个

    # 随机负例配置
    random_enabled: bool = True  # 是否启用随机负例
    random_max_samples: int = 1  # 随机负例最多取几个

    # 原有配置（保持向后兼容）
    ipc_level_for_sampling: str = "subclass"  # 采样时的IPC级别
    same_subclass_weight: float = 0.7  # 同小类权重
    exclude_same_group: bool = True  # 是否排除同大组
    group_similarity_threshold: float = 0.8  # 大组相似度阈值（调整为0.8）

    # 硬负样本
    hard_negative_weight: float = 0.2
    semantic_similarity_threshold: float = 0.3  # 调整为0.3

    # 随机负样本
    random_negative_weight: float = 0.1


@dataclass
class ModelConfig:
    """模型配置"""
    # 基础模型（专利RAG 使用 bge-base-zh-v1.5）
    base_model_name: str = "/home/cly/Embedding-FT/model/bge-base-zh-v1.5"
    # 微调选项
    use_lora: bool = False  # 是否使用LoRA微调
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # 编码器配置
    pooling_method: str = "cls"  # cls, mean, max
    normalize_embeddings: bool = True
    max_seq_length: int = 512

    # BGE 官方推荐：短query检索长文档时，需在 query 前加 instruction；passage 不加
    query_instruction_for_retrieval: str = "为这个句子生成表示以用于检索相关文章："


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础训练参数
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 50
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # 损失函数
    loss_type: str = "triplet"  # triplet, contrastive, cosine, info_nce, multi_negative
    margin: float = 0.5  # triplet loss margin
    temperature: float = 0.05  # contrastive loss temperature

    # 优化器
    optimizer: str = "adamw"
    scheduler: str = "linear"  # linear, cosine, constant

    # 早停
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001

    # 评估
    eval_steps: int = 100  # 每多少步评估一次
    save_steps: int = 500  # 每多少步保存一次（0=不按步保存）

    # 检查点：每 N 个 epoch 保存一次 epoch 级 checkpoint
    checkpoint_every_n_epochs: int = 1

    # DataLoader：None=自动（Windows 用 0，Linux 用 4），可显式指定
    dataloader_num_workers: Optional[int] = None

    # 评估指标
    eval_strict_threshold: float = 0.1  # 严格准确率阈值

    # 负样本配置扩展
    technical_negative_enabled: bool = True  # 是否启用技术性负样本
    technical_negative_weight: float = 0.3  # 技术性负样本权重

    # 损失函数增强
    use_hard_negative_mining: bool = True  # 是否使用难负样本挖掘
    hard_negative_margin: float = 0.2  # 难负样本额外margin


@dataclass
class ExperimentConfig:
    """实验配置"""
    experiment_name: str = "patent_embedding_ft"
    output_dir: str = os.path.join(_PROJECT_ROOT, "experiments")
    log_dir: str = os.path.join(_PROJECT_ROOT, "logs")

    # 随机种子
    seed: int = 42

    # 设备
    device: str = "cuda:4" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True  # 混合精度训练

    def __post_init__(self):
        # 创建必要的目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # 设置随机种子
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)


# 全局配置
class Config:
    """总配置"""

    def __init__(self):
        self.data = DataConfig()
        self.query = QueryConfig()
        self.negative = NegativeSamplingConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.experiment = ExperimentConfig()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'data': self.data.__dict__,
            'query': self.query.__dict__,
            'negative': self.negative.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'experiment': self.experiment.__dict__
        }

    def save(self, path: str):
        """保存配置到文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        """从文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        config = cls()
        # 更新配置
        for section, values in config_dict.items():
            if hasattr(config, section):
                section_obj = getattr(config, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)

        return config


# 默认配置实例
config = Config()