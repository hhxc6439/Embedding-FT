# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import logging
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
import numpy as np
from peft import LoraConfig, get_peft_model
import os

logger = logging.getLogger(__name__)


@dataclass
class ModelOutput:
    """模型输出"""
    embeddings: torch.Tensor
    pooled_output: Optional[torch.Tensor] = None
    last_hidden_state: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None


class BGEEmbeddingModel(nn.Module):
    """BGE embedding模型，支持LoRA微调"""

    def __init__(self, config):
        super().__init__()
        self.config = config.model

        # 加载预训练模型
        logger.info(f"加载预训练模型: {self.config.base_model_name}")
        self.model = AutoModel.from_pretrained(self.config.base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)

        # 获取隐藏层维度
        self.hidden_size = self.model.config.hidden_size

        # 应用LoRA（如果启用）
        if self.config.use_lora:
            self._apply_lora()

        # 池化方法
        self.pooling_method = self.config.pooling_method
        self.normalize_embeddings = self.config.normalize_embeddings

        # 梯度检查点（节省内存）
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        logger.info(f"模型初始化完成，隐藏层维度: {self.hidden_size}")
        logger.info(f"池化方法: {self.pooling_method}")
        logger.info(f"LoRA启用: {self.config.use_lora}")

    def _apply_lora(self):
        """应用LoRA配置"""
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=["query", "key", "value", "dense"],  # 针对transformer层
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            return_dict: bool = True
    ) -> ModelOutput:
        """前向传播"""
        # 获取模型输出
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=return_dict
        )

        if return_dict:
            last_hidden_state = outputs.last_hidden_state
        else:
            last_hidden_state = outputs[0]

        # 池化策略
        if self.pooling_method == "cls":
            # 使用[CLS] token
            embeddings = last_hidden_state[:, 0, :]
        elif self.pooling_method == "mean":
            # 均值池化
            embeddings = self._mean_pooling(last_hidden_state, attention_mask)
        elif self.pooling_method == "max":
            # 最大池化
            embeddings = self._max_pooling(last_hidden_state, attention_mask)
        elif self.pooling_method == "weighted":
            # 加权池化（基于注意力）
            embeddings = self._weighted_pooling(last_hidden_state, attention_mask)
        else:
            raise ValueError(f"不支持的池化方法: {self.pooling_method}")

        # 归一化
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return ModelOutput(
            embeddings=embeddings,
            pooled_output=embeddings,
            last_hidden_state=last_hidden_state,
            attention_mask=attention_mask
        )

    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """均值池化"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _max_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """最大池化"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # 将padding位置的token设为一个很小的负数
        token_embeddings = token_embeddings.clone()
        token_embeddings[input_mask_expanded == 0] = -1e9
        return torch.max(token_embeddings, 1)[0]

    def _weighted_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """加权池化（基于attention权重）"""
        # 这里使用最后一层的attention权重
        # 注意：这需要模型输出attention_weights
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # 简单的加权策略：使用token embedding的L2范数作为权重
        token_norms = torch.norm(token_embeddings, dim=2, keepdim=True)
        weights = F.softmax(token_norms * input_mask_expanded, dim=1)

        weighted_embeddings = torch.sum(token_embeddings * weights, dim=1)
        return weighted_embeddings

    def encode(
            self,
            texts: List[str],
            batch_size: int = 32,
            device: Optional[torch.device] = None,
            normalize: bool = True,
            show_progress: bool = True
    ) -> np.ndarray:
        """编码文本列表为embedding"""
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.eval()
        self.to(device)

        all_embeddings = []

        with torch.no_grad():
            # 分批处理
            n_batches = (len(texts) + batch_size - 1) // batch_size
            iterator = range(0, len(texts), batch_size)

            if show_progress:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="编码文本", total=n_batches)

            for i in iterator:
                batch_texts = texts[i:i + batch_size]

                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_length,
                    return_tensors="pt"
                )

                # 移动到设备
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                # 获取embeddings
                outputs = self(input_ids, attention_mask)
                batch_embeddings = outputs.embeddings

                # 归一化（如果需要）
                if normalize and self.normalize_embeddings:
                    batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

                all_embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def save_pretrained(self, path: str):
        """保存模型"""
        os.makedirs(path, exist_ok=True)

        # 保存模型权重
        if self.config.use_lora:
            # LoRA模型需要特殊处理
            self.model.save_pretrained(path)
        else:
            self.model.save_pretrained(path)

        # 保存tokenizer
        self.tokenizer.save_pretrained(path)

        # 保存配置
        config_dict = {
            'base_model_name': self.config.base_model_name,
            'pooling_method': self.pooling_method,
            'normalize_embeddings': self.normalize_embeddings,
            'use_lora': self.config.use_lora,
            'lora_config': {
                'r': self.config.lora_r,
                'alpha': self.config.lora_alpha,
                'dropout': self.config.lora_dropout
            } if self.config.use_lora else None,
            'hidden_size': self.hidden_size
        }

        import json
        config_path = os.path.join(path, "model_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)

        logger.info(f"模型已保存到: {path}")

    @classmethod
    def from_pretrained(cls, path: str, config=None):
        """从预训练模型加载"""
        import json

        # 加载配置
        config_path = os.path.join(path, "model_config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            saved_config = json.load(f)

        # 创建模型实例
        if config is None:
            from config import Config
            config = Config()

        # 更新模型配置
        config.model.base_model_name = saved_config['base_model_name']
        config.model.pooling_method = saved_config['pooling_method']
        config.model.normalize_embeddings = saved_config['normalize_embeddings']
        config.model.use_lora = saved_config['use_lora']

        if saved_config['use_lora']:
            lora_config = saved_config['lora_config']
            config.model.lora_r = lora_config['r']
            config.model.lora_alpha = lora_config['alpha']
            config.model.lora_dropout = lora_config['dropout']

        model = cls(config)

        # 加载权重
        if saved_config['use_lora']:
            # 加载LoRA权重
            from peft import PeftModel
            base_model = AutoModel.from_pretrained(saved_config['base_model_name'])
            model.model = PeftModel.from_pretrained(base_model, path)
        else:
            # 加载完整模型
            model.model = AutoModel.from_pretrained(path)

        # 加载tokenizer
        model.tokenizer = AutoTokenizer.from_pretrained(path)

        logger.info(f"从 {path} 加载模型")
        return model


class ContrastiveLoss(nn.Module):
    """对比学习损失函数集合"""

    def __init__(self, config):
        super().__init__()
        self.config = config.training

        # 损失函数参数
        self.margin = self.config.margin
        self.temperature = self.config.temperature

        # 损失函数类型
        self.loss_type = self.config.loss_type

        logger.info(f"初始化对比损失函数，类型: {self.loss_type}")
        logger.info(f"参数 - margin: {self.margin}, temperature: {self.temperature}")

    def forward(
            self,
            query_embeddings: torch.Tensor,
            positive_embeddings: torch.Tensor,
            negative_embeddings: torch.Tensor,
            return_components: bool = False
    ) -> torch.Tensor:
        """计算对比损失"""

        if self.loss_type == "triplet":
            loss = self.triplet_margin_loss(query_embeddings, positive_embeddings, negative_embeddings)
        elif self.loss_type == "contrastive":
            loss = self.contrastive_loss(query_embeddings, positive_embeddings, negative_embeddings)
        elif self.loss_type == "cosine":
            loss = self.cosine_embedding_loss(query_embeddings, positive_embeddings, negative_embeddings)
        elif self.loss_type == "info_nce":
            loss = self.info_nce_loss(query_embeddings, positive_embeddings, negative_embeddings)
        elif self.loss_type == "multi_negative":
            loss = self.multi_negative_ranking_loss(query_embeddings, positive_embeddings, negative_embeddings)
        else:
            raise ValueError(f"不支持的损失类型: {self.loss_type}")

        if return_components:
            # 返回损失和各个组件（用于分析）
            return loss, {
                'loss_type': self.loss_type,
                'loss_value': loss.item() if isinstance(loss, torch.Tensor) else loss
            }

        return loss

    def triplet_margin_loss(
            self,
            query: torch.Tensor,
            positive: torch.Tensor,
            negative: torch.Tensor
    ) -> torch.Tensor:
        """Triplet Margin Loss"""
        # 计算余弦距离
        pos_distance = 1 - F.cosine_similarity(query, positive, dim=-1)
        neg_distance = 1 - F.cosine_similarity(query, negative, dim=-1)

        # Triplet loss
        loss = F.relu(pos_distance - neg_distance + self.margin)

        return loss.mean()

    def contrastive_loss(
            self,
            query: torch.Tensor,
            positive: torch.Tensor,
            negative: torch.Tensor
    ) -> torch.Tensor:
        """Contrastive Loss（双重损失）"""
        # 正样本对损失
        pos_similarity = F.cosine_similarity(query, positive, dim=-1)
        pos_loss = torch.mean((1 - pos_similarity) ** 2)

        # 负样本对损失
        neg_similarity = F.cosine_similarity(query, negative, dim=-1)
        neg_loss = torch.mean(torch.clamp(neg_similarity - self.margin, min=0) ** 2)

        # 总损失
        total_loss = pos_loss + neg_loss

        return total_loss

    def cosine_embedding_loss(
            self,
            query: torch.Tensor,
            positive: torch.Tensor,
            negative: torch.Tensor
    ) -> torch.Tensor:
        """Cosine Embedding Loss"""
        # 正样本标签为1，负样本标签为-1
        pos_loss = F.cosine_embedding_loss(
            query, positive,
            torch.ones(query.size(0), device=query.device),
            margin=self.margin
        )

        neg_loss = F.cosine_embedding_loss(
            query, negative,
            torch.full((query.size(0),), -1, device=query.device),
            margin=self.margin
        )

        return (pos_loss + neg_loss) / 2

    def info_nce_loss(
            self,
            query: torch.Tensor,
            positive: torch.Tensor,
            negative: torch.Tensor
    ) -> torch.Tensor:
        """InfoNCE Loss（对比学习的标准形式）"""
        # 计算相似度
        pos_sim = F.cosine_similarity(query, positive, dim=-1) / self.temperature
        neg_sim = F.cosine_similarity(query, negative, dim=-1) / self.temperature

        # InfoNCE损失
        numerator = torch.exp(pos_sim)
        denominator = torch.exp(pos_sim) + torch.exp(neg_sim)

        loss = -torch.log(numerator / denominator)

        return loss.mean()

    def multi_negative_ranking_loss(
            self,
            query: torch.Tensor,
            positive: torch.Tensor,
            negative: torch.Tensor
    ) -> torch.Tensor:
        """多重负样本排序损失"""
        # 假设negative的形状为 [batch_size, num_negatives, hidden_dim]
        # 这里我们假设negative是单个负样本，如果需要多个负样本，需要修改

        # 计算正样本相似度
        pos_sim = F.cosine_similarity(query, positive, dim=-1)

        # 计算负样本相似度
        neg_sim = F.cosine_similarity(query, negative, dim=-1)

        # 对于每个query，我们希望正样本相似度 > 负样本相似度
        loss = F.relu(self.margin - pos_sim + neg_sim)

        return loss.mean()

    def compute_similarities(
            self,
            query: torch.Tensor,
            positive: torch.Tensor,
            negative: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """计算相似度统计"""
        with torch.no_grad():
            pos_sim = F.cosine_similarity(query, positive, dim=-1)
            neg_sim = F.cosine_similarity(query, negative, dim=-1)

            return {
                'pos_similarity_mean': pos_sim.mean().item(),
                'pos_similarity_std': pos_sim.std().item(),
                'neg_similarity_mean': neg_sim.mean().item(),
                'neg_similarity_std': neg_sim.std().item(),
                'similarity_gap': (pos_sim - neg_sim).mean().item(),
                'accuracy': (pos_sim > neg_sim).float().mean().item()
            }


class PatentEmbeddingModel(nn.Module):
    """专利embedding模型（包装器）"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 创建embedding模型
        self.embedding_model = BGEEmbeddingModel(config)

        # 创建损失函数
        self.loss_fn = ContrastiveLoss(config)

        # 获取模型维度
        self.embedding_dim = self.embedding_model.hidden_size

        logger.info(f"专利embedding模型初始化完成")
        logger.info(f"Embedding维度: {self.embedding_dim}")

    def forward(
            self,
            query_input_ids: torch.Tensor,
            query_attention_mask: torch.Tensor,
            positive_input_ids: torch.Tensor,
            positive_attention_mask: torch.Tensor,
            negative_input_ids: torch.Tensor,
            negative_attention_mask: torch.Tensor,
            compute_loss: bool = True
    ) -> Dict[str, Any]:
        """前向传播"""

        # 编码query
        query_outputs = self.embedding_model(query_input_ids, query_attention_mask)
        query_embeddings = query_outputs.embeddings

        # 编码正样本
        positive_outputs = self.embedding_model(positive_input_ids, positive_attention_mask)
        positive_embeddings = positive_outputs.embeddings

        # 编码负样本
        negative_outputs = self.embedding_model(negative_input_ids, negative_attention_mask)
        negative_embeddings = negative_outputs.embeddings

        # 计算损失（如果需要）
        if compute_loss:
            loss = self.loss_fn(query_embeddings, positive_embeddings, negative_embeddings)
        else:
            loss = None

        # 计算相似度统计
        similarity_stats = self.loss_fn.compute_similarities(
            query_embeddings, positive_embeddings, negative_embeddings
        )

        return {
            'loss': loss,
            'query_embeddings': query_embeddings,
            'positive_embeddings': positive_embeddings,
            'negative_embeddings': negative_embeddings,
            'similarity_stats': similarity_stats
        }

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        """编码查询（BGE 推荐：query 前加 instruction）"""
        instruction = getattr(self.config.model, 'query_instruction_for_retrieval', None) or ""
        if instruction:
            queries = [instruction + q for q in queries]
        return self.embedding_model.encode(queries, **kwargs)

    def encode_documents(self, documents: List[str], **kwargs) -> np.ndarray:
        """编码文档（BGE 推荐：passage 不加 instruction）"""
        return self.embedding_model.encode(documents, **kwargs)

    def save_pretrained(self, path: str):
        """保存模型"""
        self.embedding_model.save_pretrained(path)

        # 保存损失函数配置
        import json
        loss_config = {
            'loss_type': self.config.training.loss_type,
            'margin': self.config.training.margin,
            'temperature': self.config.training.temperature
        }

        loss_config_path = os.path.join(path, "loss_config.json")
        with open(loss_config_path, 'w', encoding='utf-8') as f:
            json.dump(loss_config, f, ensure_ascii=False, indent=2)

        logger.info(f"完整模型已保存到: {path}")

    @classmethod
    def from_pretrained(cls, path: str, config=None):
        """从预训练模型加载"""
        # 创建模型实例
        if config is None:
            from config import Config
            config = Config()

        model = cls(config)

        # 加载embedding模型
        model.embedding_model = BGEEmbeddingModel.from_pretrained(path, config)

        logger.info(f"从 {path} 加载完整模型")
        return model


def create_model(config) -> PatentEmbeddingModel:
    """创建专利embedding模型"""
    return PatentEmbeddingModel(config)