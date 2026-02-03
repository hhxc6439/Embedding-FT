# negative_sampler.py
import random
import numpy as np
from typing import List, Dict, Set, Optional, Any
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import logging
from tqdm import tqdm
import threading
import os
import jieba

logger = logging.getLogger(__name__)


@dataclass
class NegativeSample:
    """负样本数据类"""
    patent_id: str
    patent_title: str
    patent_abstract: str
    ipc_subclasses: List[str]
    ipc_groups: List[str]
    similarity_score: float = 0.0  # 与锚点的相似度
    negative_type: str = ""  # 负样本类型


class NegativeSampler:
    """负样本采样器（优化版：优先同大组负例）"""

    def __init__(self, data_loader, config):
        self.data_loader = data_loader
        self.config = config.negative
        self.cache = {}  # 缓存计算结果
        self.lock = threading.Lock()

        # 设置 jieba 使用用户目录作为缓存
        user_cache_dir = os.path.expanduser("~/.jieba_cache")
        os.makedirs(user_cache_dir, exist_ok=True)
        jieba.dt.tmp_dir = user_cache_dir
        # 强制重新初始化 jieba
        jieba.initialize()

        # 构建大组级别的索引（关键优化）
        self._build_group_index()

        logger.info(f"负样本采样器初始化完成，策略: {self.config.strategy}")

    def _build_group_index(self):
        """构建大组索引，加速同大组专利查找"""
        logger.info("构建大组索引...")

        self.group_to_patents = defaultdict(list)
        self.patent_to_groups = defaultdict(list)

        for patent in tqdm(self.data_loader.patents, desc="构建大组索引"):
            for group in patent.ipc_groups:
                self.group_to_patents[group].append(patent)
                self.patent_to_groups[patent.id].append(group)

    def get_negative_samples(self, anchor_patent, num_samples: Optional[int] = None) -> List[Dict]:
        """获取负样本（优化版本）"""
        if num_samples is None:
            num_samples = self.config.negatives_per_positive

        anchor_id = anchor_patent.id

        # 根据策略选择采样方法
        if self.config.strategy == "optimized_mixed":
            negatives = self._sample_optimized_mixed(anchor_patent, num_samples)
        elif self.config.strategy == "same_group_priority":
            negatives = self._sample_same_group_priority(anchor_patent, num_samples)
        elif self.config.strategy == "same_subclass_different_group":
            negatives = self._sample_same_subclass_different_group(anchor_patent, num_samples)
        elif self.config.strategy == "mixed":
            negatives = self._sample_mixed(anchor_patent, num_samples)
        elif self.config.strategy == "random":
            negatives = self._sample_random(anchor_patent, num_samples)
        elif self.config.strategy == "hard":
            negatives = self._sample_hard(anchor_patent, num_samples)
        else:
            logger.warning(f"未知策略: {self.config.strategy}，使用优化混合策略")
            negatives = self._sample_optimized_mixed(anchor_patent, num_samples)

        # 转换为字典格式
        negative_dicts = []
        for neg in negatives:
            negative_dicts.append({
                'id': neg.patent_id,
                'title': neg.patent_title,
                'abstract': neg.patent_abstract,
                'ipc_subclasses': neg.ipc_subclasses,
                'ipc_groups': neg.ipc_groups,
                'similarity_score': neg.similarity_score,
                'negative_type': getattr(neg, 'negative_type', 'unknown')
            })

        return negative_dicts

    # def _sample_optimized_mixed(self, anchor_patent, num_samples: int) -> List[NegativeSample]:
    #     """优化版混合策略：优先同大组，后备其他策略"""
    #     anchor_id = anchor_patent.id
    #     anchor_groups = self.patent_to_groups.get(anchor_id, [])
    #
    #     negatives = []
    #     selected_ids = {anchor_id}
    #
    #     # 第1步：尝试获取同大组负例（优先级最高）
    #     if getattr(self.config, 'same_group_enabled', True) and anchor_groups:
    #         same_group_negs = self._sample_same_group_negatives(
    #             anchor_patent,
    #             min(num_samples, getattr(self.config, 'same_group_max_samples', 2))
    #         )
    #
    #         for neg in same_group_negs:
    #             if neg.patent_id not in selected_ids:
    #                 neg.negative_type = 'same_group'
    #                 negatives.append(neg)
    #                 selected_ids.add(neg.patent_id)
    #
    #     # 第2步：如果还不够，尝试同小类不同大组负例
    #     if len(negatives) < num_samples and getattr(self.config, 'same_subclass_diff_group_enabled', True):
    #         same_subclass_negs = self._sample_same_subclass_different_group(
    #             anchor_patent,
    #             num_samples - len(negatives)
    #         )
    #
    #         for neg in same_subclass_negs:
    #             if neg.patent_id not in selected_ids:
    #                 neg.negative_type = 'same_subclass_diff_group'
    #                 negatives.append(neg)
    #                 selected_ids.add(neg.patent_id)
    #
    #     # 第3步：如果还不够，使用随机负例补足
    #     if len(negatives) < num_samples and getattr(self.config, 'random_enabled', True):
    #         random_negs = self._sample_random(
    #             anchor_patent,
    #             num_samples - len(negatives),
    #             exclude_ids=list(selected_ids)
    #         )
    #
    #         for neg in random_negs:
    #             neg.negative_type = 'random'
    #
    #         negatives.extend(random_negs)
    #
    #     # 记录采样统计
    #     self._log_sampling_stats(anchor_patent, negatives)
    #
    #     return negatives[:num_samples]

    def _sample_same_group_priority(self, anchor_patent, num_samples: int) -> List[NegativeSample]:
        """同大组优先策略（简化版）"""
        negatives = []
        selected_ids = {anchor_patent.id}
        anchor_groups = self.patent_to_groups.get(anchor_patent.id, [])

        # 1. 尽可能多地从同大组中取
        same_group_target = min(num_samples, len(anchor_groups))

        for group in anchor_groups:
            if len(negatives) >= same_group_target:
                break

            group_patents = self.group_to_patents.get(group, [])
            other_patents = [p for p in group_patents if p.id != anchor_patent.id]

            if other_patents:
                # 随机选择一个同大组负例
                patent = random.choice(other_patents)
                similarity = self._calculate_text_similarity(
                    anchor_patent.abstract,
                    patent.abstract
                )

                # 过滤过于相似的专利
                similarity_threshold = getattr(self.config, 'same_group_similarity_threshold', 0.8)
                if similarity < similarity_threshold:
                    negatives.append(NegativeSample(
                        patent_id=patent.id,
                        patent_title=patent.title,
                        patent_abstract=patent.abstract,
                        ipc_subclasses=list(patent.ipc_subclasses),
                        ipc_groups=list(patent.ipc_groups),
                        similarity_score=similarity,
                        negative_type='same_group'
                    ))
                    selected_ids.add(patent.id)

        # 2. 如果还不够，用随机负例补足
        if len(negatives) < num_samples:
            exclude_ids = list(selected_ids)
            random_negs = self._sample_random(
                anchor_patent,
                num_samples - len(negatives),
                exclude_ids=exclude_ids
            )

            for neg in random_negs:
                neg.negative_type = 'random'

            negatives.extend(random_negs)

        return negatives[:num_samples]

    def _sample_same_group_negatives(self, anchor_patent, num_samples: int) -> List[NegativeSample]:
        """同大组不同小组负例采样：技术领域相关（同大组）但技术不同（不同小组）。排除与锚点任一IPC完全相同的专利。"""
        anchor_id = anchor_patent.id
        anchor_groups = self.patent_to_groups.get(anchor_id, [])
        anchor_full_ipcs = set(c.strip() for c in getattr(anchor_patent, "ipc_codes", []))

        candidates = []

        for group in anchor_groups:
            group_patents = self.group_to_patents.get(group, [])

            if len(group_patents) <= 1:
                continue

            other_patents = [p for p in group_patents if p.id != anchor_id]

            for patent in other_patents:
                # 严格同大组不同小组：排除与锚点任一IPC完全相同（同小组）的专利
                neg_ipcs = set(c.strip() for c in getattr(patent, "ipc_codes", []))
                if anchor_full_ipcs & neg_ipcs:
                    continue

                similarity = self._calculate_text_similarity(
                    anchor_patent.abstract,
                    patent.abstract
                )

                similarity_threshold = getattr(self.config, 'same_group_similarity_threshold', 0.8)
                if similarity < similarity_threshold:
                    candidates.append(NegativeSample(
                        patent_id=patent.id,
                        patent_title=patent.title,
                        patent_abstract=patent.abstract,
                        ipc_subclasses=list(patent.ipc_subclasses),
                        ipc_groups=list(patent.ipc_groups),
                        similarity_score=similarity
                    ))

        # 如果候选不足，返回空列表
        min_samples_required = getattr(self.config, 'same_group_min_samples_required', 1)
        if len(candidates) < min_samples_required:
            return []

        # 选择策略：优先选择中等相似度的负例
        if len(candidates) > num_samples:
            # 按相似度排序，选择中间范围的（避免过于相似或过于不相似）
            candidates.sort(key=lambda x: abs(x.similarity_score - 0.5))
            return candidates[:num_samples]
        else:
            return candidates

    def _sample_same_subclass_different_group(self, anchor_patent, num_samples: int) -> List[NegativeSample]:
        """跨大组同小类采样策略"""
        anchor_id = anchor_patent.id
        anchor_subclasses = anchor_patent.ipc_subclasses
        anchor_groups = anchor_patent.ipc_groups

        candidates = []

        # 对于每个小类，寻找不同大组的专利
        for subclass in anchor_subclasses:
            # 获取同小类的所有专利
            subclass_patents = self.data_loader.get_patents_by_subclass(subclass, exclude_id=anchor_id)

            for patent in subclass_patents:
                # 检查是否属于不同大组
                patent_groups = patent.ipc_groups
                if not patent_groups:
                    continue

                # 计算大组重叠度
                group_overlap = len(anchor_groups.intersection(patent_groups))

                # 如果属于不同大组，添加到候选
                if group_overlap == 0:
                    # 计算文本相似度（可选）
                    similarity = self._calculate_text_similarity(
                        anchor_patent.abstract,
                        patent.abstract
                    )

                    # 避免过于相似的专利
                    group_similarity_threshold = getattr(self.config, 'group_similarity_threshold', 0.8)
                    if similarity < group_similarity_threshold:
                        candidates.append(NegativeSample(
                            patent_id=patent.id,
                            patent_title=patent.title,
                            patent_abstract=patent.abstract,
                            ipc_subclasses=list(patent.ipc_subclasses),
                            ipc_groups=list(patent.ipc_groups),
                            similarity_score=similarity
                        ))

        # 如果没有足够的跨大组样本，放宽条件
        if len(candidates) < num_samples:
            logger.debug(f"跨大组样本不足 ({len(candidates)}/{num_samples})，放宽条件...")
            candidates.extend(self._sample_same_subclass_same_group(anchor_patent, num_samples * 2))

        # 去重
        unique_candidates = self._deduplicate_candidates(candidates)

        # 按相似度排序，选择相似度适中的作为负样本
        # 过于相似的可能是正样本，过于不同的学习效果差
        if len(unique_candidates) > num_samples:
            # 按相似度排序
            unique_candidates.sort(key=lambda x: abs(x.similarity_score - 0.5))
            selected = unique_candidates[:num_samples]
        else:
            selected = unique_candidates

        logger.debug(f"为专利 {anchor_id[:20]}... 选择了 {len(selected)} 个负样本")
        return selected

    def _sample_same_subclass_same_group(self, anchor_patent, num_samples: int) -> List[NegativeSample]:
        """同小类同大组不同小组采样（作为后备）。排除与锚点任一IPC完全相同的专利。"""
        anchor_id = anchor_patent.id
        anchor_groups = anchor_patent.ipc_groups
        anchor_full_ipcs = set(c.strip() for c in getattr(anchor_patent, "ipc_codes", []))

        candidates = []

        for group in anchor_groups:
            group_patents = self.data_loader.get_patents_by_group(group, exclude_id=anchor_id)

            for patent in group_patents:
                neg_ipcs = set(c.strip() for c in getattr(patent, "ipc_codes", []))
                if anchor_full_ipcs & neg_ipcs:
                    continue

                similarity = self._calculate_text_similarity(
                    anchor_patent.abstract,
                    patent.abstract
                )
                if similarity < 0.8:
                    candidates.append(NegativeSample(
                        patent_id=patent.id,
                        patent_title=patent.title,
                        patent_abstract=patent.abstract,
                        ipc_subclasses=list(patent.ipc_subclasses),
                        ipc_groups=list(patent.ipc_groups),
                        similarity_score=similarity
                    ))

        return candidates

    def _sample_mixed(self, anchor_patent, num_samples: int) -> List[NegativeSample]:
        """混合策略采样"""
        # 计算每种策略的样本数
        total_weight = (
                getattr(self.config, 'same_subclass_weight', 1.0) +
                getattr(self.config, 'hard_negative_weight', 0.0) +
                getattr(self.config, 'random_negative_weight', 1.0)
        )

        n_same_subclass = int(num_samples * getattr(self.config, 'same_subclass_weight', 1.0) / total_weight)
        n_hard = int(num_samples * getattr(self.config, 'hard_negative_weight', 0.0) / total_weight)
        n_random = num_samples - n_same_subclass - n_hard

        negatives = []

        # 采样同小类不同大组
        if n_same_subclass > 0:
            same_subclass = self._sample_same_subclass_different_group(anchor_patent, n_same_subclass)
            for neg in same_subclass:
                neg.negative_type = 'same_subclass_diff_group'
            negatives.extend(same_subclass)

        # 采样硬负样本
        if n_hard > 0:
            hard = self._sample_hard(anchor_patent, n_hard)
            for neg in hard:
                neg.negative_type = 'hard'
            negatives.extend(hard)

        # 采样随机负样本
        if n_random > 0:
            random_samples = self._sample_random(anchor_patent, n_random)
            for neg in random_samples:
                neg.negative_type = 'random'
            negatives.extend(random_samples)

        # 去重
        unique_negatives = self._deduplicate_candidates(negatives)

        return unique_negatives[:num_samples]

    def _sample_random(self, anchor_patent, num_samples: int,
                       exclude_ids: Optional[List[str]] = None) -> List[NegativeSample]:
        """随机采样（增强版，支持排除列表）"""
        anchor_id = anchor_patent.id
        if exclude_ids is None:
            exclude_ids = [anchor_id]

        all_patents = [p for p in self.data_loader.patents
                       if p.id not in exclude_ids]

        if not all_patents:
            return []

        if len(all_patents) > num_samples:
            selected_patents = random.sample(all_patents, num_samples)
        else:
            selected_patents = all_patents

        candidates = []
        for patent in selected_patents:
            similarity = self._calculate_text_similarity(
                anchor_patent.abstract,
                patent.abstract
            )

            candidates.append(NegativeSample(
                patent_id=patent.id,
                patent_title=patent.title,
                patent_abstract=patent.abstract,
                ipc_subclasses=list(patent.ipc_subclasses),
                ipc_groups=list(patent.ipc_groups),
                similarity_score=similarity
            ))

        return candidates

    def _sample_hard(self, anchor_patent, num_samples: int) -> List[NegativeSample]:
        """硬负样本采样"""
        anchor_id = anchor_patent.id
        anchor_text = anchor_patent.abstract

        # 收集候选专利
        candidates = []

        for patent in self.data_loader.patents:
            if patent.id == anchor_id:
                continue

            # 计算相似度
            similarity = self._calculate_text_similarity(anchor_text, patent.abstract)

            # 选择相似度在一定范围内的作为硬负样本
            semantic_threshold = getattr(self.config, 'semantic_similarity_threshold', 0.3)
            group_threshold = getattr(self.config, 'group_similarity_threshold', 0.8)

            if semantic_threshold <= similarity <= group_threshold:
                candidates.append(NegativeSample(
                    patent_id=patent.id,
                    patent_title=patent.title,
                    patent_abstract=patent.abstract,
                    ipc_subclasses=list(patent.ipc_subclasses),
                    ipc_groups=list(patent.ipc_groups),
                    similarity_score=similarity
                ))

        # 按相似度排序，选择最相似的作为硬负样本
        candidates.sort(key=lambda x: x.similarity_score, reverse=True)

        return candidates[:num_samples]

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（简化版）"""
        # 使用缓存
        cache_key = f"{hashlib.md5(text1.encode()).hexdigest()}_{hashlib.md5(text2.encode()).hexdigest()}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        # 计算Jaccard相似度（基于字符）
        set1 = set(text1)
        set2 = set(text2)

        if not set1 or not set2:
            similarity = 0.0
        else:
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            similarity = intersection / union if union > 0 else 0.0

        # 缓存结果
        self.cache[cache_key] = similarity

        return similarity

    def _deduplicate_candidates(self, candidates: List[NegativeSample]) -> List[NegativeSample]:
        """去重候选样本"""
        seen_ids = set()
        unique_candidates = []

        for candidate in candidates:
            if candidate.patent_id not in seen_ids:
                seen_ids.add(candidate.patent_id)
                unique_candidates.append(candidate)

        return unique_candidates

    def _log_sampling_stats(self, anchor_patent, negatives: List[NegativeSample]):
        """记录采样统计信息"""
        if not negatives:
            return

        # 统计负例类型分布
        type_counts = defaultdict(int)
        for neg in negatives:
            if hasattr(neg, 'negative_type'):
                type_counts[neg.negative_type] += 1

        # 计算平均相似度
        avg_similarity = np.mean([n.similarity_score for n in negatives]) if negatives else 0

        logger.debug(
            f"专利 {anchor_patent.id[:15]}... 负例采样统计: "
            f"总数={len(negatives)}, "
            f"类型分布={dict(type_counts)}, "
            f"平均相似度={avg_similarity:.3f}"
        )

    def _sample_technical_negative(self, anchor_patent, num_samples: int) -> List[NegativeSample]:
        """技术性负样本：相同技术领域但不同技术方案"""
        anchor_id = anchor_patent.id
        anchor_abstract = anchor_patent.abstract

        # 提取关键词（简化版）
        keywords = self._extract_keywords(anchor_abstract)

        candidates = []

        # 寻找具有相似关键词但不同IPC的专利
        for patent in self.data_loader.patents:
            if patent.id == anchor_id:
                continue

            # 检查IPC是否有重叠（技术领域相似）
            ipc_overlap = len(anchor_patent.ipc_subclasses.intersection(patent.ipc_subclasses))
            if ipc_overlap == 0:
                continue

            # 检查是否包含相似关键词
            patent_keywords = self._extract_keywords(patent.abstract)
            keyword_overlap = len(set(keywords).intersection(set(patent_keywords)))

            # 技术相似但方案不同（关键词重叠度适中）
            if 1 <= keyword_overlap <= 3:  # 可调整
                similarity = self._calculate_text_similarity(anchor_abstract, patent.abstract)

                # 避免过于相似
                if 0.3 <= similarity <= 0.7:  # 中等相似度
                    candidates.append(NegativeSample(
                        patent_id=patent.id,
                        patent_title=patent.title,
                        patent_abstract=patent.abstract,
                        ipc_subclasses=list(patent.ipc_subclasses),
                        ipc_groups=list(patent.ipc_groups),
                        similarity_score=similarity,
                        negative_type='technical'
                    ))

        # 选择相似度最接近0.5的作为难负样本
        if candidates:
            candidates.sort(key=lambda x: abs(x.similarity_score - 0.5))
            return candidates[:min(num_samples, len(candidates))]

        return []

    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """提取关键词（简化版）"""
        # 移除停用词和标点
        import jieba
        stopwords = set(['的', '了', '在', '是', '我', '有', '和', '就',
                         '不', '人', '都', '一', '一个', '上', '也', '很',
                         '到', '说', '要', '去', '你', '会', '着', '没有',
                         '看', '好', '自己', '这', '方法', '系统', '装置'])

        words = jieba.lcut(text)
        # 过滤停用词和短词
        filtered_words = [w for w in words if len(w) > 1 and w not in stopwords]

        # 简单频率统计
        from collections import Counter
        word_freq = Counter(filtered_words)

        return [word for word, _ in word_freq.most_common(max_keywords)]

    # 修改 _sample_optimized_mixed 方法，加入技术性负样本
    def _sample_optimized_mixed(self, anchor_patent, num_samples: int) -> List[NegativeSample]:
        """优化版混合策略：包含技术性负样本"""
        anchor_id = anchor_patent.id
        anchor_groups = self.patent_to_groups.get(anchor_id, [])

        negatives = []
        selected_ids = {anchor_id}

        # 第1步：技术性负样本（最难）
        technical_negs = self._sample_technical_negative(
            anchor_patent,
            min(num_samples // 2, 2)  # 最多取2个技术性负样本
        )
        for neg in technical_negs:
            if neg.patent_id not in selected_ids:
                neg.negative_type = 'technical'
                negatives.append(neg)
                selected_ids.add(neg.patent_id)

        # 第2步：同大组负例
        if len(negatives) < num_samples and getattr(self.config, 'same_group_enabled', True) and anchor_groups:
            same_group_negs = self._sample_same_group_negatives(
                anchor_patent,
                min(num_samples - len(negatives), getattr(self.config, 'same_group_max_samples', 2))
            )
            for neg in same_group_negs:
                if neg.patent_id not in selected_ids:
                    neg.negative_type = 'same_group'
                    negatives.append(neg)
                    selected_ids.add(neg.patent_id)

        # 第3步：同小类不同大组
        if len(negatives) < num_samples and getattr(self.config, 'same_subclass_diff_group_enabled', True):
            same_subclass_negs = self._sample_same_subclass_different_group(
                anchor_patent,
                num_samples - len(negatives)
            )
            for neg in same_subclass_negs:
                if neg.patent_id not in selected_ids:
                    neg.negative_type = 'same_subclass_diff_group'
                    negatives.append(neg)
                    selected_ids.add(neg.patent_id)

        # 第4步：随机负例补足
        if len(negatives) < num_samples and getattr(self.config, 'random_enabled', True):
            random_negs = self._sample_random(
                anchor_patent,
                num_samples - len(negatives),
                exclude_ids=list(selected_ids)
            )
            for neg in random_negs:
                neg.negative_type = 'random'
            negatives.extend(random_negs)

        # 记录采样统计
        self._log_sampling_stats(anchor_patent, negatives)

        return negatives[:num_samples]

def create_negative_sampler(data_loader, config) -> NegativeSampler:
    """创建负样本采样器"""
    return NegativeSampler(data_loader, config)