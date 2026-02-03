# data_loader.py
import os
import json
import pickle
import random
import re
import glob
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


# 更新 data_loader.py 中的 PatentRecord 类
@dataclass
class PatentRecord:
    """专利记录数据类"""
    id: str  # 专利号
    title: str  # 标题
    abstract: str  # 摘要
    ipc_codes: List[str]  # IPC分类代码
    ipc_subclasses: Set[str]  # IPC小类（前4位）
    ipc_groups: Set[str]  # IPC大组（前7位）
    keywords: List[str]  # 关键词
    publication_date: str  # 公布日期
    n_claims: int  # 权利要求数
    n_citations: int  # 引用数
    main_ipc_code: str = ""  # 主要IPC代码
    main_ipc_subclass: str = ""  # 主要小类
    main_ipc_group: str = ""  # 主要大组

    @classmethod
    def from_dict(cls, data: Dict, ipc_strategy: str = "first") -> Optional['PatentRecord']:
        """从字典创建专利记录 - 适配新JSON结构"""
        try:
            from ipc_processor import IPCProcessor

            # 提取必要字段
            patent_id = data.get('publication_number', '')
            if not patent_id:
                logger.debug(f"专利无publication_number字段")
                return None

            # 提取标题
            title = str(data.get('title', '')).strip()

            # 提取摘要
            abstract = str(data.get('abstract', '')).strip()

            if not title or not abstract:
                logger.debug(f"专利 {patent_id} 标题或摘要为空")
                return None

            # 提取分类 - 处理字典列表格式
            classifications_data = data.get('classifications', [])
            classifications = []

            if isinstance(classifications_data, list):
                for item in classifications_data:
                    if isinstance(item, dict):
                        # 提取code字段
                        code = item.get('code')
                        if code:
                            classifications.append(str(code).strip())
                    elif isinstance(item, str):
                        classifications.append(item.strip())
            elif isinstance(classifications_data, dict):
                # 如果是字典，尝试提取值
                for key, value in classifications_data.items():
                    if isinstance(value, (list, dict)):
                        # 进一步处理嵌套结构
                        continue
                    else:
                        classifications.append(str(value).strip())
            elif classifications_data:
                # 其他类型直接转换
                classifications.append(str(classifications_data).strip())

            if not classifications:
                logger.debug(f"专利 {patent_id} 无IPC分类")
                return None

            # 获取主要IPC代码
            main_ipc_code = IPCProcessor.get_main_ipc_code(classifications, ipc_strategy)
            if not main_ipc_code:
                logger.debug(f"专利 {patent_id} 无有效IPC分类")
                return None

            # 提取所有IPC代码
            ipc_codes = IPCProcessor.extract_all_ipc_codes(classifications)

            # 提取小类和大组
            use_all_classifications = (ipc_strategy == "all")
            ipc_subclasses, ipc_groups = IPCProcessor.extract_subclasses_and_groups(
                ipc_codes, use_all=use_all_classifications
            )

            # 提取主要小类和大组
            main_ipc_components = IPCProcessor.extract_ipc_components(main_ipc_code)
            main_ipc_subclass = main_ipc_components.get('subclass', '')
            main_ipc_group = main_ipc_components.get('main_group_code', '')

            # 提取关键词 - 根据你的数据结构调整
            keywords = []
            keywords_data = data.get('prior_art_keywords', []) or data.get('keywords', [])

            if isinstance(keywords_data, list):
                for item in keywords_data:
                    if isinstance(item, dict):
                        # 如果关键词也是字典格式
                        for key, value in item.items():
                            keywords.append(str(value).strip())
                    else:
                        keywords.append(str(item).strip())
            elif isinstance(keywords_data, dict):
                keywords = [str(v).strip() for v in keywords_data.values() if v]
            elif keywords_data:
                keywords = [str(keywords_data).strip()]

            # 提取其他字段
            publication_date = str(data.get('publication_date', '')).strip()
            n_claims = int(data.get('n_claims', 0) or 0)
            n_citations = int(data.get('n_citations', 0) or 0)

            return cls(
                id=patent_id,
                title=title,
                abstract=abstract,
                ipc_codes=ipc_codes,
                ipc_subclasses=ipc_subclasses,
                ipc_groups=ipc_groups,
                keywords=keywords,
                publication_date=publication_date,
                n_claims=n_claims,
                n_citations=n_citations,
                main_ipc_code=main_ipc_code,
                main_ipc_subclass=main_ipc_subclass,
                main_ipc_group=main_ipc_group
            )
        except Exception as e:
            logger.error(f"创建专利记录失败: {e}, 数据: {data.get('publication_number', 'unknown')}")
            import traceback
            logger.error(traceback.format_exc())
            return None


class PatentDataLoader:
    """专利数据加载器"""

    def __init__(self, config):
        self.config = config.data
        self.patents: List[PatentRecord] = []
        self.subclass_to_patents: Dict[str, List[str]] = defaultdict(list)
        self.group_to_patents: Dict[str, List[str]] = defaultdict(list)
        self.patent_dict: Dict[str, PatentRecord] = {}
        self.lock = threading.Lock()
        self.processed_count = 0
        self.error_count = 0

    def load_from_folder(self, folder_path: str, max_samples: Optional[int] = None) -> List[PatentRecord]:
        """从文件夹加载专利数据"""
        logger.info(f"从 {folder_path} 加载专利数据...")

        # 获取所有json文件
        json_files = glob.glob(os.path.join(folder_path, "*.json"))

        if not json_files:
            # 尝试在子文件夹中查找
            json_files = glob.glob(os.path.join(folder_path, "**", "*.json"), recursive=True)

        if not json_files:
            raise FileNotFoundError(f"在 {folder_path} 中未找到json文件")

        logger.info(f"找到 {len(json_files)} 个json文件")

        if max_samples and max_samples < len(json_files):
            json_files = json_files[:max_samples]
            logger.info(f"采样 {max_samples} 个文件")

        patents = []

        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for json_file in json_files:
                futures.append(executor.submit(self._parse_single_file, json_file))

            # 收集结果
            for future in tqdm(futures, desc="加载专利", total=len(futures)):
                try:
                    patent = future.result()
                    if patent:
                        patents.append(patent)
                        self.processed_count += 1
                    else:
                        self.error_count += 1
                except Exception as e:
                    logger.error(f"处理文件失败: {e}")
                    self.error_count += 1

        logger.info(f"成功加载 {self.processed_count} 个专利，失败 {self.error_count} 个")
        return patents

    def _parse_single_file(self, file_path: str) -> Optional[PatentRecord]:
        """解析单个专利文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 创建专利记录
            patent = PatentRecord.from_dict(data)
            return patent

        except UnicodeDecodeError:
            # 尝试其他编码
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    data = json.load(f)
                    patent = PatentRecord.from_dict(data)
                    return patent
            except:
                logger.warning(f"编码问题无法解析文件: {file_path}")
                return None

        except Exception as e:
            logger.debug(f"解析文件 {file_path} 失败: {e}")

        return None

    def process_patents(self, patents: List[PatentRecord]):
        """处理专利数据，构建索引"""
        logger.info("处理专利数据，构建IPC索引...")

        self.patents = patents
        self.patent_dict = {patent.id: patent for patent in patents}

        # 构建IPC索引
        for patent in tqdm(patents, desc="构建索引"):
            # 小类索引
            for subclass in patent.ipc_subclasses:
                self.subclass_to_patents[subclass].append(patent.id)

            # 大组索引
            for group in patent.ipc_groups:
                self.group_to_patents[group].append(patent.id)

        # 过滤样本数不足的IPC小类
        if self.config.min_patents_per_ipc > 1:
            self._filter_ipc_classes()

        # 打印统计信息
        self._print_statistics()

        # 保存索引
        self.save_indexes()

    def _filter_ipc_classes(self):
        """过滤专利数不足的IPC小类。小数据时放宽条件（min_patents_per_ipc=1 时不按数量过滤）。"""
        min_n = max(1, self.config.min_patents_per_ipc)
        valid_subclasses = []
        for subclass, patent_ids in self.subclass_to_patents.items():
            if len(patent_ids) >= min_n:
                valid_subclasses.append(subclass)

        filtered_subclasses = {s: self.subclass_to_patents[s] for s in valid_subclasses}
        self.subclass_to_patents = defaultdict(list, filtered_subclasses)

        group_min = max(1, self.config.min_patents_per_ipc // 2)
        filtered_groups = {
            g: ids for g, ids in self.group_to_patents.items()
            if len(ids) >= group_min
        }
        self.group_to_patents = defaultdict(list, filtered_groups)

        logger.info(f"过滤后有效IPC小类数量: {len(self.subclass_to_patents)}")
        logger.info(f"过滤后有效IPC大组数量: {len(self.group_to_patents)}")

    def _print_statistics(self):
        """打印数据统计信息"""
        logger.info("=" * 50)
        logger.info("数据统计信息:")
        logger.info(f"专利总数: {len(self.patents)}")
        logger.info(f"IPC小类数量: {len(self.subclass_to_patents)}")
        logger.info(f"IPC大组数量: {len(self.group_to_patents)}")

        # 每个IPC小类的专利数分布
        subclass_counts = [len(ids) for ids in self.subclass_to_patents.values()]
        if subclass_counts:
            logger.info(f"每个IPC小类平均专利数: {sum(subclass_counts) / len(subclass_counts):.2f}")
            logger.info(f"每个IPC小类最大专利数: {max(subclass_counts)}")
            logger.info(f"每个IPC小类最小专利数: {min(subclass_counts)}")

        # 专利标题长度统计
        title_lengths = [len(p.title) for p in self.patents]
        if title_lengths:
            logger.info(f"标题平均长度: {sum(title_lengths) / len(title_lengths):.2f} 字符")

        # 摘要长度统计
        abstract_lengths = [len(p.abstract) for p in self.patents]
        if abstract_lengths:
            logger.info(f"摘要平均长度: {sum(abstract_lengths) / len(abstract_lengths):.2f} 字符")

        logger.info("=" * 50)

    def save_indexes(self):
        """保存索引到文件"""
        os.makedirs(self.config.processed_data_dir, exist_ok=True)

        # 保存IPC索引
        index_data = {
            'subclass_to_patents': dict(self.subclass_to_patents),
            'group_to_patents': dict(self.group_to_patents),
            'patent_ids': [p.id for p in self.patents]
        }

        index_path = os.path.join(self.config.processed_data_dir, "ipc_indexes.pkl")
        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f)

        logger.info(f"IPC索引已保存到 {index_path}")

    def load_indexes(self) -> bool:
        """从文件加载索引"""
        index_path = os.path.join(self.config.processed_data_dir, "ipc_indexes.pkl")

        if os.path.exists(index_path):
            try:
                with open(index_path, 'rb') as f:
                    index_data = pickle.load(f)

                self.subclass_to_patents = defaultdict(list, index_data['subclass_to_patents'])
                self.group_to_patents = defaultdict(list, index_data['group_to_patents'])

                logger.info(f"从 {index_path} 加载了IPC索引")
                logger.info(f"索引中包含 {len(self.subclass_to_patents)} 个IPC小类")
                return True
            except Exception as e:
                logger.error(f"加载索引失败: {e}")

        return False

    def get_patent_by_id(self, patent_id: str) -> Optional[PatentRecord]:
        """根据ID获取专利"""
        return self.patent_dict.get(patent_id)

    def get_patents_by_subclass(self, subclass: str, exclude_id: Optional[str] = None) -> List[PatentRecord]:
        """根据IPC小类获取专利"""
        patent_ids = self.subclass_to_patents.get(subclass, [])
        patents = []

        for pid in patent_ids:
            if exclude_id and pid == exclude_id:
                continue
            patent = self.get_patent_by_id(pid)
            if patent:
                patents.append(patent)

        return patents

    def get_patents_by_group(self, group: str, exclude_id: Optional[str] = None) -> List[PatentRecord]:
        """根据IPC大组获取专利"""
        patent_ids = self.group_to_patents.get(group, [])
        patents = []

        for pid in patent_ids:
            if exclude_id and pid == exclude_id:
                continue
            patent = self.get_patent_by_id(pid)
            if patent:
                patents.append(patent)

        return patents

    def get_random_patent(self, exclude_id: Optional[str] = None) -> Optional[PatentRecord]:
        """随机获取一个专利"""
        candidates = self.patents
        if exclude_id:
            candidates = [p for p in self.patents if p.id != exclude_id]

        return random.choice(candidates) if candidates else None

    def get_all_subclasses(self) -> List[str]:
        """获取所有IPC小类"""
        return list(self.subclass_to_patents.keys())


def create_data_loader(config, load_existing: bool = True) -> PatentDataLoader:
    """创建数据加载器。专利始终从 raw 目录的 JSON 加载；load_existing 仅影响是否尝试复用 IPC 索引（若存在）。"""
    loader = PatentDataLoader(config)
    if load_existing and loader.load_indexes():
        logger.info("已加载现有 IPC 索引（专利仍需从 raw 目录加载）")
    else:
        logger.info("未找到现有索引，将从头构建")
    return loader