# ipc_processor.py
import re
from typing import List, Set, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class IPCProcessor:
    """IPC分类处理器"""

    @staticmethod
    def get_main_ipc_code(classifications: List[str], strategy: str = "first") -> str:
        """获取主要IPC代码"""
        if not classifications:
            return ""

        if strategy == "first":
            # 只使用第一个分类
            return classifications[0] if classifications else ""

        elif strategy == "first_g":
            # 优先使用G部分类
            for code in classifications:
                if code.startswith('G'):
                    return code
            # 没有G类，返回第一个
            return classifications[0] if classifications else ""

        elif strategy == "first_h":
            # 优先使用H部分类
            for code in classifications:
                if code.startswith('H'):
                    return code
            # 没有H类，返回第一个
            return classifications[0] if classifications else ""

        elif strategy == "all":
            # 返回第一个分类
            return classifications[0] if classifications else ""

        else:
            # 默认返回第一个
            return classifications[0] if classifications else ""

    @staticmethod
    def extract_all_ipc_codes(classifications: List[str]) -> List[str]:
        """提取所有IPC代码"""
        return classifications

    @staticmethod
    def extract_subclasses_and_groups(ipc_codes: List[str], use_all: bool = False) -> tuple[Set[str], Set[str]]:
        """提取IPC小类和大组"""
        subclasses = set()
        groups = set()

        # 限制每个专利最多使用3个大组
        group_count = 0

        for code in ipc_codes:
            # 清理代码（移除空格和斜杠）
            code = code.strip().replace('/', '')

            # 提取小类（前4位）
            if len(code) >= 4:
                subclass = code[:4]
                subclasses.add(subclass)

            # 提取大组（前7位）
            if len(code) >= 7:
                group = code[:7]
                if group_count < 3:  # 最多保留3个大组
                    groups.add(group)
                    group_count += 1

        return subclasses, groups

    @staticmethod
    def extract_ipc_components(ipc_code: str) -> Dict[str, str]:
        """提取IPC代码的各个组成部分"""
        components = {
            'section': '',
            'class': '',
            'subclass': '',
            'main_group': '',
            'main_group_code': '',
            'subgroup': ''
        }

        if not ipc_code:
            return components

        # 清理代码
        code = ipc_code.strip()

        # 提取部（1位字母）
        if len(code) >= 1:
            components['section'] = code[0]

        # 提取大类（2位数字）
        if len(code) >= 3:
            components['class'] = code[:3]

        # 提取小类（4位：字母+数字）
        if len(code) >= 4:
            components['subclass'] = code[:4]

        # 提取主组（找到斜杠前的部分）
        if '/' in code:
            main_group_part = code.split('/')[0]
            if len(main_group_part) >= 7:
                components['main_group_code'] = main_group_part[:7]
                # 提取主组数字
                if len(main_group_part) >= 4:
                    main_group_num = main_group_part[4:]
                    components['main_group'] = main_group_num

        # 提取分组（斜杠后的部分）
        if '/' in code:
            parts = code.split('/')
            if len(parts) > 1:
                components['subgroup'] = parts[1]

        return components

    @staticmethod
    def is_valid_ipc_code(ipc_code: str) -> bool:
        """验证IPC代码格式"""
        if not ipc_code:
            return False

        # 基本格式检查
        pattern = r'^[A-H][0-9]{2}[A-Z][0-9/]+$'
        return bool(re.match(pattern, ipc_code))

    @staticmethod
    def get_ipc_hierarchy(ipc_code: str) -> Dict[str, str]:
        """获取IPC层级关系"""
        components = IPCProcessor.extract_ipc_components(ipc_code)

        return {
            'full_code': ipc_code,
            'section': components['section'],
            'class': components['class'],
            'subclass': components['subclass'],
            'main_group': components['main_group_code'],
            'description': f"{components['section']}部-{components['class']}类-{components['subclass']}小类"
        }