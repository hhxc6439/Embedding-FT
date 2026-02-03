# query_generator.py
import re
import random
import time
import requests
import logging
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import json
import torch

logger = logging.getLogger(__name__)


class SimpleQueryGenerator:
    """简单的规则查询生成器（本地）"""

    def __init__(self, config):
        self.config = config.query
        self.strategy = self.config.strategy

    def generate_from_title(self, title: str) -> str:
        """从标题生成查询"""
        if not title:
            return "该专利涉及什么技术？"

        # 清理标题
        title = title.strip()

        # 移除常见的专利前缀
        patterns_to_remove = [
            r'^一种',
            r'^一种用于',
            r'^用于',
            r'^基于',
            r'^关于',
            r'^涉及',
            r'^专利号',
            r'^公开号',
            r'的装置$',
            r'的方法$',
            r'的系统$',
            r'的制备方法$'
        ]

        for pattern in patterns_to_remove:
            title = re.sub(pattern, '', title)

        title = title.strip()

        # 使用模板生成查询
        if hasattr(self.config, 'query_templates') and self.config.query_templates:
            template = random.choice(self.config.query_templates)
            query = template.format(title=title)
        else:
            # 默认生成方式
            if len(title) > 20:
                query = f"什么是{title}？"
            else:
                query = f"如何实现{title}？"

        return query

    def generate_from_abstract(self, abstract: str, title: str = "") -> str:
        """从摘要生成查询"""
        if not abstract:
            return self.generate_from_title(title) if title else "该专利涉及什么技术？"

        # 截取摘要前100字作为关键信息
        abstract_short = abstract[:100]

        # 尝试提取技术问题
        tech_problem_patterns = [
            r'提供(?:了)?(?:一种)?([^，。]+?)(?:的方法|装置|系统|解决方案)',
            r'解决(?:了)?(?:现有技术中)?([^，。]+?)的问题',
            r'旨在(?:解决|提供)([^，。]+?)',
            r'涉及(?:一种)?([^，。]+?)',
        ]

        for pattern in tech_problem_patterns:
            match = re.search(pattern, abstract_short)
            if match:
                tech_part = match.group(1).strip()
                if 5 < len(tech_part) < 50:
                    return f"如何{tech_part}？"

        # 使用标题生成
        if title:
            return self.generate_from_title(title)

        # 生成通用查询
        return "该专利涉及什么技术方案？"

    def generate_query(self, patent) -> str:
        """生成查询 - 修复：使用属性访问而非get方法"""
        # 从PatentRecord对象获取属性
        title = patent.title
        abstract = patent.abstract

        if self.strategy == 'title_based':
            return self.generate_from_title(title)
        elif self.strategy == 'abstract_based':
            return self.generate_from_abstract(abstract, title)
        elif self.strategy == 'hybrid':
            # 先尝试用摘要，失败则用标题
            query = self.generate_from_abstract(abstract, title)
            if len(query) > 15:  # 查询质量判断
                return query
            return self.generate_from_title(title)
        else:
            return self.generate_from_title(title)

    def generate_batch(self, patents: List) -> List[Dict]:
        """批量生成查询"""
        logger.info(f"开始为 {len(patents)} 个专利生成查询...")

        samples = []

        for patent in tqdm(patents, desc="生成查询", leave=False):
            query = self.generate_query(patent)

            # 创建样本
            sample = {
                'query': query,
                'patent_id': patent.id,
                'patent_title': patent.title,
                'patent_abstract': patent.abstract,
                'ipc_subclasses': list(patent.ipc_subclasses),
                'ipc_groups': list(patent.ipc_groups)
            }

            samples.append(sample)

        logger.info(f"生成了 {len(samples)} 个查询样本")
        return samples


class LLMQueryGenerator(SimpleQueryGenerator):
    """基于LLM API的查询生成器"""

    def __init__(self, config):
        super().__init__(config)
        self.use_llm = self.config.use_llm
        self.api_key = self.config.llm_api_key
        self.base_url = getattr(self.config, 'llm_base_url', 'https://api.openai.com/v1')
        self.model = getattr(self.config, 'llm_model', 'gpt-3.5-turbo')
        self.max_retries = 3
        self.retry_delay = 2

    def _call_llm_api(self, prompt: str) -> Optional[str]:
        """调用LLM API"""
        if not self.api_key:
            logger.warning("LLM API密钥未配置，使用规则生成")
            return None

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是一个专利检索专家，擅长从专利摘要生成高质量的技术查询问题。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 100
        }

        for retry in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content'].strip()
                else:
                    logger.warning(f"API调用失败 (尝试 {retry + 1}/{self.max_retries}): {response.status_code}")
                    if retry < self.max_retries - 1:
                        time.sleep(self.retry_delay)

            except requests.exceptions.RequestException as e:
                logger.warning(f"API请求异常 (尝试 {retry + 1}/{self.max_retries}): {e}")
                if retry < self.max_retries - 1:
                    time.sleep(self.retry_delay)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON解析失败 (尝试 {retry + 1}/{self.max_retries}): {e}")
                if retry < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        return None

    def generate_query_with_llm(self, patent) -> Optional[str]:
        """使用LLM生成查询（以摘要为核心）"""
        if not self.use_llm:
            return None

        title = patent.title
        abstract = patent.abstract

        abstract_part = abstract[:500] if abstract else ""
        prompt = f"""请根据以下专利摘要生成一个专业的技术查询问题。摘要为核心依据，标题仅供参考。

专利标题：{title}

专利摘要：{abstract_part}

要求：
1. 主要依据摘要中的技术方案与创新点生成问题
2. 问题应具有检索价值，能用于专利检索
3. 使用专业的技术术语
4. 问题简洁明了，不超过50字
5. 必须是完整的问题句，以问号结尾

请只输出一个查询问题，不要其他内容："""

        query = self._call_llm_api(prompt)

        if query:
            # 验证查询质量
            if self._is_valid_query(query):
                return query
            else:
                logger.debug(f"LLM生成的查询质量不高: {query[:50]}...")

        return None

    def _is_valid_query(self, query: str) -> bool:
        """验证查询质量"""
        if not query:
            return False

        # 检查长度
        if len(query) < 10 or len(query) > 100:
            return False

        # 检查是否包含问号
        if '？' not in query and '?' not in query:
            return False

        # 检查是否过于简单
        simple_queries = [
            "该专利涉及什么技术？",
            "什么是？",
            "如何？",
            "怎样？"
        ]

        if query in simple_queries:
            return False

        return True

    def generate_query(self, patent) -> str:
        """生成查询（LLM + 后备规则）"""
        # 尝试使用LLM
        if self.use_llm and self.api_key:
            llm_query = self.generate_query_with_llm(patent)
            if llm_query:
                return llm_query

        # LLM失败或未启用，使用规则生成
        return super().generate_query(patent)

    def generate_batch(self, patents: List) -> List[Dict]:
        """批量生成查询（带进度条和统计）"""
        logger.info(f"开始为 {len(patents)} 个专利生成查询...")

        samples = []
        llm_success = 0
        rule_fallback = 0

        for patent in tqdm(patents, desc="生成查询", leave=True):
            query = self.generate_query(patent)

            # 记录生成方式
            if self.use_llm and self.api_key:
                # 判断是否是规则生成的
                if "该专利涉及什么技术" in query or "什么是" in query and "？" in query:
                    rule_fallback += 1
                else:
                    llm_success += 1

            # 创建样本
            sample = {
                'query': query,
                'patent_id': patent.id,
                'patent_title': patent.title,
                'patent_abstract': patent.abstract,
                'ipc_subclasses': list(patent.ipc_subclasses),
                'ipc_groups': list(patent.ipc_groups)
            }

            samples.append(sample)

        # 打印统计信息
        if self.use_llm and self.api_key:
            logger.info(f"查询生成统计: LLM成功 {llm_success} 个，规则后备 {rule_fallback} 个")

        logger.info(f"生成了 {len(samples)} 个查询样本")
        return samples


class LocalLLMQueryGenerator(SimpleQueryGenerator):
    """基于本地模型的查询生成器"""

    def __init__(self, config):
        super().__init__(config)
        self.use_llm = self.config.use_llm
        self.local_model_enabled = getattr(self.config, 'local_model_enabled', False)
        self.local_model_name = getattr(self.config, 'local_model_name',
                                        "/home/cly/Embedding-FT/model/Qwen2.5-7B-Instruct")
        self.local_model_device = getattr(self.config, 'local_model_device', "auto")
        self.local_model_dtype = getattr(self.config, 'local_model_dtype', "auto")
        self.max_new_tokens = getattr(self.config, 'max_new_tokens', 100)

        # 延迟加载模型
        self.model = None
        self.tokenizer = None
        self.model_loaded = False

    def _load_local_model(self):
        """加载本地模型（延迟加载）"""
        if self.model_loaded:
            return

        try:
            logger.info(f"开始加载本地模型: {self.local_model_name}")

            # 动态导入，避免不必要的依赖
            try:
                from modelscope import AutoModelForCausalLM, AutoTokenizer
                self._modelscope_available = True
            except ImportError:
                logger.warning("modelscope库未安装，尝试使用transformers")
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self._modelscope_available = False

            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_name)

            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.local_model_name,
                dtype=self.local_model_dtype,
                device_map=self.local_model_device,
                trust_remote_code=True
            )

            self.model_loaded = True
            logger.info(f"本地模型加载完成: {self.local_model_name}")

        except Exception as e:
            logger.error(f"加载本地模型失败: {e}")
            self.model_loaded = False
            raise

    def _call_local_llm(self, prompt: str) -> Optional[str]:
        """调用本地LLM模型生成查询"""
        if not self.local_model_enabled or not self.model_loaded:
            return None

        try:
            # 使用模型的聊天模板
            messages = [
                {"role": "system", "content": "你是一个专利检索专家，擅长从专利摘要生成高质量的技术查询问题。"},
                {"role": "user", "content": prompt}
            ]

            # 应用聊天模板
            if hasattr(self.tokenizer, 'apply_chat_template'):
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # 回退到简单格式
                text = f"系统: {messages[0]['content']}\n\n用户: {messages[1]['content']}\n\n助手:"

            # 生成
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
                )

            # 解码
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # 清理响应
            query = response.strip()

            # 去除可能的额外说明
            if "。" in query:
                query = query.split("。")[0]
            if "：" in query:
                query = query.split("：")[-1]

            return query

        except Exception as e:
            logger.error(f"本地模型生成查询失败: {e}")
            return None

    def generate_query_with_local_llm(self, patent) -> Optional[str]:
        """使用本地LLM生成查询"""
        if not self.use_llm:
            return None

        title = patent.title
        abstract = patent.abstract
        ipc_codes = patent.ipc_codes

        # 提取主要技术领域
        main_ipc = ipc_codes[0] if ipc_codes else ""

        # 从摘要中提取技术问题和技术效果
        technical_problem = self._extract_technical_problem(abstract)
        technical_effect = self._extract_technical_effect(abstract)

        # 精简摘要，提取核心句
        abstract_sentences = abstract.split('。')
        core_abstract = ""
        for sent in abstract_sentences:
            if len(sent) > 20 and ('包括' in sent or '涉及' in sent or '提供' in sent or '解决' in sent):
                core_abstract = sent
                break
        if not core_abstract and abstract_sentences:
            core_abstract = abstract_sentences[0]

        prompt = f"""请为以下专利生成一个**有技术深度的检索查询**。要求查询不能简单询问标题内容，而要从具体的技术问题、技术效果或应用场景出发。

专利标题：{title}
技术领域：{main_ipc}
技术问题：{technical_problem if technical_problem else '未明确说明'}
技术效果：{technical_effect if technical_effect else '未明确说明'}

**生成要求：**
1. 查询必须基于**具体的技术问题**或**技术效果**生成
2. 禁止使用"如何实现[标题]"这样的模板
3. 查询应体现技术的**创新点**或**应用场景**
4. 长度控制在20-50字之间
5. 使用疑问句，但避免过于泛化的问题

**好的示例：**
- "在高温多尘环境下，哪种光伏板的散热结构和防尘设计能保证长期稳定发电？"
- "如何通过多层复合结构提高锂电池在低温下的充放电效率？"
- "针对现有焊接工艺热影响区大的问题，哪种焊接方法能有效控制热输入？"

请生成查询："""

        # 确保模型已加载
        if not self.model_loaded:
            try:
                self._load_local_model()
            except Exception as e:
                logger.error(f"无法加载本地模型: {e}")
                return None

        query = self._call_local_llm(prompt)

        if query:
            # 验证查询质量
            if self._validate_improved_query(query, title):
                return query
            else:
                logger.debug(f"本地LLM生成的查询质量不高: {query[:50]}...")

        return None

    def _extract_technical_problem(self, abstract: str) -> str:
        """从摘要中提取技术问题"""
        if not abstract:
            return ""

        # 寻找问题相关的句子
        problem_indicators = ['解决', '问题', '缺陷', '不足', '缺点', '局限性', '挑战']
        sentences = abstract.split('。')

        for sent in sentences:
            if any(indicator in sent for indicator in problem_indicators):
                return sent.strip()

        return ""

    def _extract_technical_effect(self, abstract: str) -> str:
        """从摘要中提取技术效果"""
        if not abstract:
            return ""

        effect_indicators = ['提高', '降低', '增强', '减少', '改善', '优化', '实现', '达到']
        sentences = abstract.split('。')

        for sent in sentences:
            if any(indicator in sent for indicator in effect_indicators):
                return sent.strip()

        return ""

    def _validate_improved_query(self, query: str, title: str) -> bool:
        """改进的查询质量验证"""
        if not query:
            return False

        # 检查长度
        if len(query) < 15 or len(query) > 100:
            return False

        # 检查是否包含问号
        if '？' not in query and '?' not in query:
            return False

        # 检查是否过于模板化
        template_patterns = [
            f"如何实现{title}",
            f"什么是{title}",
            f"{title}的技术要点",
            f"{title}是什么",
            "该专利涉及什么技术"
        ]

        for pattern in template_patterns:
            if pattern in query:
                return False

        # 检查是否有具体技术细节
        tech_indicators = ['包括', '涉及', '通过', '基于', '采用', '解决', '优化', '提高', '降低']
        if not any(indicator in query for indicator in tech_indicators):
            return False

        return True

    def _is_valid_query(self, query: str) -> bool:
        """验证查询质量（兼容父类方法）"""
        # 这里简化处理，不检查标题
        if not query:
            return False

        # 检查长度
        min_length = getattr(self.config, 'min_query_length', 10)
        max_length = getattr(self.config, 'max_query_length', 100)
        if len(query) < min_length or len(query) > max_length:
            return False

        # 检查是否包含问号
        if '？' not in query and '?' not in query:
            return False

        # 检查是否过于简单
        simple_queries = [
            "该专利涉及什么技术？",
            "什么是？",
            "如何？",
            "怎样？"
        ]

        if query in simple_queries:
            return False

        return True

    def generate_query(self, patent) -> str:
        """生成查询（本地LLM + 后备规则）"""
        if self.local_model_enabled:
            local_query = self.generate_query_with_local_llm(patent)
            if local_query:
                return local_query
        return super().generate_query(patent)


def create_query_generator(config):
    """创建查询生成器。优先本地LLM，其次API（摘要→query），否则规则生成。"""
    if getattr(config.query, 'local_model_enabled', False):
        logger.info("创建本地LLM查询生成器")
        return LocalLLMQueryGenerator(config)
    if config.query.use_llm and config.query.llm_api_key:
        logger.info("创建API查询生成器（LLM）")
        return LLMQueryGenerator(config)
    logger.info("创建规则查询生成器")
    return SimpleQueryGenerator(config)