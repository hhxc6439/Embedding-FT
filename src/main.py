#!/usr/bin/env python3
"""
专利 RAG embedding 模型微调主程序。默认 bge-base-zh-v1.5，摘要→LLM→query，正样本为摘要，负样本为同大组不同小组。
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import torch

print(f"CUDA_VISIBLE_DEVICES 环境变量: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"PyTorch 可见设备数量: {torch.cuda.device_count()}")
print(f"PyTorch 当前设备: {torch.cuda.current_device()}")
print(f"PyTorch 设备名称: {torch.cuda.get_device_name()}")

_src_dir = Path(__file__).resolve().parent
_project_root = _src_dir.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import Config
from data_loader import PatentDataLoader, create_data_loader
from query_generator import create_query_generator
from negative_sampler import create_negative_sampler
from dataset_builder import DatasetBuilder, PatentDataset
from model import create_model, PatentEmbeddingModel
from trainer import create_trainer


def _get_query_cache_path(config):
    return os.path.join(
        config.data.processed_data_dir,
        getattr(config.query, "query_cache_filename", "llm_queries_cache.json"),
    )


def _load_query_cache(config, patents) -> Optional[List[dict]]:
    """若启用缓存且存在与当前 patent 集合一致的缓存，则返回 queries 列表（与 patents 同序），否则返回 None。"""
    if not getattr(config.query, "use_query_cache", True):
        return None
    path = _get_query_cache_path(config)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    by_id = data.get("queries_by_id") or {}
    current_ids = {p.id for p in patents}
    if set(by_id.keys()) != current_ids:
        return None
    out = []
    for p in patents:
        q = by_id.get(p.id)
        if not q:
            return None
        out.append(q)
    return out


def _save_query_cache(config, patents, queries):
    """将 queries 按 patent_id 索引后写入缓存。"""
    use = getattr(config.query, "use_query_cache", True)
    if not use:
        return
    path = _get_query_cache_path(config)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    by_id = {q["patent_id"]: q for q in queries}
    payload = {"patent_ids": [p.id for p in patents], "queries_by_id": by_id}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


class ProgressTracker:
    """进度跟踪器"""

    def __init__(self, total_steps=8):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.step_times = []

    def start_step(self, step_name: str):
        """开始一个新步骤"""
        self.current_step += 1
        step_start = time.time()
        print(f"\n{'=' * 60}")
        print(f"步骤 {self.current_step}/{self.total_steps}: {step_name}")
        print(f"{'=' * 60}")
        return step_start

    def end_step(self, step_start: float, message: str = ""):
        """结束当前步骤"""
        step_time = time.time() - step_start
        self.step_times.append(step_time)

        if message:
            print(f"✓ {message}")
        print(f"⏱️  步骤耗时: {step_time:.1f}秒")
        print(f"{'=' * 60}")

    def summary(self):
        """打印总结"""
        total_time = time.time() - self.start_time
        print(f"\n{'=' * 60}")
        print("🎉 所有步骤完成!")
        print(f"{'=' * 60}")
        print(f"总耗时: {total_time:.1f}秒 ({total_time / 60:.1f}分钟)")
        print(f"平均每步耗时: {total_time / self.total_steps:.1f}秒")
        print(f"{'=' * 60}")


# 设置日志
def setup_logging(log_dir: str, level: str = "INFO"):
    """设置日志配置"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "training.log"

    # 配置日志格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # 创建日志处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(log_format, date_format))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format, date_format))

    # 配置根日志
    logging.basicConfig(
        level=getattr(logging, level),
        handlers=[file_handler, console_handler]
    )

    # 设置第三方库的日志级别
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    return log_file


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="专利embedding模型微调")

    # 数据路径
    parser.add_argument("--data_dir", type=str,
                        help="专利数据目录（包含json文件）")
    parser.add_argument("--output_dir", type=str,
                        help="输出目录")

    # 模型配置
    parser.add_argument("--model_name", type=str,
                        help="基础模型名称或路径")
    parser.add_argument("--use_lora", action="store_true",
                        help="是否使用LoRA微调")

    # 训练配置
    parser.add_argument("--batch_size", type=int,
                        help="批次大小")
    parser.add_argument("--learning_rate", type=float,
                        help="学习率")
    parser.add_argument("--num_epochs", type=int,
                        help="训练轮数")

    # 负样本策略
    parser.add_argument("--negative_strategy", type=str,
                        choices=["optimized_mixed", "same_group_priority",
                                 "same_subclass_different_group", "mixed", "random", "hard"],
                        help="负样本采样策略")
    parser.add_argument("--negatives_per_positive", type=int,
                        help="每个正样本的负样本数")

    # 查询生成（默认本地 LLM；--use_api_llm 时改用 API，key 可用 --llm_api_key 或 env OPENAI_API_KEY）
    parser.add_argument("--use_llm", action="store_true",
                        help="使用 LLM 生成查询")
    parser.add_argument("--use_api_llm", action="store_true",
                        help="使用 API 型 LLM 生成查询（覆盖本地 LLM）")
    parser.add_argument("--llm_api_key", type=str,
                        help="LLM API 密钥（不指定时可从环境变量 OPENAI_API_KEY 读取）")

    # 其他
    parser.add_argument("--max_samples", type=int,
                        help="最大样本数（用于测试）")
    parser.add_argument("--use_cache", action="store_true", default=True,
                        help="使用数据集缓存")
    parser.add_argument("--no_query_cache", action="store_true",
                        help="禁用 LLM query 缓存（每次重新生成）")
    parser.add_argument("--test_only", action="store_true",
                        help="仅测试，不训练")
    parser.add_argument("--resume", type=str,
                        help="从检查点恢复训练")
    parser.add_argument("--checkpoint_every_n_epochs", type=int,
                        help="每 N 个 epoch 保存一次检查点（默认 1）")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="日志级别")
    parser.add_argument("--quick_test", action="store_true",
                        help="快速测试模式（小数据集）")
    parser.add_argument("--recommended", action="store_true",
                        help="使用经验推荐训练超参数（见 README）")

    return parser.parse_args()


def update_config_from_args(config: Config, args: argparse.Namespace):
    """根据命令行参数更新配置"""
    if args.data_dir:
        data_dir = os.path.normpath(os.path.abspath(args.data_dir))
        config.data.raw_data_dir = data_dir
        config.data.processed_data_dir = os.path.join(data_dir, "processed")

    if args.output_dir:
        out_dir = os.path.normpath(os.path.abspath(args.output_dir))
        config.experiment.output_dir = out_dir
        config.experiment.log_dir = os.path.join(out_dir, "logs")

    if args.model_name:
        config.model.base_model_name = args.model_name

    if args.use_lora:
        config.model.use_lora = args.use_lora

    if args.batch_size:
        config.training.batch_size = args.batch_size

    if args.learning_rate:
        config.training.learning_rate = args.learning_rate

    if args.num_epochs:
        config.training.num_epochs = args.num_epochs

    if args.negative_strategy:
        config.negative.strategy = args.negative_strategy

    if args.negatives_per_positive:
        config.negative.negatives_per_positive = args.negatives_per_positive

    if args.use_llm:
        config.query.use_llm = args.use_llm
    if getattr(args, "use_api_llm", False):
        config.query.local_model_enabled = False
        config.query.use_llm = True
    key = getattr(args, "llm_api_key", None) or os.environ.get("OPENAI_API_KEY")
    if key:
        config.query.llm_api_key = key

    if args.max_samples:
        config.data.max_samples = args.max_samples
    if getattr(args, "no_query_cache", False):
        config.query.use_query_cache = False
    if getattr(args, "checkpoint_every_n_epochs", None) is not None:
        config.training.checkpoint_every_n_epochs = args.checkpoint_every_n_epochs

    # 经验推荐超参数
    if getattr(args, "recommended", False):
        print("📐 使用经验推荐训练超参数")
        config.training.batch_size = 32
        config.training.learning_rate = 2e-5
        config.training.num_epochs = 5
        config.training.warmup_ratio = 0.1
        config.training.weight_decay = 0.01
        config.training.loss_type = "triplet"
        config.training.margin = 0.3
        config.training.temperature = 0.05
        config.training.early_stopping_patience = 3
        config.training.eval_steps = 100
        config.training.save_steps = 500
        config.training.checkpoint_every_n_epochs = 1
        config.model.use_lora = True
        config.model.lora_r = 16
        config.model.lora_alpha = 32
        config.model.lora_dropout = 0.1

    # 快速测试模式
    if args.quick_test:
        config.data.max_samples = 100
        config.training.num_epochs = 2
        config.training.batch_size = 4
        print("🔧 启用快速测试模式")

    return config


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 创建新的配置实例并更新
    config = Config()
    config = update_config_from_args(config, args)

    # 设置日志
    log_file = setup_logging(config.experiment.log_dir, args.log_level)
    logger = logging.getLogger(__name__)

    # 创建进度跟踪器
    tracker = ProgressTracker(total_steps=7 if args.test_only else 8)

    logger.info("=" * 60)
    logger.info("🚀 专利embedding模型微调")
    logger.info("=" * 60)

    # 打印配置信息
    logger.info("📋 配置信息:")
    logger.info(f"  数据目录: {config.data.raw_data_dir}")
    logger.info(f"  输出目录: {config.experiment.output_dir}")
    logger.info(f"  模型名称: {config.model.base_model_name}")
    logger.info(f"  LoRA微调: {config.model.use_lora}")
    logger.info(f"  批次大小: {config.training.batch_size}")
    logger.info(f"  学习率: {config.training.learning_rate}")
    logger.info(f"  训练轮数: {config.training.num_epochs}")
    logger.info(f"  负样本策略: {config.negative.strategy}")
    logger.info(f"  负样本数: {config.negative.negatives_per_positive}")
    logger.info(f"  LLM生成查询: {config.query.use_llm}（本地: {getattr(config.query, 'local_model_enabled', False)}）")
    logger.info(f"  Query 缓存: {'开' if getattr(config.query, 'use_query_cache', True) else '关'}")
    logger.info(f"  每 N epoch 保存 checkpoint: {getattr(config.training, 'checkpoint_every_n_epochs', 1)}")
    logger.info(f"  日志文件: {log_file}")
    logger.info("=" * 60)

    try:
        # 1. 加载专利数据
        step_start = tracker.start_step("加载专利数据")
        logger.info("开始加载专利数据...")

        data_loader = create_data_loader(config, load_existing=args.use_cache)

        # 检查是否有缓存的数据
        if not data_loader.patents or not args.use_cache:
            patents = data_loader.load_from_folder(
                config.data.raw_data_dir,
                max_samples=config.data.max_samples
            )
            data_loader.process_patents(patents)
        else:
            logger.info("使用已加载的缓存数据")

        logger.info(f"✅ 加载了 {len(data_loader.patents)} 个专利")
        tracker.end_step(step_start, f"加载了 {len(data_loader.patents)} 个专利")

        # 2. 生成查询（优先复用已保存的 LLM query 缓存）
        step_start = tracker.start_step("生成查询")
        logger.info("开始生成查询...")

        query_generator = create_query_generator(config)
        cached = _load_query_cache(config, data_loader.patents)
        if cached is not None:
            queries = cached
            logger.info(f"📂 复用已保存的 query 缓存（{len(queries)} 条）")
        else:
            use_llm = getattr(config.query, "local_model_enabled", False) or (
                config.query.use_llm and bool(config.query.llm_api_key)
            )
            if getattr(config.query, "local_model_enabled", False):
                logger.info("🤖 使用本地 LLM 生成查询")
            elif config.query.use_llm and config.query.llm_api_key:
                logger.info("🤖 使用 LLM API 生成查询")
            else:
                logger.info("📝 使用规则生成查询")
            queries = query_generator.generate_batch(data_loader.patents)
            if use_llm:
                _save_query_cache(config, data_loader.patents, queries)
            logger.info(f"✅ 生成了 {len(queries)} 个查询" + ("并已写入缓存" if use_llm else ""))
        tracker.end_step(step_start, f"共 {len(queries)} 个查询")

        # 3. 创建负样本采样器
        step_start = tracker.start_step("创建负样本采样器")
        logger.info("开始创建负样本采样器...")

        negative_sampler = create_negative_sampler(data_loader, config)

        logger.info(f"✅ 负样本采样器创建完成，策略: {config.negative.strategy}")
        tracker.end_step(step_start, f"负样本采样器创建完成")

        # 4. 构建数据集
        step_start = tracker.start_step("构建训练数据集")
        logger.info("开始构建数据集...")

        dataset_builder = DatasetBuilder(
            config, query_generator, negative_sampler, data_loader
        )

        # 传递已生成的查询，避免重复生成
        dataset = dataset_builder.build_dataset(
            use_cache=args.use_cache,
            pre_generated_queries=queries  # 传递步骤2生成的查询
        )

        # 保存数据集
        dataset_files = dataset_builder.save_dataset(
            os.path.join(config.experiment.output_dir, "datasets")
        )

        logger.info(f"✅ 数据集构建完成")
        tracker.end_step(step_start, "数据集构建完成")

        # 5. 创建PyTorch数据集
        step_start = tracker.start_step("创建PyTorch数据集")
        logger.info("开始创建PyTorch数据集...")

        # 创建模型（用于获取tokenizer）
        model = create_model(config)
        tokenizer = model.embedding_model.tokenizer

        # 创建数据集（BGE 推荐：query 加 instruction，passage 不加）
        query_instruction = getattr(config.model, 'query_instruction_for_retrieval', None) or ""
        train_dataset = PatentDataset(
            dataset['train'],
            tokenizer,
            max_length=config.model.max_seq_length,
            query_instruction=query_instruction
        )

        val_dataset = PatentDataset(
            dataset['val'],
            tokenizer,
            max_length=config.model.max_seq_length,
            query_instruction=query_instruction
        )

        test_dataset = PatentDataset(
            dataset['test'],
            tokenizer,
            max_length=config.model.max_seq_length,
            query_instruction=query_instruction
        ) if dataset.get('test') else None

        logger.info(f"✅ 数据集创建完成")
        logger.info(f"  训练集: {len(train_dataset)} 个样本")
        logger.info(f"  验证集: {len(val_dataset)} 个样本")
        if test_dataset:
            logger.info(f"  测试集: {len(test_dataset)} 个样本")

        tracker.end_step(step_start, f"创建了 {len(train_dataset)} 个训练样本")

        # 6. 训练模型
        if not args.test_only:
            step_start = tracker.start_step("训练模型")
            logger.info("开始训练模型...")

            # 创建训练器
            trainer = create_trainer(
                config, model, train_dataset, val_dataset, test_dataset
            )

            # 恢复训练（如果指定）
            if args.resume:
                logger.info(f"从检查点恢复训练: {args.resume}")
                trainer.load_checkpoint(args.resume)

            # 开始训练
            logger.info("开始训练过程...")
            training_results = trainer.train()

            logger.info("✅ 训练完成!")
            logger.info(f"  最佳模型: {training_results['best_model_path']}")
            logger.info(f"  最终模型: {training_results['final_model_path']}")
            logger.info(f"  最佳验证损失: {training_results['best_val_loss']:.4f}")

            tracker.end_step(step_start, "模型训练完成")
        else:
            logger.info("⏭️  跳过训练步骤（仅测试模式）")

        # 7. 测试模型
        step_start = tracker.start_step("测试模型")
        logger.info("开始测试模型...")

        if args.test_only:
            # 仅测试模式：加载最佳模型
            best_model_path = os.path.join(config.experiment.output_dir, "best_model")
            if os.path.exists(best_model_path):
                model = PatentEmbeddingModel.from_pretrained(best_model_path, config)
                logger.info(f"🔍 从 {best_model_path} 加载模型进行测试")
            else:
                logger.warning("⚠️  未找到最佳模型，使用新创建的模型")
                model = create_model(config)

        # 创建训练器进行测试
        trainer = create_trainer(config, model, train_dataset, val_dataset, test_dataset)

        # 测试
        test_results = trainer.test()

        if test_results:
            logger.info("✅ 测试完成!")
            logger.info(f"  测试准确率: {test_results['accuracy']:.4f}")
            logger.info(f"  测试相似度差距: {test_results['sim_gap']:.4f}")
        else:
            logger.warning("⚠️  没有测试结果")

        tracker.end_step(step_start, "模型测试完成")

        # 8. 保存最终配置
        step_start = tracker.start_step("保存配置和结果")
        config_path = os.path.join(config.experiment.output_dir, "final_config.json")
        config.save(config_path)
        logger.info(f"✅ 最终配置已保存到: {config_path}")

        # 保存运行摘要
        summary_path = os.path.join(config.experiment.output_dir, "run_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("专利Embedding模型微调 - 运行摘要\n")
            f.write("=" * 60 + "\n")
            f.write(f"运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据目录: {config.data.raw_data_dir}\n")
            f.write(f"专利数量: {len(data_loader.patents)}\n")
            f.write(f"训练样本: {len(train_dataset)}\n")
            f.write(f"负样本策略: {config.negative.strategy}\n")
            f.write(f"查询生成方式: {'LLM' if config.query.use_llm else '规则'}\n")
            if test_results:
                f.write(f"测试准确率: {test_results['accuracy']:.4f}\n")
                f.write(f"相似度差距: {test_results['sim_gap']:.4f}\n")
            f.write("=" * 60 + "\n")

        logger.info(f"✅ 运行摘要已保存到: {summary_path}")
        tracker.end_step(step_start, "配置和结果保存完成")

        # 打印总耗时
        tracker.summary()

        logger.info("=" * 60)
        logger.info("🎉 所有步骤完成!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}", exc_info=True)
        sys.exit(1)


def test_quick():
    """快速测试函数"""
    print("🚀 启动快速测试模式...")

    test_args = [
        sys.argv[0],
        "--quick_test",
        "--output_dir", os.path.join(str(_project_root), "test_output"),
        "--negative_strategy", "optimized_mixed",
        "--log_level", "INFO"
    ]
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        test_args.extend(["--data_dir", sys.argv[1]])

    sys.argv = test_args

    print("🔧 测试配置: 最大样本数=100, 训练轮数=2, 批次大小=4, 负样本策略=optimized_mixed")
    main()


if __name__ == "__main__":
    # 检查是否是小数据集测试模式
    if len(sys.argv) > 1 and sys.argv[1] == "--quick_test":
        # 移除参数，避免重复解析
        sys.argv.remove("--quick_test")
        test_quick()
    elif len(sys.argv) == 1:
        # 没有命令行参数，询问用户
        print("请选择运行模式:")
        print("1. 完整训练")
        print("2. 快速测试（小数据集）")
        print("3. 仅测试模式")

        choice = input("请输入选择 (1, 2或3): ").strip()

        if choice == "2":
            test_quick()
        elif choice == "3":
            sys.argv = [sys.argv[0], "--test_only"]
            main()
        else:
            main()
    else:
        # 有命令行参数，直接运行主函数
        main()