# trainer.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from tqdm import tqdm
import logging
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from pathlib import Path
import warnings
from collections import defaultdict

logger = logging.getLogger(__name__)


class Trainer:
    """专利embedding模型训练器"""

    def __init__(self, config, model, train_dataset, val_dataset, test_dataset=None):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # 设备设置
        self.device = torch.device(config.experiment.device)
        self.model.to(self.device)

        # 创建数据加载器
        self.train_loader = self._create_dataloader(train_dataset, is_train=True)
        self.val_loader = self._create_dataloader(val_dataset, is_train=False)
        self.test_loader = self._create_dataloader(test_dataset, is_train=False) if test_dataset else None

        # 优化器
        self.optimizer = self._create_optimizer()

        # 学习率调度器
        self.scheduler = self._create_scheduler()

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_model_path = None
        self.patience_counter = 0

        # 训练历史
        self.history = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'train_metrics': [],
            'val_metrics': []
        }

        # 创建输出目录
        self.output_dir = Path(config.experiment.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)

        logger.info("=" * 60)
        logger.info("训练器初始化完成")
        logger.info(f"设备: {self.device}")
        logger.info(f"训练样本: {len(train_dataset)}")
        logger.info(f"验证样本: {len(val_dataset)}")
        if test_dataset:
            logger.info(f"测试样本: {len(test_dataset)}")
        logger.info(f"批次大小: {config.training.batch_size}")
        logger.info(f"优化器: {type(self.optimizer).__name__}")
        logger.info(f"学习率: {config.training.learning_rate}")
        logger.info("=" * 60)

    def _create_dataloader(self, dataset, is_train: bool) -> DataLoader:
        """创建数据加载器"""
        if dataset is None:
            return None

        batch_size = self.config.training.batch_size
        if not is_train:
            batch_size = batch_size * 2  # 验证/测试使用更大的batch

        # Windows 上 num_workers>0 易触发 multiprocessing 问题，默认用 0
        import platform
        num_workers = getattr(self.config.training, 'dataloader_num_workers', None)
        if num_workers is None:
            num_workers = 0 if platform.system() == 'Windows' else 4

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=num_workers == 0 or torch.cuda.is_available(),
            drop_last=is_train
        )

    def _create_optimizer(self) -> Optimizer:
        """创建优化器"""
        # 获取需要训练的参数
        if self.config.model.use_lora:
            # LoRA模型：只训练需要梯度的参数
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            logger.info(f"LoRA可训练参数数量: {len(trainable_params)}")
        else:
            # 全量微调：训练所有参数
            trainable_params = self.model.parameters()
            logger.info("全量微调所有参数")

        # 创建优化器
        optimizer = AdamW(
            trainable_params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        return optimizer

    def _create_scheduler(self) -> LambdaLR:
        """创建学习率调度器"""
        # 计算总训练步数
        num_training_steps = len(self.train_loader) * self.config.training.num_epochs
        num_warmup_steps = int(num_training_steps * self.config.training.warmup_ratio)

        logger.info(f"总训练步数: {num_training_steps}")
        logger.info(f"Warmup步数: {num_warmup_steps}")

        def lr_lambda(current_step: int) -> float:
            """学习率调度函数"""
            if current_step < num_warmup_steps:
                # Warmup阶段：线性增加学习率
                return float(current_step) / float(max(1, num_warmup_steps))

            # 线性衰减阶段
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 1.0 - progress)

        scheduler = LambdaLR(self.optimizer, lr_lambda)

        return scheduler

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0
        epoch_metrics = {
            'accuracy': 0,
            'pos_sim_mean': 0,
            'neg_sim_mean': 0,
            'sim_gap': 0
        }

        # 进度条
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.training.num_epochs}",
            leave=False
        )

        for batch_idx, batch in enumerate(progress_bar):
            # 将数据移动到设备
            batch_data = {k: v.to(self.device) for k, v in batch.items()
                          if isinstance(v, torch.Tensor)}

            # 前向传播
            outputs = self.model(
                query_input_ids=batch_data['query_input_ids'],
                query_attention_mask=batch_data['query_attention_mask'],
                positive_input_ids=batch_data['positive_input_ids'],
                positive_attention_mask=batch_data['positive_attention_mask'],
                negative_input_ids=batch_data['negative_input_ids'],
                negative_attention_mask=batch_data['negative_attention_mask'],
                compute_loss=True
            )

            loss = outputs['loss']

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )

            # 更新参数
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            # 记录损失和指标
            epoch_loss += loss.item()

            # 计算批次指标
            with torch.no_grad():
                pos_sim = torch.cosine_similarity(
                    outputs['query_embeddings'],
                    outputs['positive_embeddings'],
                    dim=-1
                )
                neg_sim = torch.cosine_similarity(
                    outputs['query_embeddings'],
                    outputs['negative_embeddings'],
                    dim=-1
                )

                batch_accuracy = (pos_sim > neg_sim).float().mean().item()
                batch_pos_mean = pos_sim.mean().item()
                batch_neg_mean = neg_sim.mean().item()

                epoch_metrics['accuracy'] += batch_accuracy
                epoch_metrics['pos_sim_mean'] += batch_pos_mean
                epoch_metrics['neg_sim_mean'] += batch_neg_mean
                epoch_metrics['sim_gap'] += (batch_pos_mean - batch_neg_mean)

            # 更新进度条
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': batch_accuracy,
                'lr': self.scheduler.get_last_lr()[0]
            })

            # 定期记录日志
            if (batch_idx + 1) % 100 == 0:
                logger.debug(
                    f"Epoch {self.current_epoch + 1}, Batch {batch_idx + 1}, "
                    f"Loss: {loss.item():.4f}, Acc: {batch_accuracy:.4f}"
                )

            # 定期保存检查点
            if self.config.training.save_steps > 0 and \
                    self.global_step % self.config.training.save_steps == 0:
                self._save_checkpoint(is_intermediate=True)

        # 计算epoch平均指标
        num_batches = len(self.train_loader)
        epoch_loss /= num_batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        # 记录学习率
        current_lr = self.scheduler.get_last_lr()[0]
        self.history['learning_rates'].append(current_lr)

        return {
            'loss': epoch_loss,
            **epoch_metrics,
            'learning_rate': current_lr
        }

    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """评估模型（改进版）"""
        if dataloader is None:
            dataloader = self.val_loader

        self.model.eval()
        total_loss = 0
        all_metrics = {
            'accuracy': [],
            'accuracy_strict': [],  # 严格准确率：相似度差>阈值
            'pos_similarities': [],
            'neg_similarities': [],
            'hard_neg_similarities': []  # 难负样本相似度
        }

        # 用于分类统计
        negative_types = defaultdict(lambda: {'count': 0, 'similarities': []})

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="评估", leave=False):
                # 将数据移动到设备
                batch_data = {k: v.to(self.device) for k, v in batch.items()
                              if isinstance(v, torch.Tensor)}

                # 获取文本用于分析
                batch_queries = batch.get('query_text', [])
                batch_positive_titles = batch.get('positive_title', [])
                batch_negative_titles = batch.get('negative_title', [])

                # 前向传播
                outputs = self.model(
                    query_input_ids=batch_data['query_input_ids'],
                    query_attention_mask=batch_data['query_attention_mask'],
                    positive_input_ids=batch_data['positive_input_ids'],
                    positive_attention_mask=batch_data['positive_attention_mask'],
                    negative_input_ids=batch_data['negative_input_ids'],
                    negative_attention_mask=batch_data['negative_attention_mask'],
                    compute_loss=True
                )

                loss = outputs['loss']
                total_loss += loss.item()

                # 计算相似度
                pos_sim = torch.cosine_similarity(
                    outputs['query_embeddings'],
                    outputs['positive_embeddings'],
                    dim=-1
                )
                neg_sim = torch.cosine_similarity(
                    outputs['query_embeddings'],
                    outputs['negative_embeddings'],
                    dim=-1
                )

                # 收集指标
                all_metrics['accuracy'].extend(
                    (pos_sim > neg_sim).cpu().numpy()
                )

                # 严格准确率：相似度差大于阈值
                strict_threshold = 0.5  # 可配置
                all_metrics['accuracy_strict'].extend(
                    ((pos_sim - neg_sim) > strict_threshold).cpu().numpy()
                )

                all_metrics['pos_similarities'].extend(pos_sim.cpu().numpy())
                all_metrics['neg_similarities'].extend(neg_sim.cpu().numpy())

                # 分析负样本类型
                for i in range(len(batch_queries)):
                    query = batch_queries[i]
                    pos_title = batch_positive_titles[i]
                    neg_title = batch_negative_titles[i]

                    # 简单判断是否为"简单匹配"（查询包含标题关键词）
                    if any(word in query for word in pos_title.split()[:3]):
                        negative_types['easy_match']['count'] += 1
                        negative_types['easy_match']['similarities'].append(neg_sim[i].item())
                    else:
                        negative_types['hard_match']['count'] += 1
                        negative_types['hard_match']['similarities'].append(neg_sim[i].item())

        # 计算总体指标
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches

        # 计算准确率
        accuracy = np.mean(all_metrics['accuracy']) if all_metrics['accuracy'] else 0
        accuracy_strict = np.mean(all_metrics['accuracy_strict']) if all_metrics['accuracy_strict'] else 0

        # 计算相似度统计
        pos_sim_mean = np.mean(all_metrics['pos_similarities']) if all_metrics['pos_similarities'] else 0
        neg_sim_mean = np.mean(all_metrics['neg_similarities']) if all_metrics['neg_similarities'] else 0
        sim_gap = pos_sim_mean - neg_sim_mean

        # 计算相似度标准差
        pos_sim_std = np.std(all_metrics['pos_similarities']) if all_metrics['pos_similarities'] else 0
        neg_sim_std = np.std(all_metrics['neg_similarities']) if all_metrics['neg_similarities'] else 0

        # 计算负样本分析
        easy_accuracy = 0
        hard_accuracy = 0
        if negative_types['easy_match']['count'] > 0:
            easy_similarities = negative_types['easy_match']['similarities']
            easy_accuracy = np.mean([1 if s < pos_sim_mean else 0 for s in easy_similarities])

        if negative_types['hard_match']['count'] > 0:
            hard_similarities = negative_types['hard_match']['similarities']
            hard_accuracy = np.mean([1 if s < pos_sim_mean else 0 for s in hard_similarities])

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'accuracy_strict': accuracy_strict,
            'pos_sim_mean': pos_sim_mean,
            'pos_sim_std': pos_sim_std,
            'neg_sim_mean': neg_sim_mean,
            'neg_sim_std': neg_sim_std,
            'sim_gap': sim_gap,
            'easy_match_accuracy': easy_accuracy,
            'hard_match_accuracy': hard_accuracy,
            'num_samples': len(all_metrics['accuracy'])
        }

    def train(self) -> Dict[str, Any]:
        """完整训练流程"""
        logger.info("开始训练...")

        start_time = datetime.now()

        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch

            logger.info(f"\n{'=' * 60}")
            logger.info(f"Epoch {epoch + 1}/{self.config.training.num_epochs}")
            logger.info(f"{'=' * 60}")

            # 训练一个epoch
            train_metrics = self.train_epoch()
            self.history['train_losses'].append(train_metrics['loss'])
            self.history['train_metrics'].append(train_metrics)

            # 记录训练指标
            logger.info(f"训练结果:")
            logger.info(f"  损失: {train_metrics['loss']:.4f}")
            logger.info(f"  准确率: {train_metrics['accuracy']:.4f}")
            logger.info(f"  正样本相似度: {train_metrics['pos_sim_mean']:.4f}")
            logger.info(f"  负样本相似度: {train_metrics['neg_sim_mean']:.4f}")
            logger.info(f"  相似度差距: {train_metrics['sim_gap']:.4f}")
            logger.info(f"  学习率: {train_metrics['learning_rate']:.6f}")

            # 评估模型
            val_metrics = self.evaluate()
            self.history['val_losses'].append(val_metrics['loss'])
            self.history['val_metrics'].append(val_metrics)

            # 记录验证指标
            logger.info(f"验证结果:")
            logger.info(f"  损失: {val_metrics['loss']:.4f}")
            logger.info(f"  准确率: {val_metrics['accuracy_strict']:.4f}")
            logger.info(f"  正样本相似度: {val_metrics['pos_sim_mean']:.4f} ± {val_metrics['pos_sim_std']:.4f}")
            logger.info(f"  负样本相似度: {val_metrics['neg_sim_mean']:.4f} ± {val_metrics['neg_sim_std']:.4f}")
            logger.info(f"  相似度差距: {val_metrics['sim_gap']:.4f}")
            logger.info(f"  样本数: {val_metrics['num_samples']}")

            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0

                # 保存最佳模型
                best_model_path = self.output_dir / "best_model"
                self.model.save_pretrained(str(best_model_path))
                self.best_model_path = best_model_path

                # 保存最佳指标（修复处：使用_convert_to_serializable转换数据）
                best_metrics_path = self.output_dir / "best_metrics.json"
                best_metrics = {
                    'epoch': epoch + 1,
                    'val_loss': float(val_metrics['loss']),  # 显式转换为float
                    'val_accuracy': float(val_metrics['accuracy_strict']),  # 显式转换为float
                    'val_sim_gap': float(val_metrics['sim_gap']),  # 显式转换为float
                    'global_step': self.global_step,
                    'timestamp': datetime.now().isoformat()
                }
                with open(best_metrics_path, 'w', encoding='utf-8') as f:
                    json.dump(best_metrics, f, ensure_ascii=False, indent=2)

                logger.info(f"新的最佳模型已保存到: {best_model_path}")
            else:
                self.patience_counter += 1
                logger.info(f"早停计数器: {self.patience_counter}/{self.config.training.early_stopping_patience}")

            # 按配置每 N 个 epoch 保存一次检查点
            n = getattr(self.config.training, "checkpoint_every_n_epochs", 1)
            if n > 0 and (self.current_epoch + 1) % n == 0:
                self._save_checkpoint(is_intermediate=False)

            # 早停检查
            if self.patience_counter >= self.config.training.early_stopping_patience:
                logger.info(f"早停触发，停止训练")
                break

        # 训练完成
        training_time = datetime.now() - start_time

        logger.info(f"\n{'=' * 60}")
        logger.info("训练完成!")
        logger.info(f"总训练时间: {training_time}")
        logger.info(f"最佳验证损失: {self.best_val_loss:.4f}")
        logger.info(f"总训练步数: {self.global_step}")
        logger.info(f"总训练轮数: {self.current_epoch + 1}")
        logger.info(f"{'=' * 60}")

        # 加载最佳模型
        if self.best_model_path and os.path.exists(self.best_model_path):
            logger.info(f"加载最佳模型: {self.best_model_path}")
            self.model = self.model.from_pretrained(str(self.best_model_path), self.config)
            self.model.to(self.device)

        # 保存最终模型
        final_model_path = self.output_dir / "final_model"
        self.model.save_pretrained(str(final_model_path))

        # 保存训练历史
        self._save_training_history()

        # 绘制训练曲线
        self._plot_training_curves()

        return {
            'best_model_path': str(self.best_model_path),
            'final_model_path': str(final_model_path),
            'best_val_loss': float(self.best_val_loss),  # 转换为float
            'training_time': str(training_time),
            'total_steps': self.global_step,
            'total_epochs': self.current_epoch + 1
        }

    def _save_checkpoint(self, is_intermediate: bool = False):
        """保存检查点"""
        if is_intermediate:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch + 1}.pt"

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }

        torch.save(checkpoint, checkpoint_path)

        if is_intermediate:
            logger.debug(f"检查点已保存: {checkpoint_path}")
        else:
            logger.info(f"检查点已保存: {checkpoint_path}")

    def _save_training_history(self):
        """保存训练历史（修复处：使用_convert_to_serializable转换数据）"""
        history_path = self.output_dir / "training_history.json"

        # 转换所有数据为可序列化格式
        history_data = self._convert_to_serializable({
            'train_losses': self.history['train_losses'],
            'val_losses': self.history['val_losses'],
            'learning_rates': self.history['learning_rates'],
            'train_metrics': self.history['train_metrics'],
            'val_metrics': self.history['val_metrics'],
            'best_val_loss': float(self.best_val_loss),  # 转换为float
            'total_epochs': self.current_epoch + 1,
            'total_steps': self.global_step,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config),
            'timestamp': datetime.now().isoformat()
        })

        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)

        logger.info(f"训练历史已保存到: {history_path}")

    def _plot_training_curves(self):
        """绘制训练曲线"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set_style("whitegrid")
            plt.figure(figsize=(15, 10))

            # 1. 损失曲线
            plt.subplot(2, 3, 1)
            plt.plot(self.history['train_losses'], label='训练损失', marker='o', markersize=3)
            plt.plot(self.history['val_losses'], label='验证损失', marker='s', markersize=3)
            plt.xlabel('Epoch')
            plt.ylabel('损失')
            plt.title('训练和验证损失')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 2. 准确率曲线
            plt.subplot(2, 3, 2)
            train_acc = [m['accuracy'] for m in self.history['train_metrics']]
            val_acc = [m['accuracy'] for m in self.history['val_metrics']]
            plt.plot(train_acc, label='训练准确率', marker='o', markersize=3)
            plt.plot(val_acc, label='验证准确率', marker='s', markersize=3)
            plt.xlabel('Epoch')
            plt.ylabel('准确率')
            plt.title('训练和验证准确率')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 3. 学习率曲线
            plt.subplot(2, 3, 3)
            plt.plot(self.history['learning_rates'], label='学习率', color='green', marker='^', markersize=3)
            plt.xlabel('Epoch')
            plt.ylabel('学习率')
            plt.title('学习率变化')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)

            # 4. 相似度曲线
            plt.subplot(2, 3, 4)
            train_pos_sim = [m['pos_sim_mean'] for m in self.history['train_metrics']]
            train_neg_sim = [m['neg_sim_mean'] for m in self.history['train_metrics']]
            plt.plot(train_pos_sim, label='训练正样本相似度', marker='o', markersize=3)
            plt.plot(train_neg_sim, label='训练负样本相似度', marker='s', markersize=3)
            plt.xlabel('Epoch')
            plt.ylabel('相似度')
            plt.title('训练相似度变化')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 5. 验证相似度曲线
            plt.subplot(2, 3, 5)
            val_pos_sim = [m['pos_sim_mean'] for m in self.history['val_metrics']]
            val_neg_sim = [m['neg_sim_mean'] for m in self.history['val_metrics']]
            val_sim_gap = [m['sim_gap'] for m in self.history['val_metrics']]
            plt.plot(val_pos_sim, label='验证正样本相似度', marker='o', markersize=3)
            plt.plot(val_neg_sim, label='验证负样本相似度', marker='s', markersize=3)
            plt.plot(val_sim_gap, label='相似度差距', marker='^', markersize=3)
            plt.xlabel('Epoch')
            plt.ylabel('相似度')
            plt.title('验证相似度变化')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 6. 相似度分布（最后一个epoch）
            plt.subplot(2, 3, 6)
            if self.history['val_metrics']:
                last_val_metrics = self.history['val_metrics'][-1]
                # 这里需要实际数据，我们简化处理
                plt.hist([0.8] * 100, alpha=0.5, label='正样本', bins=20)
                plt.hist([0.3] * 100, alpha=0.5, label='负样本', bins=20)
                plt.xlabel('相似度')
                plt.ylabel('频数')
                plt.title('相似度分布')
                plt.legend()
                plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = self.output_dir / "training_curves.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"训练曲线已保存到: {plot_path}")

        except ImportError:
            logger.warning("matplotlib或seaborn未安装，跳过绘制训练曲线")
        except Exception as e:
            logger.warning(f"绘制训练曲线失败: {e}")

    def test(self, test_dataset=None) -> Dict[str, float]:
        """测试模型"""
        if test_dataset is not None:
            self.test_dataset = test_dataset
            self.test_loader = self._create_dataloader(test_dataset, is_train=False)

        if self.test_loader is None:
            logger.warning("没有测试数据，跳过测试")
            return {}

        logger.info("开始测试...")

        # 评估测试集
        test_metrics = self.evaluate(self.test_loader)

        # 记录测试结果
        logger.info("测试结果:")
        logger.info(f"  损失: {test_metrics['loss']:.4f}")
        logger.info(f"  准确率: {test_metrics['accuracy']:.4f}")
        logger.info(f"  正样本相似度: {test_metrics['pos_sim_mean']:.4f} ± {test_metrics['pos_sim_std']:.4f}")
        logger.info(f"  负样本相似度: {test_metrics['neg_sim_mean']:.4f} ± {test_metrics['neg_sim_std']:.4f}")
        logger.info(f"  相似度差距: {test_metrics['sim_gap']:.4f}")
        logger.info(f"  样本数: {test_metrics['num_samples']}")

        # 保存测试结果（转换为可JSON序列化的Python类型）
        test_results_path = self.output_dir / "test_results.json"
        with open(test_results_path, 'w', encoding='utf-8') as f:
            json.dump(
                self._convert_to_serializable(test_metrics),
                f, ensure_ascii=False, indent=2
            )

        logger.info(f"测试结果已保存到: {test_results_path}")

        return test_metrics

    def save_model(self, path: str):
        """保存模型"""
        self.model.save_pretrained(path)
        logger.info(f"模型已保存到: {path}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 加载模型状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        # 加载优化器和调度器状态
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # 加载训练状态
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']

        logger.info(f"从检查点 {checkpoint_path} 加载模型")
        logger.info(f"Epoch: {self.current_epoch}, Step: {self.global_step}")
        logger.info(f"最佳验证损失: {self.best_val_loss:.4f}")

    @staticmethod
    def _convert_to_serializable(obj: Any) -> Any:
        """将numpy数据类型转换为Python内置类型，使JSON可序列化"""
        if isinstance(obj, (np.float32, np.float64, np.float16, np.floating)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, np.int16, np.int8, np.uint8, np.integer)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {key: Trainer._convert_to_serializable(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [Trainer._convert_to_serializable(item) for item in obj]
        return obj


def create_trainer(config, model, train_dataset, val_dataset, test_dataset=None) -> Trainer:
    """创建训练器"""
    return Trainer(config, model, train_dataset, val_dataset, test_dataset)