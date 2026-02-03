# 专利 RAG Embedding 微调

基于 **BGE-base-zh-v1.5** 的专利检索 embedding 微调：摘要→LLM 生成 query，正样本为专利摘要，负样本为 **同大组不同小组**（技术领域相关、技术不同）。

## 数据格式

- 原始数据：`data/embedding/微调数据-5753/` 下每专利一个 `*.json`。
- 必备字段：`publication_number`、`title`、`abstract`、`classifications`（含 `code`，如 `G06T3/4076`）。

## 流程概要

1. **Query**：由专利 **摘要** 经 LLM（或规则后备）生成检索 query。
2. **正样本**：该专利的 **摘要**。
3. **负样本**：同 IPC **大组**、**不同小组** 的专利摘要；排除与锚点任一 IPC 完全相同的专利。

## 运行

```bash
pip install -r requirements.txt
```

### 推荐命令（经验超参数）

```bash
python src/main.py --data_dir "data/embedding/微调数据-5753" --output_dir ./experiments --recommended
```

`--recommended` 会使用下列经验超参数：`batch_size=32`、`lr=2e-5`、`epochs=5`、`warmup_ratio=0.1`、`loss=triplet`、`margin=0.3`、LoRA `r=16`/`alpha=32`、`checkpoint_every_n_epochs=1`、`early_stopping_patience=3`。

### 其他示例

```bash
# 默认数据目录
python src/main.py --recommended

# 指定数据 / 输出
python src/main.py --data_dir "data/embedding/微调数据-5753" --output_dir ./experiments --recommended

# 快速测试（小数据、少轮数）
python src/main.py --quick_test

# 使用 API LLM 生成 query（需提供 key）
python src/main.py --recommended --use_api_llm --llm_api_key "sk-..."
# 或 export OPENAI_API_KEY=sk-... 后：
python src/main.py --recommended --use_api_llm
```

### 常用参数

- **数据**：`--data_dir`、`--output_dir`、`--max_samples`
- **训练**：`--batch_size`、`--learning_rate`、`--num_epochs`、`--resume`、`--checkpoint_every_n_epochs`
- **Query**：默认**本地 LLM**；`--use_api_llm` 使用 API，`--llm_api_key` 或 `OPENAI_API_KEY`
- **缓存**：`--use_cache`（数据集）、`--no_query_cache`（禁用 query 复用）
- **其他**：`--model_name`、`--negative_strategy`、`--test_only`

## 配置要点

- **base 模型**：`BAAI/bge-base-zh-v1.5`（可在 `config` 或 `--model_name` 修改）。
- **Query**：默认 **本地 LLM**；支持 `--use_api_llm` + API key。LLM 生成的 query 会**缓存**到 `{processed_data_dir}/llm_queries_cache.json`，重复训练时自动复用；`--no_query_cache` 可关闭。
- **Checkpoint**：每 `checkpoint_every_n_epochs` 个 epoch 保存一次（默认 1），存于 `{output_dir}/checkpoints/`。
- **负样本**：`optimized_mixed`，优先同大组不同小组，再同小类不同大组、随机补足。
- **划分**：按 **专利 ID** 划分 train/val/test，避免泄露。

## 单条样例说明

当前 `微调数据-5753` 仅一条样例时，**无法构造同大组负例**（需至少 2 个专利同大组）。扩充同目录下 JSON 数量后即可正常构建数据集与训练。
