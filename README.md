# Qwen3 Model Server

一个基于 FastAPI 的模型服务，加载 Qwen3 后常驻 GPU，通过 HTTP 端口提供生成和可解释性分析功能，避免重复加载模型的开销。

## 架构

```
┌────────────────────────────────────┐     HTTP (8000)     ┌──────────────────────┐
│  qwen3_server.py                   │ ◄───────────────── │  qwen3_client.py      │
│                                    │                     │                       │
│  GPU:  Qwen3-8B (常驻显存)         │                     │  CLI 工具             │
│  ├── /generate     (流式/非流式)    │                     │  ├── generate         │
│  ├── /hidden_states (残差流提取)    │                     │  ├── stream           │
│  ├── /logits_lens  (中间层解码)    │                     │  ├── hidden           │
│  ├── /sae          (稀疏特征分析)   │                     │  ├── logits           │
│  ├── /sae_intervene(特征干预生成)   │                     │  ├── sae              │
│  └── /health                       │                     │  ├── sae_set_dir      │
└────────────────────────────────────┘                     │  ├── intervene        │
                                                           │  └── health           │
                                                           └──────────────────────┘
```

## 快速开始

### 1. 安装依赖

```bash
pip install fastapi uvicorn pydantic requests
```

### 2. 启动服务器

```bash
# 启动服务（模型尚未加载）
python3 qwen3_server.py &

# 加载模型
python3 qwen3_client.py load /home/shadowmydx/.cache/modelscope/hub/models/Qwen/Qwen3-8B
```

服务器加载模型后常驻进程，后续所有请求共享同一个模型实例。

### 3. 文本生成

```bash
# 非流式生成
python3 qwen3_client.py generate "how to make a cake" -m 512 -t 0.7

# 流式生成（逐 token 输出）
python3 qwen3_client.py stream "tell me a story"
```

### 4. 可解释性分析

```bash
# 提取指定层的残差流 hidden state
python3 qwen3_client.py hidden "The capital of France is" -l 0,10,20,31

# Logits Lens — 看各中间层"认为"下一个词是什么
python3 qwen3_client.py logits "The capital of France is" -l 0,8,16,24,28,31 -k 10
```

### 5. SAE (稀疏自编码器) 分析

先下载 SAE 权重到本地（Qwen 官方提供），然后：

```bash
# 指定 SAE 目录
python3 qwen3_client.py sae_set_dir /path/to/SAE-Res-Qwen3-8B-Base-W64K-L0_100

# 运行 SAE 分析
python3 qwen3_client.py sae "The capital of France is" -l 0,15,31 -n 10 -r
```

SAE 会将每层的 4096 维残差流投影到 65536 维稀疏空间（TopK=100），返回激活最强的特征 ID 和激活值，用于可解释性研究。

### 6. SAE 特征干预

先通过 SAE 分析找到一个活跃的特征 ID，然后在生成过程中干预它：

```bash
# 1. 发现特征：看 layer 15 哪些特征在"France is"之后激活
python3 qwen3_client.py sae "The capital of France is" -l 15 -n 5

# 2. 干预生成：把 #45231 特征置零，观察输出变化
python3 qwn3_client.py intervene "The capital of France is" -l 15 -f 45231 -a zero

# 3. 流式干预（逐 token 看变化）
python3 qwen3_client.py intervene "The capital of France is" -l 15 -f 45231 -a zero -s
```

支持的干预类型：

| 类型 | 说明 |
|---|---|
| `zero` | 将该特征激活值置为 0 |
| `scale` | 按比例缩放激活值，例如 `-v 2.0` 放大两倍 |
| `set` | 设为指定值，例如 `-v 10.0` 强制高激活 |
| `clamp_max` | 限制最大激活值，例如 `-v 3.0` 裁剪峰值 |

干预机制通过 PyTorch `register_forward_hook` 实现，在每次前向传播时拦截残差流 → 经 SAE 解码 → 修改指定特征 → 重建残差流 → 继续前向传播。生成完成后 hook 自动移除，不影响后续请求。

## 完整测试

```bash
# 基础功能测试（生成、流式、hidden state、logits lens）
python3 test_qwen3.py /home/shadowmydx/.cache/modelscope/hub/models/Qwen/Qwen3-8B

# 含 SAE 测试
SAE_DIR=/path/to/SAE-Res-Qwen3-8B-Base-W64K-L0_100 \
  python3 test_qwen3.py /home/shadowmydx/.cache/modelscope/hub/models/Qwen/Qwen3-8B
```

## API 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/load` | 加载模型到 GPU |
| GET | `/health` | 健康检查 |
| POST | `/generate` | 文本生成（支持 SSE 流式） |
| POST | `/hidden_states` | 提取指定层残差流 hidden state |
| POST | `/logits_lens` | 中间层投影到 LM head 看 token 概率 |
| POST | `/sae_set_dir` | 设定 SAE 权重目录 |
| GET | `/sae_loaded_layers` | 查看已缓存的 SAE 层 |
| POST | `/sae` | 运行 SAE 提取稀疏特征 |
| POST | `/sae_intervene` | 带 SAE 特征干预的文本生成 |

## Client 命令参考

```
python3 qwen3_client.py <command> [args]

Commands:
  load <model_dir>          加载模型
  generate <prompt>         非流式生成
  stream <prompt>           SSE 流式生成
  hidden <prompt> -l LAYERS 提取 hidden state
  logits <prompt> -l LAYERS Logits Lens
  sae_set_dir <sae_dir>     设定 SAE 目录
  sae <prompt> -l LAYERS    SAE 特征分析
  intervene <prompt> -l LAYER -f FEATURE  SAE 特征干预
  health                    检查服务器状态

通用生成参数 (generate / stream / intervene):
  -m, --max-tokens     最大生成长度 (默认 1024)
  -t, --temperature    采样温度 (默认 0.6)
  --top-p               top-p 采样 (默认 0.95)
  --top-k               top-k 采样 (默认 20)

intervene 特有参数:
  -l, --layer          干预的目标层 (必填)
  -f, --feature-id     SAE 特征 ID (必填)
  -a, --action          干预类型: zero/scale/set/clamp_max (默认 zero)
  -v, --value           干预值 (scale/set/clamp_max 时有效)
  -s, --stream          启用 SSE 流式输出
```

## SAE 规格 (Qwen-Scope Qwen3-8B)

| 属性 | 值 |
|---|---|
| SAE 宽度 (`d_sae`) | 65536 |
| 隐层维度 (`d_model`) | 4096 |
| 扩展比 | 16× |
| 激活 Top-K | 100 |
| 挂载点 | Residual stream |
| 层范围 | 0–35 (36 层) |
| 文件格式 | PyTorch `.pt` dict |

每个 `.pt` 文件包含四个张量：`W_enc` (65536×4096), `b_enc` (65536), `W_dec` (4096×65536), `b_dec` (4096)。

## 可解释性研究方向

该服务的设计支持以下研究方向：

- **特征发现** — 在不同 prompt 上激活 SAE，聚类发现语义特征
- **特征消融** — 将某特征置零后观察输出变化，验证因果作用
- **Logits Lens + SAE 联合分析** — 某特征激活时，logits 分布如何偏移
- **逐层演化** — 分析同一概念在浅层→深层如何被逐步构建
- **SAE 特征调控** — 在生成过程中干预特征激活，实现 steerable generation
