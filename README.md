# LLM Inference the Hard Way

Learn how Large Language Model inference *actually* works by implementing it from scratch — no high-level libraries, just raw math and tensors.

## What's This About?

Instead of calling `model.generate()` and treating the model as a black box, we'll build every piece ourselves:

- **Parse raw model weights** directly from SafeTensors files
- **Implement tokenization** from scratch (BPE algorithm)
- **Build the Transformer** layer by layer
- **Understand Q, K, V matrices** and how attention really works
- **Generate text** token by token (autoregressive decoding)
- **Optimize with KV caching** for efficient inference

We use GPT-2 (124M parameters) as our learning model — small enough to run on a CPU, yet architecturally identical to modern LLMs like GPT-4.

---

## 🗺️ Learning Plan

Each step is a **Jupyter notebook** with explanations, visualizations, and runnable code.

### Step 1: Download & Explore the Model ✅
> **Notebook:** `step1_download_model.ipynb`

- Download GPT-2 weights from HuggingFace
- Parse SafeTensors format manually (no library!)
- Understand tensors, token embeddings, and position embeddings
- Learn about attention heads and the full architecture

**Key insight:** The `c_attn.weight [768, 2304]` tensor produces Q, K, V together (2304 = 768 × 3)

---

### Step 2: Build the Tokenizer 🔤
> **Notebook:** `step2_tokenizer.ipynb`

- Implement Byte Pair Encoding (BPE) from scratch
- Load `vocab.json` and `merges.txt`
- Encode: text → token IDs
- Decode: token IDs → text

**Key concepts:**
- BPE starts with bytes, merges common pairs
- `"Hello"` → `[15496]` (single token)
- `"tokenization"` → `["token", "ization"]` → `[30001, 1634]`

---

### Step 3: Embeddings & Positional Encoding 📍
> **Notebook:** `step3_embeddings.ipynb`

- Token embeddings: `wte.weight [50257, 768]`
- Position embeddings: `wpe.weight [1024, 768]`
- Combine: `hidden = token_emb + position_emb`

**Key insight:** Unlike modern models (RoPE), GPT-2 uses learned positional embeddings

---

### Step 4: Attention Mechanism Deep Dive 🎯
> **Notebook:** `step4_attention.ipynb`

The heart of the transformer! We'll implement:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

- **Q (Query):** "What am I looking for?"
- **K (Key):** "What do I contain?"
- **V (Value):** "What information do I provide?"

**Key concepts:**
- Multi-head attention: 12 heads × 64 dims = 768
- Causal masking: Can't attend to future tokens
- Attention patterns visualization

---

### Step 5: The Full Transformer Block 🧱
> **Notebook:** `step5_transformer_block.ipynb`

Put it together:
```
x = x + Attention(LayerNorm(x))  # Residual + attention
x = x + MLP(LayerNorm(x))        # Residual + feedforward
```

Components:
- LayerNorm (pre-norm architecture)
- Multi-head self-attention
- MLP (expand 4×, GELU, project back)
- Residual connections

---

### Step 6: Forward Pass & Logits 📊
> **Notebook:** `step6_forward_pass.ipynb`

- Stack 12 transformer blocks
- Final LayerNorm
- Project to vocabulary: `logits = hidden @ wte.weight.T`
- Understand weight tying (same matrix for input/output)

---

### Step 7: Autoregressive Generation 🔄
> **Notebook:** `step7_generation.ipynb`

Generate text token by token:
```python
for _ in range(max_tokens):
    logits = forward(tokens)
    next_token = sample(logits[-1])
    tokens.append(next_token)
```

Sampling strategies:
- Greedy (argmax)
- Temperature scaling
- Top-k sampling
- Top-p (nucleus) sampling

---

### Step 8: KV Caching ⚡
> **Notebook:** `step8_kv_cache.ipynb`

Make inference fast!

**Problem:** Recomputing K, V for all previous tokens is wasteful

**Solution:** Cache K, V from previous steps
```python
# Without cache: O(n²) per token
# With cache: O(n) per token
```

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/yourusername/llm-inference-hard-way.git
cd llm-inference-hard-way

# Install dependencies with uv
uv sync

# Launch Jupyter Lab
uv run jupyter lab

# Then open step1_download_model.ipynb and run the cells!
```

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management

### Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Tensor operations |
| `numpy` | Array manipulation |
| `requests` | Download model weights |
| `regex` | BPE tokenizer |
| `tqdm` | Progress bars |
| `jupyterlab` | Interactive notebooks |
| `ipykernel` | Jupyter Python kernel |

## GPT-2 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GPT-2 (124M) Architecture                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: "Hello world"                                       │
│           ↓                                                 │
│  ┌─────────────────┐                                        │
│  │   Tokenizer     │  "Hello world" → [15496, 995]         │
│  └────────┬────────┘                                        │
│           ↓                                                 │
│  ┌─────────────────┐                                        │
│  │ Token Embedding │  [50257, 768] lookup                   │
│  │   + Position    │  [1024, 768] lookup                    │
│  └────────┬────────┘                                        │
│           ↓                                                 │
│  ┌─────────────────┐                                        │
│  │ Transformer ×12 │  Each block:                           │
│  │                 │  ├─ LayerNorm                          │
│  │                 │  ├─ Multi-Head Attention (12 heads)    │
│  │                 │  ├─ Residual Connection                │
│  │                 │  ├─ LayerNorm                          │
│  │                 │  ├─ MLP (768→3072→768)                 │
│  │                 │  └─ Residual Connection                │
│  └────────┬────────┘                                        │
│           ↓                                                 │
│  ┌─────────────────┐                                        │
│  │ Final LayerNorm │                                        │
│  └────────┬────────┘                                        │
│           ↓                                                 │
│  ┌─────────────────┐                                        │
│  │   LM Head       │  hidden @ wte.T → [50257] logits      │
│  └────────┬────────┘                                        │
│           ↓                                                 │
│  Output: probability distribution over 50,257 tokens        │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Hyperparameters:
  • n_vocab  = 50,257  (vocabulary size)
  • n_ctx    = 1,024   (max sequence length)
  • n_embd   = 768     (embedding dimension)
  • n_head   = 12      (attention heads)
  • n_layer  = 12      (transformer blocks)
  • d_head   = 64      (768 / 12, dimension per head)
```

## Project Structure

```
llm-inference-hard-way/
├── models/
│   └── gpt2/                         # Downloaded model weights
│       ├── config.json
│       ├── model.safetensors         # 548 MB of weights
│       ├── vocab.json                # Token → ID mapping
│       └── merges.txt                # BPE merge rules
├── step1_download_model.ipynb        # ✅ Download & explore
├── step2_tokenizer.ipynb             # 🔜 BPE tokenization
├── step3_embeddings.ipynb            # 🔜 Token + position embeddings
├── step4_attention.ipynb             # 🔜 Q, K, V and attention
├── step5_transformer_block.ipynb     # 🔜 Full transformer block
├── step6_forward_pass.ipynb          # 🔜 Complete forward pass
├── step7_generation.ipynb            # 🔜 Autoregressive decoding
├── step8_kv_cache.ipynb              # 🔜 KV caching optimization
├── pyproject.toml
└── README.md
```

## Why "The Hard Way"?

Most tutorials use `transformers.AutoModel` or similar abstractions. While convenient, this hides the fascinating details:

- How does the model convert "Hello" into numbers?
- What exactly are Q, K, V in attention?
- Why does the model have 160 separate weight tensors?
- How does sampling work?
- Why is KV caching so important for inference speed?

By building from scratch, you'll truly understand what happens between input text and generated output.

## License

MIT
