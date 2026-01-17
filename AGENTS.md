# AGENTS.md — Guidelines for AI Assistants

This document provides instructions for AI agents working on this repository. The goal is to maintain consistency in teaching style and ensure content is accessible to the target audience.

---

## Project Overview

**LLM Inference the Hard Way** is an educational project that teaches how Large Language Model inference works by implementing it from scratch — no high-level ML libraries, just raw math and tensors.

### Target Model
- **GPT-2 (124M parameters)** — Small enough to run on CPU, architecturally identical to modern LLMs

### Project Structure
```
llm-inference-hard-way/
├── models/gpt2/           # Downloaded model weights
├── step1_*.ipynb          # Download & explore model
├── step2_*.ipynb          # BPE tokenization
├── step3_*.ipynb          # Token + position embeddings
├── step4_*.ipynb          # Attention mechanism
├── step5_*.ipynb          # Transformer block
├── step6_*.ipynb          # Forward pass
├── step7_*.ipynb          # Autoregressive generation
├── step8_*.ipynb          # KV caching
├── main.py                # Final unified implementation
└── pyproject.toml         # Dependencies (managed by uv)
```

---

## 🎯 Target Audience

### Who We're Writing For

**Software Engineers with infrastructure/backend background who are NEW to Deep Learning.**

They are:
- Experienced with code, debugging, and systems thinking
- Familiar with: databases, caching, load balancers, distributed systems, APIs
- Comfortable with: Python, data structures, algorithms
- **NOT** familiar with: ML jargon, calculus, linear algebra terminology, PyTorch/TensorFlow

### What They DON'T Know
- What a "tensor" is beyond "multi-dimensional array"
- Why neural networks need "training"
- What gradients, backpropagation, or loss functions are
- ML-specific terms: embedding, attention, softmax, logits, etc.
- Academic notation (∂, ∇, Σ, etc.)

### What They DO Know
- Key-value stores (Redis, Memcached)
- Database queries and indexing
- Load balancing and traffic distribution
- Caching strategies
- API request/response patterns
- Hash maps, arrays, matrices as data structures
- Normalization in the context of data processing

---

## ✍️ Writing Style Guidelines

### 1. Always Use Infrastructure Analogies

Map ML concepts to systems they already understand:

| ML Concept | Infrastructure Analogy |
|------------|------------------------|
| Attention Q, K, V | Database query: Query searches Keys to retrieve Values |
| Softmax | Weighted load balancing (traffic percentages that sum to 100%) |
| Embedding lookup | Hash table lookup / dictionary access |
| KV Cache | Request memoization / Redis cache for repeated computations |
| Token | Chunk of data with an ID (like a cache key) |
| Forward pass | Request flowing through a pipeline of middleware |
| Residual connection | Preserving original request while adding enrichments |
| Layer normalization | Normalizing metrics before aggregation |
| Multi-head attention | Parallel workers each handling part of the workload |
| Batch processing | Processing multiple requests in parallel |

### 2. Explain Before You Show Code

Never drop code without context. Follow this pattern:

```markdown
## The Problem We're Solving

[1-2 sentences on WHY this matters]

## How It Works (Conceptual)

[Explain the idea using an analogy they know]

## The Implementation

[Now show the code with inline comments]
```

### 3. Avoid or Define ML Jargon

❌ **Bad:**
> "We compute the logits by projecting the hidden states through the LM head."

✅ **Good:**
> "We convert the internal representation to vocabulary scores (called 'logits') — one score per possible next word. Higher score = model thinks that word is more likely."

When introducing a term:
1. Give the intuitive explanation first
2. Then introduce the technical term
3. Use the term consistently afterward

### 4. Use Concrete Examples

❌ **Bad:**
> "The attention mechanism computes pairwise similarities."

✅ **Good:**
> "Each word asks: 'Who in this sentence is relevant to me?' The word 'it' looks at all other words and decides 'server' (0.85 attention) is what I refer to, not 'crashed' (0.05)."

### 5. Show Dimensions Explicitly

Always show tensor shapes — this grounds abstract concepts:

```python
# Token embeddings: look up each token ID in the embedding table
# Shape: [seq_len] → [seq_len, 768]
token_emb = wte[token_ids]  # (5,) → (5, 768)
```

### 6. Use Visual ASCII Diagrams

```
Input: "The server crashed"
         ↓
┌─────────────────────────────────────┐
│  Tokenizer                          │
│  "The server crashed" → [464, 4382, 14293] │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│  Embedding Lookup                   │
│  [464, 4382, 14293] → (3, 768) matrix │
└─────────────────────────────────────┘
         ↓
       ...
```

### 7. Address "Why?" Before "How?"

Engineers want to know the reasoning, not just the formula:

❌ **Bad:**
> "We scale by √d_k as shown in the formula."

✅ **Good:**
> "We scale by √d_k because dot products naturally grow larger with more dimensions. Without scaling, softmax 'saturates' — outputs become nearly 0 or 1, and gradients vanish. Think of it like normalizing metrics before comparing them."

---

## 📓 Notebook Structure Template

Each `step*.ipynb` should follow this structure:

```markdown
# Step N: [Topic] [Emoji]

[2-3 sentence hook explaining why this matters]

## What We'll Learn

1. **[Concept 1]** - [One-line description]
2. **[Concept 2]** - [One-line description]
3. ...

---

## [Section 1: The Problem / Motivation]

[Why do we need this? What problem does it solve?]
[Use an analogy from infrastructure/systems]

---

## [Section 2: Conceptual Explanation]

[Explain how it works at a high level]
[Include ASCII diagrams]
[Use concrete examples]

---

## [Section 3: Implementation]

[Code cells with extensive comments]
[Show shapes at each step]
[Print intermediate results for intuition]

---

## [Section 4: Putting It Together]

[Combine into a reusable function/class]
[Test with real examples]

---

## [Section 5: Visualizations / Experiments]

[Interactive exploration]
[Plots, attention heatmaps, etc.]

---

## Summary: What We Learned

- [Key takeaway 1]
- [Key takeaway 2]
- [Connection to next step]

---

## Next Steps

[Preview of the next notebook]
```

---

## 🔧 Code Style Guidelines

### 1. Prefer NumPy Over PyTorch for Teaching

NumPy is more familiar to general software engineers. Use it for core implementations.

```python
# ✅ Prefer this
import numpy as np
output = np.matmul(Q, K.T) / np.sqrt(d_k)

# ❌ Avoid this (for teaching)
import torch
output = torch.matmul(Q, K.T) / torch.sqrt(d_k)
```

### 2. Explicit Over Clever

Write readable code, not clever code:

```python
# ✅ Good: Clear and explicit
def softmax(x, axis=-1):
    # Subtract max for numerical stability
    x_shifted = x - x.max(axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)

# ❌ Bad: Too compact
def softmax(x, axis=-1):
    return np.exp(x - x.max(axis, keepdims=True)) / np.exp(x - x.max(axis, keepdims=True)).sum(axis, keepdims=True)
```

### 3. Always Print Shapes

```python
print(f"Q shape: {Q.shape}")      # Q shape: (5, 768)
print(f"K shape: {K.shape}")      # K shape: (5, 768)
print(f"scores shape: {scores.shape}")  # scores shape: (5, 5)
```

### 4. Use Descriptive Variable Names

```python
# ✅ Good
attention_weights = softmax(scaled_scores)
context_vector = attention_weights @ values

# ❌ Bad
a = softmax(s)
c = a @ v
```

### 5. Add Type Hints for Function Signatures

```python
def attention(
    Q: np.ndarray,  # (seq_len, d_k)
    K: np.ndarray,  # (seq_len, d_k)
    V: np.ndarray,  # (seq_len, d_v)
    mask: np.ndarray = None  # (seq_len, seq_len)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scaled dot-product attention.
    
    Returns:
        output: (seq_len, d_v)
        weights: (seq_len, seq_len)
    """
```

---

## 🚫 Anti-Patterns to Avoid

### 1. Don't Assume ML Background
- ❌ "As you know from backpropagation..."
- ❌ "This is just a standard cross-entropy loss"
- ❌ "The gradient flows through..."

### 2. Don't Use Academic Notation Without Explanation
- ❌ "∇L with respect to θ"
- ✅ "The gradient (direction of steepest increase) of the loss"

### 3. Don't Skip Steps
- ❌ "Obviously, this gives us the attention weights"
- ✅ "Softmax converts raw scores to probabilities that sum to 1"

### 4. Don't Use "Simple" or "Just"
- ❌ "You simply multiply the matrices"
- ❌ "It's just a linear projection"
- ✅ "We multiply the matrices, which computes..."

### 5. Don't Ignore Edge Cases Engineers Care About
- What happens with empty input?
- What about numerical stability?
- Why this specific architecture choice?

---

## 📋 Checklist for New Content

Before considering a notebook complete, verify:

- [ ] **Motivation**: Does it explain WHY before HOW?
- [ ] **Analogies**: Does it use infrastructure/SWE analogies?
- [ ] **Jargon**: Are all ML terms defined when first used?
- [ ] **Shapes**: Are tensor dimensions shown at each step?
- [ ] **Examples**: Are there concrete, worked examples?
- [ ] **Diagrams**: Are there ASCII or visual diagrams?
- [ ] **Code Comments**: Is the code well-commented?
- [ ] **Outputs**: Do code cells show meaningful print output?
- [ ] **Connection**: Does it connect to previous/next steps?
- [ ] **Summary**: Is there a clear summary of key takeaways?

---

## 📚 Reference: Key Concepts to Cover

### Core Transformer Concepts
1. **Tokenization (BPE)** — How text becomes numbers
2. **Embeddings** — How token IDs become dense vectors
3. **Positional Encoding** — How the model knows word order
4. **Self-Attention** — How tokens look at each other
5. **Multi-Head Attention** — Parallel attention patterns
6. **Causal Masking** — Why GPT can't see the future
7. **Layer Normalization** — Keeping values stable
8. **Feed-Forward Network (MLP)** — Per-token processing
9. **Residual Connections** — Preserving information flow
10. **Softmax & Logits** — Converting to probabilities
11. **Autoregressive Generation** — Producing text token by token
12. **KV Caching** — Efficient inference optimization

### GPT-2 Specific Details
- Pre-norm architecture (LayerNorm before attention/MLP)
- Learned positional embeddings (not RoPE)
- Weight tying (wte used for both input and output)
- GELU activation in MLP
- 12 layers, 12 heads, 768 embedding dim

---

## 💡 Example: Good Explanation Pattern

Here's an example of explaining softmax to our audience:

---

### ❌ Bad Explanation (Too Academic)

> Softmax is defined as σ(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ). It maps ℝⁿ → (0,1)ⁿ such that the outputs form a valid probability distribution.

### ✅ Good Explanation (Infrastructure-Friendly)

> **Softmax = Weighted Load Balancing**
>
> Imagine you have 4 servers and need to distribute traffic. Raw scores might be:
> ```
> Server scores: [10, 5, 3, 2]
> ```
>
> Softmax converts these to percentages that sum to 100%:
> ```
> Traffic split: [73%, 17%, 7%, 3%]  ← sums to 100%
> ```
>
> The server with the highest score gets the most traffic, but others still get some.
>
> In attention, softmax answers: "What percentage of attention should this word pay to each other word?"
>
> ```python
> def softmax(scores):
>     # exp() makes all values positive and amplifies differences
>     exp_scores = np.exp(scores - scores.max())  # subtract max for stability
>     # Divide by sum to get percentages
>     return exp_scores / exp_scores.sum()
> ```

---

## 🔄 Continuous Improvement

When working on this repo:

1. **Read existing notebooks first** — Maintain consistency with established style
2. **Test with the target audience in mind** — Would a backend engineer understand this?
3. **Iterate on analogies** — If an analogy doesn't land, try another
4. **Prefer concrete over abstract** — Numbers and examples beat formulas
5. **Keep it interactive** — Jupyter notebooks should be run, not just read
