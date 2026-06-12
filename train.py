"""
Train a word-level GPT on Adrian Chiles Guardian article titles.

Adapted from karpathy's microgpt:
https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95

No external dependencies required — pure Python autograd engine.
"""

import argparse
import math
import os
import pickle
import random
import sys

sys.setrecursionlimit(10000)

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--n-layer", type=int, default=2)
parser.add_argument("--n-embd", type=int, default=32)
parser.add_argument("--block-size", type=int, default=20)
parser.add_argument("--n-generate", type=int, default=40)
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint-interval", type=int, default=25)
args = parser.parse_args()

random.seed(args.seed)

checkpoint_interval = args.checkpoint_interval

# ---------------------------------------------------------------------------
# Load checkpoint if available
# ---------------------------------------------------------------------------
checkpoint_data = None
start_step = 0
if os.path.exists("model.pkl"):
    with open("model.pkl", "rb") as f:
        checkpoint_data = pickle.load(f)
    if "step" not in checkpoint_data:
        # Old checkpoint format without optimizer state — can't resume
        checkpoint_data = None

if checkpoint_data:
    uwords = checkpoint_data["uwords"]
    cfg = checkpoint_data["config"]
    n_layer = cfg["n_layer"]
    n_embd = cfg["n_embd"]
    block_size = cfg["block_size"]
    n_head = cfg["n_head"]
    head_dim = cfg["head_dim"]
    vocab_size = checkpoint_data["vocab_size"]
    start_step = checkpoint_data["step"]
    print(f"Resuming from step {start_step}")
else:
    n_layer = args.n_layer
    n_embd = args.n_embd
    block_size = args.block_size
    n_head = 4
    head_dim = n_embd // n_head

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
all_docs = [line.strip() for line in open("titles.txt") if line.strip()]

if checkpoint_data is None:
    uwords = sorted(set(w for doc in all_docs for w in doc.split()))
    vocab_size = len(uwords) + 1

word2idx = {w: i for i, w in enumerate(uwords)}
BOS = len(uwords)

# When resuming, new titles may contain words not in the saved vocabulary — skip them.
docs = [d for d in all_docs if all(w in word2idx for w in d.split())]
random.shuffle(docs)
print(f"num docs: {len(docs)}")
print(f"vocab size: {vocab_size} words")

# ---------------------------------------------------------------------------
# Autograd engine  (from karpathy/microgpt, unchanged)
# ---------------------------------------------------------------------------
class Value:
    __slots__ = ("data", "grad", "_children", "_local_grads")

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data ** other, (self,), (other * self.data ** (other - 1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self):
        ex = math.exp(self.data)
        return Value(ex, (self,), (ex,))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):            return self * -1
    def __radd__(self, other):    return self + other
    def __sub__(self, other):     return self + (-other)
    def __rsub__(self, other):    return other + (-self)
    def __rmul__(self, other):    return self * other
    def __truediv__(self, other): return self * other ** -1
    def __rtruediv__(self, other): return other * self ** -1

    def backward(self):
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad

# ---------------------------------------------------------------------------
# Model  (from karpathy/microgpt, unchanged)
# ---------------------------------------------------------------------------
matrix = lambda nout, nin, std=0.08: [
    [Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)
]

state_dict = {
    "wte": matrix(vocab_size, n_embd),
    "wpe": matrix(block_size, n_embd),
    "lm_head": matrix(vocab_size, n_embd),
}
for i in range(n_layer):
    state_dict[f"layer{i}.attn_wq"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wk"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wv"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.attn_wo"] = matrix(n_embd, n_embd)
    state_dict[f"layer{i}.mlp_fc1"] = matrix(4 * n_embd, n_embd)
    state_dict[f"layer{i}.mlp_fc2"] = matrix(n_embd, 4 * n_embd)

params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")

# Restore weights from checkpoint
if checkpoint_data:
    for k, mat in state_dict.items():
        for i, row in enumerate(mat):
            for j, p in enumerate(row):
                p.data = checkpoint_data["state_dict"][k][i][j]


def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    m = max(v.data for v in logits)
    exps = [(v - m).exp() for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict["wte"][token_id]
    pos_emb = state_dict["wpe"][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        x_res = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f"layer{li}.attn_wq"])
        k = linear(x, state_dict[f"layer{li}.attn_wk"])
        v = linear(x, state_dict[f"layer{li}.attn_wv"])
        keys[li].append(k)
        values[li].append(v)

        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs + head_dim]
            k_h = [ki[hs:hs + head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs + head_dim] for vi in values[li]]
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim ** 0.5
                for t in range(len(k_h))
            ]
            attn_weights = softmax(attn_logits)
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]
            x_attn.extend(head_out)

        x = linear(x_attn, state_dict[f"layer{li}.attn_wo"])
        x = [a + b for a, b in zip(x, x_res)]

        x_res = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f"layer{li}.mlp_fc1"])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f"layer{li}.mlp_fc2"])
        x = [a + b for a, b in zip(x, x_res)]

    return linear(x, state_dict["lm_head"])

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
lr = 0.01
beta1, beta2, eps_adam = 0.85, 0.99, 1e-8

if checkpoint_data:
    m_adam = checkpoint_data["m_adam"]
    v_adam = checkpoint_data["v_adam"]
else:
    m_adam = [0.0] * len(params)
    v_adam = [0.0] * len(params)

num_steps = args.steps
print(f"training {num_steps} steps from step {start_step}...")


def save_checkpoint(step):
    ckpt = {
        "step": step,
        "state_dict": {k: [[p.data for p in row] for row in mat] for k, mat in state_dict.items()},
        "m_adam": m_adam,
        "v_adam": v_adam,
        "uwords": uwords,
        "vocab_size": vocab_size,
        "config": {
            "n_layer": n_layer,
            "n_embd": n_embd,
            "block_size": block_size,
            "n_head": n_head,
            "head_dim": head_dim,
        },
    }
    with open("model.pkl", "wb") as f:
        pickle.dump(ckpt, f)
    print(f"  [checkpoint saved at step {step}]", flush=True)


for local_step in range(num_steps):
    global_step = start_step + local_step
    doc = docs[global_step % len(docs)]
    tokens = [BOS] + [word2idx[w] for w in doc.split()] + [BOS]
    n = min(block_size, len(tokens) - 1)

    keys_buf = [[] for _ in range(n_layer)]
    vals_buf = [[] for _ in range(n_layer)]
    losses = []

    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys_buf, vals_buf)
        probs = softmax(logits)
        losses.append(-probs[target_id].log())

    loss = sum(losses) * (1 / n)
    loss.backward()

    lr_t = lr * (1 - local_step / num_steps)
    for i, p in enumerate(params):
        m_adam[i] = beta1 * m_adam[i] + (1 - beta1) * p.grad
        v_adam[i] = beta2 * v_adam[i] + (1 - beta2) * p.grad ** 2
        # Use global_step for Adam bias correction so momentum carries across runs
        m_hat = m_adam[i] / (1 - beta1 ** (global_step + 1))
        v_hat = v_adam[i] / (1 - beta2 ** (global_step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    if (local_step + 1) % 50 == 0 or local_step == 0:
        print(f"step {global_step+1:4d} | loss {loss.data:.4f}", flush=True)

    if (local_step + 1) % checkpoint_interval == 0:
        save_checkpoint(global_step + 1)

print()

# Save final checkpoint if the last block didn't land on the interval boundary
if num_steps % checkpoint_interval != 0:
    save_checkpoint(start_step + num_steps)

# ---------------------------------------------------------------------------
# Generate titles
# ---------------------------------------------------------------------------
n_generate = args.n_generate
temperature = args.temperature
print(f"generating {n_generate} titles (temperature={temperature})...")

generated = []
attempts = 0
while len(generated) < n_generate and attempts < n_generate * 6:
    attempts += 1
    keys_buf = [[] for _ in range(n_layer)]
    vals_buf = [[] for _ in range(n_layer)]
    token_id = BOS
    words_out = []

    for pos_id in range(block_size - 1):
        logits = gpt(token_id, pos_id, keys_buf, vals_buf)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        words_out.append(uwords[token_id])

    title = " ".join(words_out).strip()
    if len(title.split()) >= 3 and title not in generated:
        generated.append(title)

with open("generated.txt", "w", encoding="utf-8") as f:
    f.write("# Generated Adrian Chiles-style Guardian titles\n")
    f.write(f"# microgpt word-level: {n_layer}L / {n_embd}d / {block_size} ctx / {num_steps} steps\n\n")
    for title in generated:
        f.write(title + "\n")

print(f"saved {len(generated)} titles to generated.txt")
print("\nSample:")
for t in generated[:5]:
    print(f"  {t}")
