"""
Train a word-level GPT on Adrian Chiles Guardian article titles.

Adapted from karpathy's microgpt:
https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95

No external dependencies required — pure Python autograd engine.
"""

import argparse
import math
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
args = parser.parse_args()

random.seed(args.seed)

n_layer = args.n_layer
n_embd = args.n_embd
block_size = args.block_size
n_head = 4
head_dim = n_embd // n_head
num_steps = args.steps

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
docs = [line.strip() for line in open("titles.txt") if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

uwords = sorted(set(w for doc in docs for w in doc.split()))
word2idx = {w: i for i, w in enumerate(uwords)}
BOS = len(uwords)   # beginning / end of sequence token
vocab_size = len(uwords) + 1
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
# Training  (from karpathy/microgpt, unchanged)
# ---------------------------------------------------------------------------
lr = 0.01
beta1, beta2, eps_adam = 0.85, 0.99, 1e-8
m_adam = [0.0] * len(params)
v_adam = [0.0] * len(params)

print(f"training {num_steps} steps...")
for step in range(num_steps):
    doc = docs[step % len(docs)]
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

    lr_t = lr * (1 - step / num_steps)
    for i, p in enumerate(params):
        m_adam[i] = beta1 * m_adam[i] + (1 - beta1) * p.grad
        v_adam[i] = beta2 * v_adam[i] + (1 - beta2) * p.grad ** 2
        m_hat = m_adam[i] / (1 - beta1 ** (step + 1))
        v_hat = v_adam[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    if (step + 1) % 50 == 0 or step == 0:
        print(f"step {step+1:4d}/{num_steps} | loss {loss.data:.4f}", flush=True)

print()

# ---------------------------------------------------------------------------
# Save checkpoint
# ---------------------------------------------------------------------------
checkpoint = {
    "state_dict": {k: [[p.data for p in row] for row in mat] for k, mat in state_dict.items()},
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
    pickle.dump(checkpoint, f)
print("saved model.pkl")

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
