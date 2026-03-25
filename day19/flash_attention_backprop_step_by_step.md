# FlashAttention Backpropagation, Step by Step

This note is for learning. It keeps the math simple and uses conservative Markdown formatting.

If you later upload the FlashAttention-4 paper, we can rewrite this note to match the paper exactly.

## 0. The minimum calculus you need

You do not need to be a calculus expert to understand FlashAttention backward.

You only need these ideas.

### 0.1 A derivative means "if I change this a little, what happens?"

If $y = 3x$, then changing $x$ a little changes $y$ by 3 times that amount.

So the derivative is:

$$
\frac{dy}{dx} = 3
$$

For backprop, this is the mindset:

- output changes a little
- which earlier variable caused that change
- how strongly did it affect the result

### 0.2 The chain rule

If a value is computed in steps, like $x \to z \to y$, then the effect of $x$ on $y$ passes through $z$.

If

$$
z = 2x
$$

and

$$
y = z^2
$$

then

$$
\frac{dy}{dx} = \frac{dy}{dz}\frac{dz}{dx}
$$

Backprop is mostly repeated use of the chain rule.

### 0.3 Derivative of a weighted sum

If

$$
o = p_1 v_1 + p_2 v_2 + \cdots
$$

then:

- changing $v_j$ affects only its own term
- changing $p_j$ changes how much of $v_j$ is mixed into the output

This is why attention backward naturally splits into gradient to $V$ and gradient to $P$.

### 0.4 Dot product as similarity

If

$$
s_j = q^T k_j
$$

then $s_j$ gets larger when $q$ and $k_j$ point in similar directions.

So:

- positive gradient on $s_j$ means "make them more aligned"
- negative gradient on $s_j$ means "make them less aligned"

This is the intuition behind gradients to $Q$ and $K$.

### 0.5 The only softmax fact you need

Softmax turns scores into probabilities:

$$
p_j = \frac{e^{s_j}}{\sum_t e^{s_t}}
$$

The important idea is:

- increasing one score increases its own probability
- but it also decreases the other probabilities

That coupling is why softmax backward has the subtraction term:

$$
dS_j = p_j (dP_j - \Delta)
$$

You do not need the full Jacobian yet. Just remember that softmax entries compete with each other.

### 0.6 What "gradient" means in this note

When I write $dO$, $dP$, $dQ$, $dK$, $dV$, read that as:

- the loss signal flowing into that variable
- or how much the loss wants that variable to change

That interpretation is enough for learning backprop.

## 1. What attention does

For one query vector $q$, a set of key vectors $k_j$, and value vectors $v_j$:

$$
s_j = q^T k_j
$$

$$
p = softmax(s)
$$

$$
o = \sum_j p_j v_j
$$

Meaning:

- $q$ asks a question
- $k_j$ says how relevant token $j$ is
- softmax turns raw scores into probabilities
- $o$ is a weighted average of the values $v_j$

So attention is:

1. compare $q$ with every $k$
2. turn those comparisons into weights
3. mix the $v$ vectors using those weights

## 2. What backpropagation means here

Suppose training gives us an upstream gradient $dO$.

That means:

- the final loss wants the output $o$ to change
- we need to figure out how much of that change should go into $q$, $k$, and $v$

We will do this one piece at a time.

## 3. Gradient with respect to $V$

The output is:

$$
o = p_1 v_1 + p_2 v_2 + \cdots
$$

Each $v_j$ is used with weight $p_j$, so:

$$
dV_j = p_j dO
$$

Intuition:

- if token $j$ got 70% of the attention weight, then 70% of the output gradient flows into $v_j$

## 4. Gradient with respect to attention weights $P$

The output also depends on the weights $p_j$.

If we change $p_j$, we change how much $v_j$ contributes to the output. That gives:

$$
dP_j = dO^T v_j
$$

Intuition:

- if $v_j$ points in a direction that would help reduce the loss, then increasing $p_j$ is useful
- the dot product $dO^T v_j$ measures that usefulness

## 5. The only hard part: backprop through softmax

We have:

$$
p = softmax(s)
$$

Softmax is special because all probabilities are connected:

- if one probability goes up
- some others must go down

So we cannot treat each $p_j$ independently.

The correct result for one row is:

$$
dS_j = p_j (dP_j - \Delta)
$$

where

$$
\Delta = \sum_k p_k dP_k
$$

Here:

- $dS_j$ is the gradient of the score $s_j$
- $\Delta$ is one scalar for the whole row

### Why $\Delta$ matters

$\Delta$ removes the row-average effect. It keeps the softmax competition consistent.

Without it, gradients would act like each probability could change independently, which is false.

## 6. A very important identity

In attention:

$$
dP_j = dO^T v_j
$$

So:

$$
\Delta = \sum_k p_k dP_k
$$

$$
\Delta = \sum_k p_k (dO^T v_k)
$$

$$
\Delta = dO^T \left(\sum_k p_k v_k\right)
$$

$$
\Delta = dO^T o
$$

So instead of computing $\Delta$ from the full attention matrix, we can use:

$$
\Delta = dO^T o
$$

This identity is one of the key reasons FlashAttention backward is practical.

## 7. Gradient with respect to $Q$ and $K$

The scores came from dot products:

$$
s_j = q^T k_j
$$

Once we know $dS_j$, the rest is standard dot-product backward:

$$
dQ = \sum_j dS_j k_j
$$

$$
dK_j = dS_j q
$$

Intuition:

- if $dS_j$ is positive, we want $q$ and $k_j$ to align more
- if $dS_j$ is negative, we want them to align less

## 8. Full backward pass for ordinary attention

For matrices:

$$
S = QK^T
$$

$$
P = softmax(S)
$$

$$
O = PV
$$

Backward:

$$
dV = P^T dO
$$

$$
dP = dO V^T
$$

$$
\Delta_i = dO_i^T O_i
$$

$$
dS_{ij} = P_{ij} (dP_{ij} - \Delta_i)
$$

$$
dQ = dS K
$$

$$
dK = dS^T Q
$$

If there is a scale factor

$$
S = \alpha QK^T
$$

then $dQ$ and $dK$ also get multiplied by $\alpha$.

## 9. What FlashAttention changes

FlashAttention does not change the math above.

It changes memory usage.

Naive attention stores the whole probability matrix $P$, which is very large.

FlashAttention avoids storing full $P$.

Instead, during forward it saves only small summaries per row:

- $O$
- $LSE = \log \sum_j e^{s_j}$

Then during backward it recomputes score blocks on chip:

$$
S_{block} = Q_{block} K_{block}^T
$$

$$
P_{block} = \exp(S_{block} - LSE_{row})
$$

Then it applies the same backward formulas block by block:

$$
dV \leftarrow dV + P_{block}^T dO_{block}
$$

$$
dP = dO_{block} V_{block}^T
$$

$$
\Delta = row\_sum(dO_{block} * O_{block})
$$

$$
dS = P_{block} * (dP - \Delta)
$$

$$
dQ \leftarrow dQ + dS K_{block}
$$

$$
dK \leftarrow dK + dS^T Q_{block}
$$

So the main idea is:

- ordinary attention backward says what gradients to compute
- FlashAttention recomputes probabilities instead of storing them
- saved $LSE$ and $O$ are enough to recover the same gradients exactly

## 10. Tiny numeric example

Assume one query attends to 3 keys.

### Forward

Scores:

$$
s = [1, 2, 0]
$$

Softmax:

$$
p \approx [0.245, 0.665, 0.090]
$$

Suppose:

$$
v_1 = [1, 0]
$$

$$
v_2 = [0, 2]
$$

$$
v_3 = [3, 1]
$$

Then:

$$
o = 0.245 v_1 + 0.665 v_2 + 0.090 v_3
$$

$$
o \approx [0.515, 1.420]
$$

Now suppose the upstream gradient is:

$$
dO = [1, -1]
$$

### Backward to $V$

$$
dV_1 = 0.245 [1, -1] = [0.245, -0.245]
$$

$$
dV_2 = 0.665 [1, -1] = [0.665, -0.665]
$$

$$
dV_3 = 0.090 [1, -1] = [0.090, -0.090]
$$

### Backward to $P$

$$
dP_1 = [1, -1] [1, 0]^T = 1
$$

$$
dP_2 = [1, -1] [0, 2]^T = -2
$$

$$
dP_3 = [1, -1] [3, 1]^T = 2
$$

### Compute $\Delta$

$$
\Delta = dO^T o
$$

$$
\Delta = [1, -1] [0.515, 1.420]^T
$$

$$
\Delta \approx -0.905
$$

### Backward through softmax

$$
dS_1 = 0.245 (1 - (-0.905)) \approx 0.467
$$

$$
dS_2 = 0.665 (-2 - (-0.905)) \approx -0.728
$$

$$
dS_3 = 0.090 (2 - (-0.905)) \approx 0.261
$$

Notice:

$$
dS_1 + dS_2 + dS_3 \approx 0
$$

That is what we expect from softmax competition.

## 11. The single most important mental model

FlashAttention backward is just ordinary attention backward done in tiles.

Instead of storing the full attention matrix:

- forward stores row summaries
- backward recomputes score tiles
- backward reconstructs probability tiles
- backward accumulates $dQ$, $dK$, and $dV$

Same math, different memory strategy.

## 12. What to learn next

The best learning order is:

1. understand ordinary attention forward
2. understand why $dV = P^T dO$
3. understand why $dP = dO V^T$
4. understand why $dS = P * (dP - \Delta)$
5. understand why $\Delta = dO^T O$
6. then study FlashAttention tiling and recomputation

## 13. Sources

- FlashAttention repository: <https://github.com/Dao-AILab/flash-attention>
- FlashAttention paper: <https://arxiv.org/abs/2205.14135>
- FlashAttention-3 paper: <https://tridao.me/publications/flash3/flash3.pdf>
