Below is a self-contained description of the full analysis: multivariate known columns with heterogeneous cardinalities, a (K)-ary target column, per-class suppression at threshold (T), bounded integer noise ([-e,e]), and a reconstruction attacker/solver that knows all feature values and only solves for targets.

---

## Basic intuition

You can view each released query answer as a **linear constraint** on the unknown target labels.

* The attacker knows exactly which rows match a query (because they know all feature columns).
* For each query and each target class (r), the released value is a noisy version of “how many rows in this query’s answer set have label (r).”

If a row’s label were different, it would change certain query counts by (\pm 1). The attacker can try to assign labels to rows so that all released noisy counts are simultaneously satisfied.

The question is: **when is that assignment essentially unique and correct?**

Two things drive identifiability:

1. **Coverage**: how many released constraints “touch” each row (especially constraints involving the row’s true class), and
2. **Noise strength**: how much ambiguity each constraint leaves because of the noise range ([-e,e]).

Suppression complicates coverage: constraints are only released when true per-class counts in the query’s answer set are at least (T). So a query that matches only a few rows often yields no usable constraints.

This analysis builds a **cheap mixing/coverage proxy** that predicts whether high-accuracy reconstruction is likely—without running the expensive solver.

---

## Model and notation

### Data

* (R) rows indexed by (i \in {1,\dots,R}).
* (C) known feature columns indexed by (j \in {1,\dots,C}).
* Column (j) takes values in a finite set of size (D_j).
* Feature values are i.i.d. and uniform:
  [
  X_{i,j} \sim \mathrm{Unif}{1,\dots,D_j},
  \quad \text{independent over } i,j.
  ]
* One target column (y_i) taking values in ({1,\dots,K}).
* Targets are i.i.d. with class probabilities
  [
  \Pr(y_i=r)=\pi_r,\qquad r=1,\dots,K,\qquad \sum_{r=1}^K \pi_r=1.
  ]

### Queries

A query is defined by choosing a subset of feature columns (S \subseteq {1,\dots,C}) and specifying a value for each chosen column:
[
q = (S, v_S).
]
It matches a row (i) if
[
X_{i,j} = v_j \quad \forall j\in S.
]
Let the matched set be
[
A(q) := { i : X_{i,j}=v_j\ \forall j\in S},
\quad |A(q)| =: G(q).
]

### Released answers (per class)

For each class (r), define the true class count:
[
U_{q,r} := #{ i\in A(q) : y_i=r}.
]

**Suppression rule (per class):**
[
\text{release class } r \text{ for query } q \iff U_{q,r} \ge T.
]

**Noise rule (if released):**
[
\tilde U_{q,r} = U_{q,r} + \eta_{q,r},\qquad
\eta_{q,r} \sim \mathrm{Unif}{-e,-e+1,\dots,e},
]
independently across ((q,r)).

The attacker observes the set of released (\tilde U_{q,r}) and knows (X) (hence knows every (A(q))). They try to reconstruct (y).

---

## Which queries matter: “row-specific” queries

To reason about how well a single row (i) is constrained, it is convenient to consider **row-specific queries**:

* Choose a subset of columns (S).
* Set (v_j = X_{i,j}) for each (j\in S).
* This query always includes row (i), because it matches itself.

There is exactly one such query for each pair ((i,S)).

These are the queries that “touch” row (i). The analysis measures how many usable released constraints these queries generate.

---

## Step 1: Match probability for a subset (S)

Fix a row (i) and a subset of columns (S). Consider a different row (\ell \ne i).

For a single column (j),
[
\Pr(X_{\ell,j}=X_{i,j}) = \frac{1}{D_j}
]
because both are uniform over (D_j) values.

Assuming independence across columns,
[
\Pr\big(\forall j\in S:\ X_{\ell,j}=X_{i,j}\big)
= \prod_{j\in S} \frac{1}{D_j}.
]

Define the subset match probability
[
p_S := \prod_{j\in S} D_j^{-1}.
]
For the empty subset (S=\varnothing), (p_S=1) (the query matches everyone).

---

## Step 2: Distribution of the query answer-set size

For fixed (S), the number of *other* rows matching row (i) on all columns in (S) is binomial:
[
X_S \sim \mathrm{Binomial}(R-1, p_S).
]
Therefore the total matched set size is
[
G_S := |A(i,S)| = 1 + X_S
= 1 + \mathrm{Binomial}(R-1,p_S).
]

This is the key link between feature cardinalities and how large query answers tend to be.

---

## Step 3: Target class counts within a group of size (g)

Condition on a group size (G_S=g). Then the class-count vector for that group is multinomial:
[
(U_{1},\dots,U_{K}) \sim \mathrm{Multinomial}(g; \pi_1,\dots,\pi_K).
]

Marginally, for each class (r),
[
U_r \sim \mathrm{Binomial}(g,\pi_r).
]

This gives tractable expressions for suppression probabilities.

---

## Step 4: Expected number of released class-counts for a group of size (g)

Define the indicator that class (r) is released:
[
I_r(g) := \mathbf{1}{U_r \ge T}.
]

The number of released class-counts for a group of size (g) is
[
Z(g) := \sum_{r=1}^K I_r(g).
]

Taking expectation and using the binomial marginal:
[
\mathbb{E}[Z(g)]
= \sum_{r=1}^K \Pr(U_r \ge T)
= \sum_{r=1}^K \Pr(\mathrm{Binomial}(g,\pi_r)\ge T).
]

Write this probability with the binomial survival function:
[
\Pr(\mathrm{Binomial}(g,\pi_r)\ge T)
= \mathrm{BinomSF}(T-1;,g,\pi_r).
]

So
[
\boxed{
\mathbb{E}[Z(g)] = \sum_{r=1}^K \mathrm{BinomSF}(T-1;,g,\pi_r).
}
]

Interpretation: for a group of size (g), this is the expected number of per-class counts that are actually published.

---

## Step 5: A reconstruction-relevant “own-class released” probability

For reconstructing row (i)’s label, the most directly informative released count is the count for its **true class**.

Let (y_i=r). In a group of size (g) that includes row (i), the count for class (r) can be written as:
[
U_r = 1 + V_r,\quad V_r \sim \mathrm{Binomial}(g-1,\pi_r),
]
because among the other (g-1) rows, each is class (r) with probability (\pi_r).

The own-class count is released iff (U_r \ge T), i.e.
[
1+V_r \ge T \iff V_r \ge T-1.
]

Thus, conditional on (y_i=r),
[
\Pr(\text{own-class released} \mid g,, y_i=r)
= \Pr(\mathrm{Binomial}(g-1,\pi_r)\ge T-1)
= \mathrm{BinomSF}(T-2;,g-1,\pi_r),
]
(with the usual convention that for (T\le 1), this probability is 1).

Averaging over the prior on (r),
[
\boxed{
p_{\mathrm{own}}(g)
:= \Pr(\text{own-class released}\mid g)
= \sum_{r=1}^K \pi_r,\mathrm{BinomSF}(T-2;,g-1,\pi_r).
}
]

Interpretation: in a group of size (g), this is the probability that the published output includes the count for row (i)’s true class.

---

## Step 6: Per-row “released coverage” measures across all subsets

Each row (i) participates in one row-specific query for every subset (S\subseteq {1,\dots,C}). There are (2^C) such subsets.

For a fixed subset (S), group size is random (G_S = 1+\mathrm{Binomial}(R-1,p_S)).

### 6.1 All-class released coverage: (d_{\mathrm{eff}})

For subset (S), expected number of released class-counts is
[
\mathbb{E}[\mathbb{E}[Z(G_S)\mid G_S]]
= \mathbb{E}!\left[\sum_{r=1}^K \mathrm{BinomSF}(T-1;,G_S,\pi_r)\right].
]

Summing over all subsets gives the expected number of released constraints “touching” a typical row:
[
\boxed{
d_{\mathrm{eff}}
================

\sum_{S\subseteq [C]}
\mathbb{E}!\left[\sum_{r=1}^K \mathrm{BinomSF}(T-1;,G_S,\pi_r)\right],
\quad
G_S=1+\mathrm{Binomial}(R-1,p_S),\quad
p_S=\prod_{j\in S} D_j^{-1}.
}
]

Interpretation: expected number of released noisy counts involving a row, counting all classes.

### 6.2 Own-class released coverage: (d_{\mathrm{true}})

For subset (S), the probability the own-class count is released is
[
\mathbb{E}[p_{\mathrm{own}}(G_S)].
]

Summing over all subsets:
[
\boxed{
d_{\mathrm{true}}
=================

\sum_{S\subseteq [C]}
\mathbb{E}!\left[
\sum_{r=1}^K \pi_r,\mathrm{BinomSF}(T-2;,G_S-1,\pi_r)
\right].
}
]

Interpretation: expected number of released constraints that directly constrain the row’s correct label.

---

## Step 7: Lower-tail (“weakest-covered row”) proxies (d_{\min}) and (d_{\mathrm{true},\min})

In a random dataset, not every row has identical realized coverage. To model a “hard” row, we approximate a lower quantile of the per-row coverage distribution.

Let the per-row coverage be a sum over subsets:
[
D_{\mathrm{true}}(i) = \sum_{S\subseteq[C]} W_{i,S}
]
where (W_{i,S}) is the Bernoulli indicator that the own-class count is released for row (i)’s query ((i,S)).

We approximate:

* (\mathbb{E}[D_{\mathrm{true}}(i)] \approx d_{\mathrm{true}}),
* (\mathrm{Var}(D_{\mathrm{true}}(i)) \approx \mathrm{var_true}) (computed by the code using conditional variance approximations),
* and then take a rough (1/R) lower quantile via a normal approximation:
  [
  \boxed{
  d_{\mathrm{true},\min}
  \approx
  \max\Big{0,\ d_{\mathrm{true}} + z_{1/R}\sqrt{\mathrm{var_true}}\Big},
  }
  ]
  where (z_{1/R} = \Phi^{-1}(1/R)) is the (1/R) standard normal quantile (negative).

Analogously, (d_{\min}) is computed from (d_{\mathrm{eff}}) and (\mathrm{var_eff}).

Interpretation: (d_{\mathrm{true},\min}) represents a conservative “weakest-covered record” level of own-class released constraints.

---

## Step 8: Noise-normalized identifiability scales (I)

A released count changes by (\pm 1) if you flip one row’s label between two classes, but the noise added is uniform on ({-e,\dots,e}), a set of size (2e+1).

Heuristic: each released noisy count provides about (1/(2e+1)) effective “independent discrimination units” against an incorrect label.

Therefore define noise-normalized measures:
[
\boxed{
I_{\mathrm{true}} := \frac{d_{\mathrm{true}}}{2e+1},
\qquad
I_{\mathrm{true},\min} := \frac{d_{\mathrm{true},\min}}{2e+1}.
}
]
and similarly (I_{\mathrm{eff}}=d_{\mathrm{eff}}/(2e+1)), (I_{\min}=d_{\min}/(2e+1)).

Interpretation: (I_{\mathrm{true},\min}) is the “effective number of noise-normalized constraints” per hard row.

---

## Step 9: From (I_{\mathrm{true},\min}) to a per-row error proxy (p_{\mathrm{row}})

To mislabel a row, at least one incorrect class label must remain feasible after all released constraints involving that row.

Heuristic: each released own-class constraint eliminates a fixed incorrect label with probability on the order of (1/(2e+1)). Over (d) such constraints, survival probability for a fixed wrong label behaves like
[
\exp!\left(-\frac{d}{2e+1}\right) = e^{-I}.
]

There are (K-1) incorrect labels. By a union bound,
[
\boxed{
p_{\mathrm{row}}
;\approx;
\min\Big{1,\ (K-1)\exp(-I_{\mathrm{true},\min})\Big}.
}
]

Interpretation: conservative proxy for the probability a randomly chosen (weakly covered) row is misclassified.

---

## Step 10: Translating (p_{\mathrm{row}}) into an accuracy-confidence statement

Let (E) be the number of misclassified rows. Under a simplifying approximation that row errors are i.i.d. Bernoulli with rate (p_{\mathrm{row}}),
[
E \sim \mathrm{Binomial}(R, p_{\mathrm{row}}).
]

Accuracy is (A = 1 - E/R). For a target accuracy (1-\alpha), the failure event is
[
A < 1-\alpha \iff E > \alpha R.
]

A Chernoff/KL bound yields, for (p_{\mathrm{row}} < \alpha),
[
\boxed{
\Pr(A < 1-\alpha)
=================

\Pr(E \ge \alpha R)
;\le;
\exp!\big(-R,\mathrm{KL}(\alpha|p_{\mathrm{row}})\big),
}
]
where
[
\mathrm{KL}(\alpha|p)
=====================

\alpha\ln\frac{\alpha}{p} + (1-\alpha)\ln\frac{1-\alpha}{1-p}.
]

Thus, if the right-hand side is (\le \beta), you obtain:
[
\boxed{
\Pr(A \ge 1-\alpha) \ge 1-\beta.
}
]

This is the basis for the ((\alpha,\beta)) screening output.

---

## Summary of the derived measures

1. **Subset match probability**
   [
   p_S = \prod_{j\in S} D_j^{-1}.
   ]

2. **Group size**
   [
   G_S = 1 + \mathrm{Binomial}(R-1,p_S).
   ]

3. **All-class released coverage per row**
   [
   d_{\mathrm{eff}}
   =
   \sum_{S\subseteq[C]}
   \mathbb{E}!\left[\sum_{r=1}^K \mathrm{BinomSF}(T-1;,G_S,\pi_r)\right].
   ]

4. **Own-class released coverage per row**
   [
   d_{\mathrm{true}}
   =
   \sum_{S\subseteq[C]}
   \mathbb{E}!\left[
   \sum_{r=1}^K \pi_r,\mathrm{BinomSF}(T-2;,G_S-1,\pi_r)
   \right].
   ]

5. **Lower-tail proxies** (d_{\min}), (d_{\mathrm{true},\min}) via approximate normal quantiles using computed variances.

6. **Noise-normalized versions**
   [
   I_{\mathrm{true},\min} = \frac{d_{\mathrm{true},\min}}{2e+1}
   \quad\text{(and similarly for others)}.
   ]

7. **Per-row error proxy**
   [
   p_{\mathrm{row}} = \min{1,(K-1)e^{-I_{\mathrm{true},\min}}}.
   ]

8. **Accuracy-confidence bound**
   [
   \Pr(\text{accuracy} < 1-\alpha) \le \exp!\big(-R,\mathrm{KL}(\alpha|p_{\mathrm{row}})\big),
   ]
   yielding (\Pr(\text{accuracy}\ge 1-\alpha)\ge 1-\beta) when the RHS (\le \beta).

---

Below is a “principled” (within the same modeling assumptions) **expected accuracy proxy**, plus how to turn it into **approximate accuracy intervals**—using only quantities you already have in `HeteroDAnalysis`.

---

## 1) Two levels of prediction: average-case vs worst-case

Your analysis produces two relevant “own-class coverage” quantities:

* (d_{\mathrm{true}}) (stored as `d_true_eff`): expected number of released own-class constraints touching a *typical* row.
* (d_{\mathrm{true},\min}) (stored as `d_true_min`): a conservative lower-tail proxy for the weakest-covered row.

These map to noise-normalized “evidence”:

[
I_{\mathrm{true}}=\frac{d_{\mathrm{true}}}{2e+1}
\qquad\text{and}\qquad
I_{\mathrm{true},\min}=\frac{d_{\mathrm{true},\min}}{2e+1}.
]

Intuitively:

* (I_{\mathrm{true}}) is an **average-case evidence budget** per row.
* (I_{\mathrm{true},\min}) is a **hard-row evidence budget**.

For expected accuracy, the right starting point is (I_{\mathrm{true}}), not (I_{\mathrm{true},\min}).

---

## 2) A principled expected per-row error proxy

Under the same heuristic used earlier:

* A fixed wrong label survives each released own-class constraint with probability about (\exp(-1/(2e+1))).
* Over (d) independent released own-class constraints, survival probability for one wrong label is roughly (\exp(-d/(2e+1))=\exp(-I)).
* There are (K-1) wrong labels, so by a union bound:
  [
  p_{\mathrm{row}}(I)\approx \min{1,\ (K-1)\exp(-I)}.
  ]

### Expected per-row error (average-case)

Use (I_{\mathrm{true}}) to get an average-case error rate:

[
\boxed{
p_{\mathrm{row,eff}}
:= \min{1,\ (K-1)\exp(-I_{\mathrm{true}})}.
}
]

### Expected accuracy proxy

Then the expected accuracy (proxy) is

[
\boxed{
\mathbb{E}[\mathrm{accuracy}]
\approx 1 - p_{\mathrm{row,eff}}.
}
]

This is “principled” in the sense that it is the natural expectation under the i.i.d. model if you treat rows as exchangeable and use the same exponential-survival approximation, but with the **mean evidence** (I_{\mathrm{true}}).

---

## 3) Optional refinement using `var_true` (heterogeneity correction)

The above uses only the mean (I_{\mathrm{true}}). If rows vary in coverage, Jensen’s inequality tells you:

* Because (e^{-I}) is convex, variation in (I) increases (\mathbb{E}[e^{-I}]), thus **increasing expected error** relative to using (e^{-\mathbb{E}[I]}).

You already have an approximate variance of the own-class coverage sum, `var_true`. Convert it to an (I)-variance:

[
\sigma_d^2 \approx \mathrm{var_true},
\qquad
\sigma_I^2 \approx \frac{\sigma_d^2}{(2e+1)^2}.
]

If you approximate (I) across rows as normal:
[
I \sim \mathcal{N}(\mu_I, \sigma_I^2),
\quad \mu_I = I_{\mathrm{true}},
]
then you can compute
[
\mathbb{E}[e^{-I}] = e^{-\mu_I + \frac{1}{2}\sigma_I^2}.
]

So a heterogeneity-adjusted expected per-row error proxy is:

[
\boxed{
p_{\mathrm{row,het}}
\approx \min\Big{1,\ (K-1)\exp\Big(-I_{\mathrm{true}}+\tfrac{1}{2}\sigma_I^2\Big)\Big}.
}
]

And therefore:

[
\boxed{
\mathbb{E}[\mathrm{accuracy}]
\approx 1 - p_{\mathrm{row,het}}.
}
]

This is a clean “next step” because it uses only (\mu_I) and (\sigma_I^2), which your analysis already estimates.

---

## 4) Approximate accuracy intervals (confidence intervals)

If you pick a per-row error rate (p) (either (p_{\mathrm{row,eff}}) or (p_{\mathrm{row,het}})), and assume row errors are i.i.d.:

[
E \sim \mathrm{Binomial}(R, p),
\qquad
\mathrm{accuracy} = 1 - \frac{E}{R}.
]

A central ((1-\gamma)) interval for the number of errors is:

[
E_{\mathrm{lo}} = F^{-1}(\gamma/2),
\quad
E_{\mathrm{hi}} = F^{-1}(1-\gamma/2),
]
with (F) the binomial CDF.

Convert to an accuracy interval:

[
\boxed{
\mathrm{acc}*{\mathrm{lo}} = 1 - \frac{E*{\mathrm{hi}}}{R},
\qquad
\mathrm{acc}*{\mathrm{hi}} = 1 - \frac{E*{\mathrm{lo}}}{R}.
}
]

Interpretation: “If the true per-row error rate were (p), then accuracy would fall in ([\mathrm{acc}*{lo}, \mathrm{acc}*{hi}]) with probability about (1-\gamma).”

This is not an adversarial guarantee; it is a model-based uncertainty interval.

---

## 5) What you’d compute from `HeteroDAnalysis`

Given you already store:

* `I_true_eff` (= I_{\mathrm{true}})
* `var_true` (\approx \sigma_d^2)
* `R`, `K`, `e`

you can compute:

### Expected accuracy (simple)

[
p_{\mathrm{row,eff}}=\min{1,(K-1)e^{-I_{\mathrm{true}}}},\quad
\mathbb{E}[\mathrm{accuracy}] \approx 1-p_{\mathrm{row,eff}}.
]

### Expected accuracy (heterogeneity-adjusted)

[
\sigma_I^2 \approx \frac{\mathrm{var_true}}{(2e+1)^2},
\quad
p_{\mathrm{row,het}}=\min{1,(K-1)e^{-I_{\mathrm{true}}+\frac{1}{2}\sigma_I^2}},
\quad
\mathbb{E}[\mathrm{accuracy}] \approx 1-p_{\mathrm{row,het}}.
]

### Approximate CI

Pick (p\in{p_{\mathrm{row,eff}}, p_{\mathrm{row,het}}}), and compute binomial quantiles for (E), then map to accuracy.

---

## 6) A concrete “reporting” suggestion

For a cheap summary that’s still informative, report:

* **Expected accuracy (mean evidence):** (1-p_{\mathrm{row,eff}})
* **Expected accuracy (heterogeneity-adjusted):** (1-p_{\mathrm{row,het}}) (usually more conservative)
* **Hard-row proxy:** (1-p_{\mathrm{row}}(I_{\mathrm{true},\min}))
* **Accuracy CI:** binomial interval using (p_{\mathrm{row,het}}) (if you want one number)

This gives you:

* an average-case expected accuracy,
* a conservative expected accuracy accounting for coverage variance,
* and a “weakest-row” sanity check.

