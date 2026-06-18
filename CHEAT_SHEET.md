# Essential Math for Data Science — Cheat Sheet

Compact revision guide with formulas, Python examples, and intuition.

---

## 1. Math Basics (SymPy)

| Operation | Formula / idea | Python |
|-----------|----------------|--------|
| Substitute | $f(x)=x^2 \Rightarrow f(2)=4$ | `f.subs(x, 2)` |
| Derivative | $\frac{d}{dx}\log(x^2)=\frac{2}{x}$ | `diff(log(x**2), x)` |
| Partial derivative | $\frac{\partial}{\partial x}(2x+2y)=2$ | `diff(2*x+2*y, x)` |
| Integral | $\int_0^1 x^2\,dx=\frac{1}{3}$ | `integrate(x**2, (x,0,1))` |
| Limit | $\lim_{n\to\infty}(1+\frac{1}{n})^n=e$ | `limit((1+1/n)**n, n, oo)` |

```python
from sympy import symbols, diff, integrate, limit, oo, log
x, n = symbols("x n")
diff(log(x**2), x)              # 2/x
integrate(x**2, (x, 0, 1))      # 1/3
limit((1 + 1/n)**n, n, oo)      # E
```

---

## 2. Derivatives

**Definition:** $f'(x)=\lim_{h\to 0}\frac{f(x+h)-f(x)}{h}$

**Intuition:** Instantaneous rate of change / slope of the tangent.

```python
def derivative_at_x(f, x, eps=1e-10):
    return (f(x + eps) - f(x)) / eps

f = lambda x: x**2
derivative_at_x(f, 2)   # ~4.0  (exact: f'(x)=2x)
```

| Rule | Formula |
|------|---------|
| Power | $\frac{d}{dx}x^n = nx^{n-1}$ |
| Exponential | $\frac{d}{dx}e^x = e^x$ |
| Log | $\frac{d}{dx}\ln x = \frac{1}{x}$ |
| Chain | $\frac{d}{dx}f(g(x)) = f'(g(x))\cdot g'(x)$ |

---

## 3. Integrals

**Riemann sum:** $\int_a^b f(x)\,dx = \lim_{n\to\infty}\sum_{i=1}^{n} f(x_i)\,\Delta x$, where $\Delta x=\frac{b-a}{n}$

**Intuition:** Area under the curve = sum of infinitely thin rectangles.

```python
def riemann_sum(f, a, b, n):
  width = (b - a) / n
  return sum(f(a + width*i + width/2) * width for i in range(n))

riemann_sum(lambda x: x**2 + 1, 0, 1, 100_000)  # ~1.333 (exact: 4/3)

from sympy import integrate, symbols
x = symbols("x")
integrate(x**2 + 1, (x, 0, 1))  # 4/3
```

---

## 4. Probability

### Binomial — fixed $n$ trials, success prob $p$

$$P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}$$

```python
from scipy.stats import binom
binom.pmf(k=3, n=10, p=0.4)                    # P(X=3)
binom.cdf(k=5, n=10, p=0.4)                    # P(X≤5)
1 - binom.cdf(k=49, n=137, p=0.4)              # P(X≥50)
```

### Beta — uncertainty about true $p$ after observing data

$$f(p)=\frac{p^{\alpha-1}(1-p)^{\beta-1}}{B(\alpha,\beta)}, \quad \alpha=\text{successes}+1,\; \beta=\text{failures}+1$$

```python
from scipy.stats import beta
a, b = 8 + 1, 2 + 1   # 8 successes, 2 failures
beta.cdf(0.9, a, b)                              # P(p ≤ 0.9)
beta.cdf(0.9, a, b) - beta.cdf(0.8, a, b)       # P(0.8 ≤ p ≤ 0.9)
```

---

## 5. Descriptive Statistics

### Central tendency

| Measure | Formula | Python |
|---------|---------|--------|
| Sample mean $\bar{x}$ | $\frac{1}{n}\sum x_i$ | `np.mean(x)` or `sum(x)/len(x)` |
| Population mean $\mu$ | $\frac{1}{N}\sum x_i$ | same, on full population |
| Weighted mean | $\frac{\sum x_i w_i}{\sum w_i}$ | `np.average(x, weights=w)` |
| Median | middle value (sorted) | `np.median(x)` |
| Mode | most frequent | `from scipy.stats import mode; mode(x)` |

```python
import numpy as np
x = [1, 3, 2, 5, 7, 0, 2, 3]
np.mean(x)                          # 2.875
np.average([90,80,63,87], weights=[.2,.2,.2,.4])  # 81.4
np.median([0, 1, 5, 9, 10, 14])    # 7.0
```

### Spread

| | Population | Sample |
|---|-----------|--------|
| Variance | $\sigma^2=\frac{\sum(x_i-\mu)^2}{N}$ | $s^2=\frac{\sum(x_i-\bar{x})^2}{n-1}$ |
| Std dev | $\sigma=\sqrt{\sigma^2}$ | $s=\sqrt{s^2}$ |

```python
np.var(x, ddof=0)    # population variance (divide by N)
np.var(x, ddof=1)    # sample variance (divide by n-1)
np.std(x, ddof=1)    # sample std dev
```

**Why $n-1$?** Bessel's correction — using $\bar{x}$ from the same sample underestimates spread; dividing by $n-1$ gives an unbiased estimate of population variance.

---

## 6. Normal Distribution

$X \sim \mathcal{N}(\mu, \sigma^2)$

**PDF:** $f(x)=\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$

**68–95–99.7 rule:** ~68% within $\mu\pm\sigma$, ~95% within $\mu\pm 2\sigma$, ~99.7% within $\mu\pm 3\sigma$

```python
from scipy.stats import norm
mean, std = 64.43, 2.99
norm.cdf(66, mean, std) - norm.cdf(62, mean, std)   # P(62 < X < 66)
norm.ppf(0.95, loc=mean, scale=std)                  # x where P(X≤x)=0.95
```

### Z-score

$$z=\frac{x-\mu}{\sigma} \qquad \text{reverse: } x=z\sigma+\mu$$

```python
z = (150_000 - 140_000) / 3_000   # 3.33 (3.33 std above mean)
x = z * 3_000 + 140_000           # back to original scale
```

---

## 7. Central Limit Theorem (CLT)

For i.i.d. data with mean $\mu$, std $\sigma$:

$$\bar{X} \sim N\!\left(\mu,\;\frac{\sigma}{\sqrt{n}}\right) \qquad SE=\frac{\sigma}{\sqrt{n}}$$

**Intuition:** Sample means cluster around $\mu$ and become normally distributed as $n$ grows (often $n\geq 30$), even if the original data is not normal. Larger $n$ → smaller SE → tighter estimates.

```python
import numpy as np
population = np.random.exponential(scale=2, size=100_000)
sample_means = [np.mean(np.random.choice(population, 30)) for _ in range(1000)]
# histogram of sample_means → bell-shaped
```

---

## 8. Confidence Intervals

### Known $\sigma$ — use z

$$CI = \bar{x} \pm z \cdot \frac{\sigma}{\sqrt{n}}$$

### Unknown $\sigma$ — use t

$$CI = \bar{x} \pm t_{\alpha/2,\; df} \cdot \frac{s}{\sqrt{n}}$$

| Confidence | z (known σ) | t (df=30, approx.) |
|------------|-------------|---------------------|
| 90% | 1.645 | 1.697 |
| 95% | 1.96 | 2.042 |
| 99% | 2.576 | 2.750 |

```python
from scipy.stats import norm, t
import math

def ci_mean(x, confidence=0.95, sigma=None):
    n = len(x)
    x_bar = np.mean(x)
    if sigma is None:
        se = np.std(x, ddof=1) / math.sqrt(n)
        crit = t.ppf(1 - (1 - confidence) / 2, df=n - 1)
    else:
        se = sigma / math.sqrt(n)
        crit = norm.ppf(1 - (1 - confidence) / 2)
    margin = crit * se
    return x_bar - margin, x_bar + margin

ci_mean([98, 102, 97, 101, 100])   # e.g. (97.1, 101.5) at 95%
```

**Interpretation:** If we repeated sampling many times, ~95% of intervals would contain the true $\mu$. The true mean is fixed; the interval varies per sample.

---

## 9. Hypothesis Testing — Full Guide

### General workflow

1. State $H_0$ (null) and $H_1$ (alternative)
2. Choose significance level $\alpha$ (usually 0.05)
3. Compute test statistic
4. Find p-value (or compare statistic to critical value)
5. **Reject $H_0$** if $p < \alpha$

**P-value:** Probability of seeing data this extreme (or more) if $H_0$ is true.

---

### Degrees of Freedom (df) — when to use $n-1$, $n-2$, $n-3$

| Context | df | Why |
|---------|-----|-----|
| One-sample t-test | $n-1$ | Estimate 1 parameter ($\mu$) from sample mean |
| Two-sample t-test (equal var) | $n_1+n_2-2$ | Estimate 2 group means |
| Paired t-test | $n-1$ | Work on $n$ differences, estimate 1 mean |
| Simple linear regression | $n-2$ | Estimate slope + intercept (2 params) |
| Multiple regression ($k$ predictors) | $n-k-1$ | Estimate $k$ slopes + 1 intercept |
| One-way ANOVA ($k$ groups) | $df_{between}=k-1$, $df_{within}=N-k$ | $k$ group means estimated |
| Chi-square goodness of fit ($k$ categories) | $k-1$ | One constraint: probabilities sum to 1 |
| Chi-square independence ($r$ rows, $c$ cols) | $(r-1)(c-1)$ | Row and column totals are fixed |

**Rule of thumb:** df = number of observations minus number of parameters estimated.

---

### t-value vs z-value

| | z-test | t-test |
|---|--------|--------|
| Use when | $\sigma$ known OR large $n$ | $\sigma$ unknown, small/medium $n$ |
| Distribution | Standard normal $N(0,1)$ | Student's t (heavier tails) |
| Formula | $z=\frac{\bar{x}-\mu_0}{\sigma/\sqrt{n}}$ | $t=\frac{\bar{x}-\mu_0}{s/\sqrt{n}}$ |

As $df \to \infty$, t-distribution → normal. For $n>30$, z and t are often close.

```python
from scipy.stats import t
t.ppf(0.975, df=9)    # critical t for 95% CI, df=9 → 2.262
t.cdf(2.262, df=9)    # 0.975
```

---

### 9.1 One-Sample Z-Test (known $\sigma$)

$H_0: \mu=\mu_0$ vs $H_1: \mu\neq\mu_0$

$$z=\frac{\bar{x}-\mu_0}{\sigma/\sqrt{n}}$$

```python
from scipy.stats import norm
x_bar, mu0, sigma, n = 11641, 10345, 552, 42
z = (x_bar - mu0) / (sigma / np.sqrt(n))
p_two_tailed = 2 * (1 - norm.cdf(abs(z)))
# p ≈ 0.0001 → reject H0
```

---

### 9.2 One-Sample t-Test (unknown $\sigma$)

$H_0: \mu=\mu_0$ vs $H_1: \mu\neq\mu_0$, **df = $n-1$**

$$t=\frac{\bar{x}-\mu_0}{s/\sqrt{n}}$$

```python
from scipy.stats import ttest_1samp
data = [16, 17, 15, 18, 16, 17, 16]
t_stat, p_value = ttest_1samp(data, popmean=18)
# H0: mean recovery = 18 days; small p → reject
```

---

### 9.3 Two-Sample t-Test (independent groups)

$H_0: \mu_1=\mu_2$ vs $H_1: \mu_1\neq\mu_2$

**Equal variances (pooled), df = $n_1+n_2-2$:**

$$t=\frac{\bar{x}_1-\bar{x}_2}{s_p\sqrt{\frac{1}{n_1}+\frac{1}{n_2}}}, \quad s_p^2=\frac{(n_1-1)s_1^2+(n_2-1)s_2^2}{n_1+n_2-2}$$

```python
from scipy.stats import ttest_ind
group_a = [85, 90, 78, 92, 88]
group_b = [70, 75, 72, 68, 74]
t_stat, p = ttest_ind(group_a, group_b)              # equal var (default)
t_stat, p = ttest_ind(group_a, group_b, equal_var=False)  # Welch's t
```

**Welch's t-test** (unequal variances): df computed via Welch–Satterthwaite (not a simple integer).

---

### 9.4 Paired t-Test (same subjects, before/after)

$H_0: \mu_d=0$ where $d_i = x_{before,i}-x_{after,i}$, **df = $n-1$**

$$t=\frac{\bar{d}}{s_d/\sqrt{n}}$$

```python
from scipy.stats import ttest_rel
before = [120, 135, 128, 140, 125]
after  = [115, 130, 122, 138, 120]
t_stat, p = ttest_rel(before, after)
```

---

### 9.5 One-Proportion Z-Test

$H_0: p=p_0$ vs $H_1: p\neq p_0$

$$z=\frac{\hat{p}-p_0}{\sqrt{p_0(1-p_0)/n}}, \quad \hat{p}=\frac{x}{n}$$

```python
from statsmodels.stats.proportion import proportions_ztest
count = 45    # successes
nobs = 100
z_stat, p = proportions_ztest(count, nobs, value=0.5)  # H0: p=0.5
```

---

### 9.6 Two-Proportion Z-Test

$H_0: p_1=p_2$

$$z=\frac{\hat{p}_1-\hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})(\frac{1}{n_1}+\frac{1}{n_2})}}, \quad \hat{p}=\frac{x_1+x_2}{n_1+n_2}$$

```python
from statsmodels.stats.proportion import proportions_ztest
count = np.array([45, 30])    # successes per group
nobs  = np.array([100, 100])  # trials per group
z_stat, p = proportions_ztest(count, nobs)
```

---

### 9.7 Chi-Square Goodness of Fit

$H_0$: observed frequencies match expected distribution, **df = $k-1$** ($k$ categories)

$$\chi^2=\sum_{i=1}^{k}\frac{(O_i-E_i)^2}{E_i}$$

```python
from scipy.stats import chisquare
observed  = [50, 30, 20]
expected  = [40, 40, 20]   # must sum to same total as observed
chi2, p = chisquare(f_obs=observed, f_exp=expected)
```

---

### 9.8 Chi-Square Test of Independence

$H_0$: variables are independent, **df = $(r-1)(c-1)$**

```python
from scipy.stats import chi2_contingency
# rows = treatment, cols = outcome
table = [[30, 10],
         [20, 40]]
chi2, p, dof, expected = chi2_contingency(table)
# dof = (2-1)(2-1) = 1
```

---

### 9.9 One-Way ANOVA

$H_0: \mu_1=\mu_2=\cdots=\mu_k$ (all group means equal)

**Between-group df = $k-1$, within-group df = $N-k$** ($k$ groups, $N$ total observations)

$$F=\frac{MS_{between}}{MS_{within}}$$

```python
from scipy.stats import f_oneway
group1 = [23, 25, 22, 24]
group2 = [30, 28, 32, 29]
group3 = [27, 26, 28, 25]
f_stat, p = f_oneway(group1, group2, group3)
```

**When to use ANOVA vs t-test:** t-test for 2 groups; ANOVA for 3+ groups. ANOVA significant → at least one mean differs (follow up with post-hoc tests).

---

### Test Selection Cheat Sheet

| Question | Test | Python |
|----------|------|--------|
| Is mean = $\mu_0$? (σ known) | One-sample z | manual + `norm.cdf` |
| Is mean = $\mu_0$? (σ unknown) | One-sample t | `ttest_1samp` |
| Are two group means equal? | Two-sample t | `ttest_ind` |
| Did treatment change same subjects? | Paired t | `ttest_rel` |
| Is proportion = $p_0$? | One-proportion z | `proportions_ztest` |
| Are two proportions equal? | Two-proportion z | `proportions_ztest` |
| Does data fit expected distribution? | Chi-square GOF | `chisquare` |
| Are two categorical vars independent? | Chi-square | `chi2_contingency` |
| Are 3+ group means equal? | One-way ANOVA | `f_oneway` |

---

## 10. Linear Algebra

$$\mathbf{v}' = \text{basis} \cdot \mathbf{v} \qquad A\mathbf{x}=\mathbf{b} \Rightarrow \mathbf{x}=A^{-1}\mathbf{b}$$

**Determinant:** scaling factor of area/volume; $\det(A)=0$ → not invertible.

```python
import numpy as np
from numpy.linalg import det, inv

basis = np.array([[2, 0], [0, 3]]).T
v = np.array([1, 1])
basis @ v                    # [2, 3]
det(basis)                   # 6.0

A = np.array([[4, 2, 4], [5, 3, 7], [9, 3, 6]])
B = np.array([44, 56, 72])
inv(A) @ B                   # [2, 34, -8]
np.linalg.solve(A, B)        # preferred over inv(A)@B
```

---

## 11. NumPy

### Array creation & properties

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([[14, 16, 15], [10, 12, 13]])
np.zeros((3, 4));  np.ones(5);  np.arange(0, 10, 2)
np.linspace(0, 1, 5);  np.random.rand(3, 3);  np.eye(3)

b.shape    # (2, 3)   b.ndim → 2   b.size → 6   b.dtype
```

### Indexing & slicing

```python
b[:, 0]              # all rows, col 0  → [14, 10]
b[0, :]              # first row
b[b[:, 0] >= 14]     # boolean filter rows
b[[True, False]]     # row mask
b[0, 1]              # single element → 16
```

### Axis operations

```python
b.mean(axis=0)       # per-column mean → [12, 14, 14]
b.mean(axis=1)       # per-row mean
b.sum(axis=0);  b.max(axis=1);  b.std(axis=0, ddof=1)
np.column_stack((b, b.mean(axis=1)))   # add column
```

### Broadcasting

Smaller array is "stretched" to match larger — no copy if possible.

```python
a = np.array([[1, 2, 3], [4, 5, 6]])   # (2, 3)
a + 10                                  # add 10 to every element
a * np.array([1, 2, 3])                # (2,3) * (3,) → row-wise scale
```

### Linear algebra

```python
A = np.array([[1, 2], [3, 4]])
A.T;  A @ B;  np.linalg.inv(A);  np.linalg.det(A)
np.linalg.eig(A)          # eigenvalues & eigenvectors
np.linalg.qr(A)           # QR decomposition
```

### Useful functions

| Function | Purpose |
|----------|---------|
| `np.reshape`, `np.flatten` | Change shape |
| `np.concatenate`, `np.vstack`, `np.hstack` | Join arrays |
| `np.where(cond, x, y)` | Conditional selection |
| `np.unique` | Unique values |
| `np.argmax`, `np.argmin` | Index of max/min |
| `np.percentile` | Quantiles |

---

## 12. Pandas

### Create & inspect

```python
import pandas as pd

df = pd.DataFrame({"math": [10, 20, 15], "bio": [12, 18, 14]})
s  = pd.Series([1, 2, 3], index=["a", "b", "c"])

df.head();  df.tail();  df.info();  df.describe();  df.shape
```

### Selection

```python
df["math"]                    # Series (one column)
df[["math", "bio"]]           # DataFrame (multiple columns)
df.loc[0, "math"]             # by label
df.iloc[0:2, 0:1]             # by integer position
df.loc[df["math"] > 15, :]    # filter rows
df[df["math"].isin([10, 15])]
```

### Sorting, missing values, new columns

```python
df.sort_values("math", ascending=False)
df["total"] = df["math"] + df["bio"]
df.drop("total", axis=1)
df.rename(columns={"math": "mathematics"})

df.isna().sum()               # count missing per column
df.dropna()                   # drop rows with any NaN
df.fillna(0)                  # fill missing with 0
df["bio"].fillna(df["bio"].mean())
```

### Grouping & aggregation

```python
df.groupby("bio")["math"].mean()
df.groupby("bio").agg({"math": ["mean", "sum", "count"], "bio": "mean"})
df["bio"].rank()
```

### Merge & concat

```python
pd.merge(df1, df2, on="id", how="inner")   # inner, left, right, outer
pd.concat([df1, df2], axis=0)              # stack rows
pd.concat([df1, df2], axis=1)              # side by side
```

### Useful methods

| Method | Purpose |
|--------|---------|
| `df.value_counts()` | Frequency of values |
| `df.corr()` | Correlation matrix |
| `df.apply(func)` | Apply function to rows/cols |
| `df.pivot_table()` | Spreadsheet-style summary |
| `pd.read_csv()`, `df.to_csv()` | I/O |

---

## 13. Matplotlib

```python
import matplotlib.pyplot as plt
import numpy as np

# Line plot (function)
x = np.linspace(-3, 3, 100)
plt.plot(x, x**2, label="x²")
plt.xlabel("x");  plt.ylabel("y");  plt.legend();  plt.grid(True)

# Scatter plot (dataset)
plt.scatter(df["x"], df["y"], color="blue", alpha=0.7)

# Histogram
plt.hist(data, bins=20, edgecolor="black")

# Regression line overlay
plt.scatter(X, y)
plt.plot(X, m * X + b, color="red")

# Multiple subplots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(x, x**2)
axes[1].hist(data, bins=15)
plt.tight_layout()
plt.show()

# Save
plt.savefig("plot.png", dpi=150)
```

### Customization quick ref

| Call | Effect |
|------|--------|
| `plt.title("...")` | Title |
| `plt.xlim(0, 10)` | X-axis range |
| `plt.colorbar()` | Color scale (for heatmaps) |
| `plt.style.use("seaborn-v0_8")` | Style preset |
| `fig, ax = plt.subplots()` | Object-oriented API (preferred) |

```python
fig, ax = plt.subplots()
ax.scatter(x, y, s=50, c="green", marker="o")
ax.set_title("My Plot");  ax.set_xlabel("x")
```

---

## 14. Scikit-learn

### Train / test split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)
```

### Preprocessing

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

scaler = StandardScaler()          # z-score: (x - mean) / std
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)   # use train stats!

le = LabelEncoder()
y_encoded = le.fit_transform(["cat", "dog", "cat"])  # [0, 1, 0]
```

### Regression

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

model.coef_          # slopes
model.intercept_     # intercept
r2_score(y_test, y_pred)
mean_squared_error(y_test, y_pred)
```

**Formulas:**

$$\hat{y}=mx+b \qquad SSE=\sum(y_i-\hat{y}_i)^2 \qquad R^2=1-\frac{SS_{res}}{SS_{tot}}$$

**Closed form:** $m=\frac{n\sum xy-(\sum x)(\sum y)}{n\sum x^2-(\sum x)^2}$, $b=\bar{y}-m\bar{x}$

**Normal equation:** $\beta=(X^T X)^{-1}X^T y$

```python
import numpy as np
X_1 = np.column_stack([X.flatten(), np.ones(len(X))])
beta = np.linalg.inv(X_1.T @ X_1) @ X_1.T @ y   # [slope, intercept]
```

### Classification (Logistic Regression)

$$p=\frac{1}{1+e^{-(\beta_0+\beta_1 x)}} \qquad \log L=\sum_i\big[y_i\log p_i+(1-y_i)\log(1-p_i)\big]$$

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
model.predict_proba(X_test)       # probabilities per class

confusion_matrix(y_test, y_pred)
classification_report(y_test, y_pred)   # precision, recall, F1
accuracy_score(y_test, y_pred)
```

### Evaluation metrics

| Metric | Formula | sklearn |
|--------|---------|---------|
| Accuracy | $\frac{TP+TN}{TP+TN+FP+FN}$ | `accuracy_score` |
| Precision | $\frac{TP}{TP+FP}$ | `precision_score` |
| Recall | $\frac{TP}{TP+FN}$ | `recall_score` |
| F1 | $2\frac{PR}{P+R}$ | `f1_score` |
| MSE | $\frac{1}{n}\sum(y-\hat{y})^2$ | `mean_squared_error` |
| R² | $1-\frac{SS_{res}}{SS_{tot}}$ | `r2_score` |

### Common classes

| Task | Class |
|------|-------|
| Linear regression | `LinearRegression` |
| Logistic regression | `LogisticRegression` |
| k-NN | `KNeighborsClassifier` |
| Decision tree | `DecisionTreeClassifier` |
| Random forest | `RandomForestClassifier` |
| SVM | `SVC` |
| k-means | `KMeans` |
| PCA | `PCA` |

### Typical workflow

```python
# 1. Load → 2. Split → 3. Scale → 4. Fit → 5. Predict → 6. Evaluate
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
```

---

## 15. Gradient Descent

$$\theta \leftarrow \theta - \eta \nabla J(\theta)$$

```python
# Minimize f(x) = (x-3)² + 4
x, lr = 0.0, 0.001
for _ in range(100_000):
    x -= lr * 2 * (x - 3)    # f'(x) = 2(x-3)
# x → 3.0

# Linear regression: minimize SSE
m, b, lr = 0.0, 0.0, 0.0001
for _ in range(100_000):
    D_m = sum(2 * p.x * ((m*p.x + b) - p.y) for p in points)
    D_b = sum(2 * ((m*p.x + b) - p.y) for p in points)
    m -= lr * D_m;  b -= lr * D_b
```

**Stochastic GD:** update on random mini-batches each epoch — faster on large data.

---

## 16. Pearson Correlation

$$r=\frac{n\sum xy-(\sum x)(\sum y)}{\sqrt{n\sum x^2-(\sum x)^2}\;\sqrt{n\sum y^2-(\sum y)^2}}$$

| $r$ | Meaning |
|-----|---------|
| 1 | Perfect positive linear |
| 0 | No linear relationship |
| −1 | Perfect negative linear |

```python
df.corr(method="pearson")
np.corrcoef(x, y)
```

---

## Quick Reference: Libraries

| Library | Role |
|---------|------|
| **sympy** | Symbolic math |
| **numpy** | Arrays, linear algebra |
| **pandas** | Tabular data |
| **scipy.stats** | Distributions, hypothesis tests |
| **statsmodels** | Proportion tests, statistical models |
| **matplotlib** | Plotting |
| **sklearn** | Machine learning |

---

## Concept Map

```
Calculus → Probability → Statistics (normal, CLT, CI)
                              ↓
                    Hypothesis Testing (z, t, χ², ANOVA)
                              ↓
         Linear Algebra → NumPy / Pandas / Matplotlib
                              ↓
              Linear Regression → Logistic Regression → sklearn
```

---

*Covers notebooks `1-mathbasics` through `12-logistic-regression`.*
