# IA7060 - HW3 - Discriminant analysis (two populations)
# Data: breast_cancer.csv (569 patients, 30 predictors, malignant / benign)
# Training set (70%) -> weights k ; validation set (30%) -> cutoff tc

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, precision=4)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

FILE = "breast_cancer.csv"
VALID_SHARE = 0.30
SEED = 1

C_MALIG = "#eb6834"
C_BENIGN = "#2a78d6"


def save(name):
    plt.tight_layout()
    plt.savefig(name, dpi=150)
    plt.show()
    plt.close()


# Step 1: data and split 
print("\n STEP 1: DATA ")

names = ["radius", "texture", "perimeter", "area", "smoothness",
         "compactness", "concavity", "concave_points", "symmetry",
         "fractal_dimension"]
columns = ([f"mean_{n}" for n in names] + [f"se_{n}" for n in names] +
           [f"worst_{n}" for n in names] + ["target"])

# first row of the file is not data (569, 30, malignant, benign)
df = pd.read_csv(FILE, skiprows=1, header=None, names=columns)

# group 1 = malignant (target 0), group 2 = benign (target 1)
df["group"] = np.where(df["target"] == 0, 1, 2)
df["group_name"] = df["group"].map({1: "malignant", 2: "benign"})

print("Shape:", df.shape)
print("Missing values:", df.isna().sum().sum())
print(df["group_name"].value_counts())

# stratified split: 70/30 inside each group
rng = np.random.default_rng(SEED)
train_idx, valid_idx = [], []
for g in [1, 2]:
    idx = rng.permutation(np.where(df["group"] == g)[0])
    n_train = int(round((1 - VALID_SHARE) * len(idx)))
    train_idx += list(idx[:n_train])
    valid_idx += list(idx[n_train:])

train = df.iloc[sorted(train_idx)].reset_index(drop=True)
valid = df.iloc[sorted(valid_idx)].reset_index(drop=True)

for name, part in [("All", df), ("Training", train), ("Validation", valid)]:
    m = (part["group"] == 1).sum()
    print(f"{name:10s}: {len(part)} patients, {m} malignant ({m / len(part):.1%}), "
          f"{len(part) - m} benign")

n1 = (train["group"] == 1).sum()
n2 = (train["group"] == 2).sum()


#  Step 2: descriptives (training) 
print("\n----- STEP 2: DESCRIPTIVES (training) -----")

predictors = columns[:-1]
g1 = train[train["group"] == 1]
g2 = train[train["group"] == 2]

rows = []
for var in predictors:
    t_stat, p = stats.ttest_ind(g1[var], g2[var], equal_var=False)   # Welch t-test
    rows.append([var, g1[var].mean(), g1[var].std(), g2[var].mean(), g2[var].std(), t_stat, p])
desc = pd.DataFrame(rows, columns=["variable", "mean_malig", "sd_malig",
                                   "mean_benign", "sd_benign", "t", "p_value"])
desc = desc.sort_values("t", key=abs, ascending=False)
print(desc.round(4).to_string(index=False))
print("Significant at 5%:", (desc["p_value"] < 0.05).sum(), "of", len(desc))

top4 = desc["variable"].head(4).tolist()
print("\nCorrelations, 4 strongest variables:")
print(train[top4].corr().round(3))
print("\nCorrelations, size variables:")
print(train[["worst_radius", "worst_perimeter", "worst_area"]].corr().round(3))
print("\nCorrelation of worst_texture with the 4 strongest:")
print(train[["worst_texture"] + top4].corr().round(3).iloc[0, 1:])


#  Step 3: forward selection (training) 
# Y = 1 malignant, 0 benign
# start with the variable most correlated with Y, then add the variable with
# the largest SSR(X | variables in), F-test with SLE = 0.05
print("\n----- STEP 3: FORWARD SELECTION (training) -----")

y_train = train["group"].values
n_tr = len(y_train)
Y = (y_train == 1).astype(float)
SLE = 0.05
R2_MIN_GAIN = 0.01        # R2 gain below 1 point -> R2 has saturated
SST_Y = ((Y - Y.mean()) ** 2).sum()


def sse(var_list):
    Xr = np.column_stack([np.ones(n_tr)] + [train[v].values for v in var_list])
    b = np.linalg.lstsq(Xr, Y, rcond=None)[0]
    e = Y - Xr @ b
    return e @ e


def vif(var, others):
    if not others:
        return 1.0
    Xr = np.column_stack([np.ones(n_tr)] + [train[o].values for o in others])
    x = train[var].values
    b = np.linalg.lstsq(Xr, x, rcond=None)[0]
    r2 = 1 - ((x - Xr @ b) ** 2).sum() / ((x - x.mean()) ** 2).sum()
    return 1 / (1 - r2)


print("Most correlated with Y:",
      train[predictors].corrwith(pd.Series(Y, index=train.index)).abs().idxmax())

selected, fwd_rows, r2_old = [], [], 0.0
for step in range(1, 11):
    sse_in = sse(selected) if selected else SST_Y
    best_var, best_ssr = None, -1
    for var in predictors:
        if var in selected:
            continue
        ssr_extra = sse_in - sse(selected + [var])
        if ssr_extra > best_ssr:
            best_var, best_ssr = var, ssr_extra
    p_ = len(selected) + 1
    sse_new = sse(selected + [best_var])
    F = best_ssr / (sse_new / (n_tr - p_ - 1))
    p_val = stats.f.sf(F, 1, n_tr - p_ - 1)
    r2 = 1 - sse_new / SST_Y
    fwd_rows.append([step, best_var, best_ssr, F, p_val, r2, r2 - r2_old, vif(best_var, selected)])
    selected.append(best_var)
    r2_old = r2
    if p_val > SLE:
        break

fwd = pd.DataFrame(fwd_rows, columns=["step", "variable", "SSR(X|in)", "F", "p_value",
                                      "R2", "R2_gain", "VIF"])
print(fwd.round(4).to_string(index=False))

# number of variables: stop when the R2 gain is below 1 point
n_keep = 1
while n_keep < len(fwd) and fwd["R2_gain"].iloc[n_keep] >= R2_MIN_GAIN:
    n_keep += 1
X_VARS = selected[:n_keep]
print("Chosen variables:", X_VARS)

R_chosen = train[X_VARS].corr()
print(R_chosen.round(3))
print("VIF:", np.round(np.diag(np.linalg.inv(R_chosen.values)), 2))


# Step 4: matrices T, W, A (training) 
print("\n----- STEP 4: T, W, A -----")

X = train[X_VARS].values
X1 = g1[X_VARS].values
X2 = g2[X_VARS].values
xbar, xbar1, xbar2 = X.mean(axis=0), X1.mean(axis=0), X2.mean(axis=0)

Xd = X - xbar
Xd1 = X1 - xbar1
Xd2 = X2 - xbar2

T = Xd.T @ Xd
W = Xd1.T @ Xd1 + Xd2.T @ Xd2
A = T - W

print("xbar  =", xbar)
print("xbar1 =", xbar1)
print("xbar2 =", xbar2)
print("T =\n", T)
print("W =\n", W)
print("A =\n", A)
print("A/T on the diagonal:", np.round(np.diag(A) / np.diag(T), 3))


# Step 5: weights k (training)
print("\n----- STEP 5: k = S^-1 d -----")

S = W / (n1 + n2 - 2)
d = xbar1 - xbar2
S_inv = np.linalg.inv(S)
k = S_inv @ d

print("S =\n", np.round(S, 6))
print("S^-1 =\n", S_inv)
print("d =", d)
print("k =", k)
print("t = " + " + ".join(f"{kj:.4f}*{var}" for kj, var in zip(k, X_VARS)))

print("k_j * d_j:", np.round(k * d, 4), " total =", round(k @ d, 4))

# check 1: k has the same direction as the eigenvector of W^-1 A
eigval, eigvec = np.linalg.eig(np.linalg.inv(W) @ A)
i = np.argmax(eigval.real)
e1 = eigvec[:, i].real
print("eigenvector / first value:", e1 / e1[0])
print("k / first value:          ", k / k[0])
print("largest eigenvalue:", round(eigval[i].real, 4))

# check 2: SST = SSTR + SSE
SST, SSTR, SSE = k @ T @ k, k @ A @ k, k @ W @ k
print(f"SST = {SST:.4f}, SSTR = {SSTR:.4f}, SSE = {SSE:.4f}, SSTR/SSE = {SSTR / SSE:.4f}")

ratios = {var: A[j, j] / W[j, j] for j, var in enumerate(X_VARS)}
ratios["score t"] = SSTR / SSE
print("SSTR/SSE:", {key: round(float(val), 4) for key, val in ratios.items()})

panels = [(X1[:, j], X2[:, j], var) for j, var in enumerate(X_VARS)]
panels.append((X1 @ k, X2 @ k, "score t = k'x"))
fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.2))
for ax, (a1, a2, label), rt in zip(axes, panels, ratios.values()):
    bins = np.linspace(min(a1.min(), a2.min()), max(a1.max(), a2.max()), 30)
    ax.hist(a2, bins=bins, color=C_BENIGN, alpha=0.55, label="benign")
    ax.hist(a1, bins=bins, color=C_MALIG, alpha=0.55, label="malignant")
    ax.axvline(a2.mean(), color=C_BENIGN, lw=2, ls="--")
    ax.axvline(a1.mean(), color=C_MALIG, lw=2, ls="--")
    ytop = ax.get_ylim()[1]
    ax.annotate("", xy=(a1.mean(), ytop * 0.9), xytext=(a2.mean(), ytop * 0.9),
                arrowprops=dict(arrowstyle="<->", color="black"))
    ax.text((a1.mean() + a2.mean()) / 2, ytop * 0.93, "between", ha="center")
    ax.set_title(f"{label}\nSSTR/SSE = {rt:.2f}")
axes[0].set_ylabel("number of patients")
axes[0].legend()
save("step5_between_within.png")


#  Step 6: scores of the validation patients
print("\n----- STEP 6: SCORES (validation) -----")

valid["score"] = valid[X_VARS].values @ k
y_val = valid["group"].values
s_val = valid["score"].values
t1_bar, t2_bar = s_val[y_val == 1].mean(), s_val[y_val == 2].mean()
s1, s2 = s_val[y_val == 1].std(ddof=1), s_val[y_val == 2].std(ddof=1)
nv1, nv2 = (y_val == 1).sum(), (y_val == 2).sum()
print(f"n1 = {nv1}, n2 = {nv2}")
print(f"t1_bar = {t1_bar:.4f} (SD {s1:.4f}),  t2_bar = {t2_bar:.4f} (SD {s2:.4f})")
print("first patient:", valid.loc[0, X_VARS].tolist(), "score =", round(valid.loc[0, "score"], 4))

bins = np.linspace(s_val.min(), s_val.max(), 30)
plt.figure(figsize=(8, 4.2))
plt.hist(s_val[y_val == 2], bins=bins, color=C_BENIGN, alpha=0.55, label=f"benign (n={nv2}, SD {s2:.2f})")
plt.hist(s_val[y_val == 1], bins=bins, color=C_MALIG, alpha=0.55, label=f"malignant (n={nv1}, SD {s1:.2f})")
plt.axvline(t2_bar, color=C_BENIGN, lw=2, ls="--")
plt.axvline(t1_bar, color=C_MALIG, lw=2, ls="--")
plt.xlabel("score t")
plt.ylabel("number of patients")
plt.title("Scores of the validation patients")
plt.legend()
save("step6_scores.png")


# Step 7: cutoff tc (validation) 
# rule: t >= tc -> malignant (1), t < tc -> benign (2)
print("\n----- STEP 7: CUTOFF (validation) -----")


def classify(scores, tc):
    return np.where(scores >= tc, 1, 2)


def evaluate(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    missed = np.sum((y_true == 1) & (y_pred == 2))
    false_alarm = np.sum((y_true == 2) & (y_pred == 1))
    tn = np.sum((y_true == 2) & (y_pred == 2))
    return {"accuracy": (tp + tn) / len(y_true), "sensitivity": tp / (tp + missed),
            "specificity": tn / (tn + false_alarm), "missed": missed, "false_alarm": false_alarm}


# course formulas
tc_mid = (t1_bar + t2_bar) / 2
tc_n = (nv2 * t1_bar + nv1 * t2_bar) / (nv1 + nv2)
for name, tc in [("midpoint", tc_mid), ("unequal n", tc_n)]:
    e = evaluate(y_val, classify(s_val, tc))
    print(f"{name}: tc = {tc:.3f}, missed {e['missed']}, false alarms {e['false_alarm']}")

# error table: patients sorted by score, each score tried as the cutoff
err = valid[["group"] + X_VARS + ["score"]].sort_values("score").reset_index(drop=True)
err["malignant"] = (err["group"] == 1).astype(int)
fa, mi = [], []
for tc in err["score"]:
    e = evaluate(y_val, classify(s_val, tc))
    fa.append(e["false_alarm"])
    mi.append(e["missed"])
err["false_alarm"] = fa
err["missed"] = mi
err["errors"] = err["false_alarm"] + err["missed"]

best = err[err["errors"] == err["errors"].min()]
TC_MIN = best["score"].iloc[0]
print("min errors =", err["errors"].min(), "at tc =", best["score"].round(3).tolist())

# for each number of missed cancers, the cutoff with the fewest false alarms
front = (err.loc[err.groupby("missed")["false_alarm"].idxmin(), ["missed", "false_alarm", "score"]]
         .sort_values("missed", ascending=False))
m_min = int(err.loc[err["score"] == TC_MIN, "missed"].iloc[0])
front = front[front["missed"] <= m_min].reset_index(drop=True)
front["extra_FA_per_cancer"] = front["false_alarm"].diff() / -front["missed"].diff()
print(front.round(3).to_string(index=False))

TC = front["score"].iloc[-1]          # no missed cancer, fewest false alarms
print(f"chosen tc = {TC:.4f}")

# table for the Excel file (built by hand)
err[["malignant"] + X_VARS + ["score", "false_alarm", "missed", "errors"]].to_csv(
    "error_table_validation.csv", index=False)

plt.figure(figsize=(8, 4.4))
xs = err["score"].values
plt.step(xs, err["false_alarm"], where="post", color=C_BENIGN, lw=2, label="false alarms")
plt.step(xs, err["missed"], where="post", color=C_MALIG, lw=2, label="missed cancers")
plt.step(xs, err["errors"], where="post", color="#0f4c5c", lw=2.5, label="# of errors")
plt.axvline(TC_MIN, color="#c97a00", ls="--")
plt.axvline(TC, color="black", lw=2)
plt.xlim(15, 22.5)
plt.ylim(0, 30)
plt.xlabel("cutoff tc")
plt.ylabel("number of patients")
plt.title("Errors for each cutoff (validation)")
plt.legend()
save("step7_errors.png")


# Step 8: classification at tc (validation) 
print("\n----- STEP 8: CLASSIFICATION (validation) -----")

valid["predicted"] = classify(s_val, TC)
valid["classification"] = valid["predicted"].map({1: "malignant", 2: "benign"})
print(pd.crosstab(valid["group_name"], valid["classification"], margins=True))
e = evaluate(y_val, valid["predicted"].values)
print(f"accuracy {e['accuracy']:.4f}, sensitivity {e['sensitivity']:.4f}, "
      f"specificity {e['specificity']:.4f}, missed {e['missed']}, false alarms {e['false_alarm']}")
print(f"baseline (always benign): {(y_val == 2).mean():.4f}")

plt.figure(figsize=(8, 4.4))
plt.hist(s_val[y_val == 2], bins=bins, color=C_BENIGN, alpha=0.55, label="benign")
plt.hist(s_val[y_val == 1], bins=bins, color=C_MALIG, alpha=0.55, label="malignant")
plt.axvline(TC, color="black", lw=2, label=f"tc = {TC:.2f}")
plt.xlabel("score t")
plt.ylabel("number of patients")
plt.title("Classification rule on the validation patients")
plt.legend()
save("step8_rule.png")
