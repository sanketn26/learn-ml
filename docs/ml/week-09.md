# Week 9 — PCA: JPEG for Tables

**Course:** Applied ML Foundations for SaaS Analytics  
**Who this is for:** Engineers who have compressed images or used `SELECT` a subset of columns because 200 of them were correlated.

---

## 🎯 What you will be able to do

- Picture PCA as **rotating the cloud** so the first axis points along the stretch
- Read a scree plot: “how many axes until the leftover is noise?”
- Use 2-D PCA as a map, not as a causal feature named “growth”
- Spot odd customers via reconstruction error
- Know when *not* to bother (we only have a handful of columns)

!!! think "Think of it like… photographing a dinner plate from above vs from the edge."

    The plate is 3-D. From the edge it looks like a line — you lost the interesting shape. From above you kept the wide part. PCA finds the “from above” angles automatically: the directions where customers differ the most.

    Or: **JPEG for a table**. Keep the big blobs of ink, drop the speckle. You cannot read a pixel-perfect original, but you can still tell it is a face (or a whale customer).

```python
from lib.course_data import find_data_dir, load_customer_360

DATA = find_data_dir()
```


## If you already write software

PCA is JPEG for a table. You keep the big blobs of ink (the directions customers actually differ) and drop the speckle (columns that are noisy copies of each other).

You already do a cheap version of this: `SELECT` five of two hundred correlated columns because the rest are linear echoes. PCA finds the “from above” camera angle automatically.

```
200 columns, many copies of “how engaged is this user?”
        │  PCA
        ▼
3 axes that explain most of the stretch
        │
        ▼
a 2-D scatter you can actually look at
+ a reconstruction error that flags oddballs
```

### What you must not do

Do not tell a PM to “improve PC3.” Principal components are *rotations*, not product levers. They do not have names until you look at the loadings (which original columns they lean on) — and even then, “this axis is mostly usage + events” is a description, not a roadmap item.

### Picture the dinner plate

A plate is 3-D. Photographed from the edge it is a line — you lost the interesting shape. Photographed from above you kept the wide part. PCA finds those “from above” angles: the directions where the cloud of customers is longest.

If you only have 8 columns and they already mean something, skip PCA. Compression is for when the table is wide and redundant, not for when it is already a clean Customer 360.

## Honesty first: the “curse of dimensionality”

Textbooks scare you with 10,000 columns. CloudWave’s customer table has ~7 numeric fields. PCA will still **draw a useful map**. It will not demonstrate the curse. If you one-hot 200 countries + 500 feature flags, *then* distances die and k-NN / K-Means get dizzy. That is the curse: in high dimensions everyone is far from everyone, so “nearest” stops meaning “similar.”

!!! engineer "Engineer mental model"

    PCA is a rotation + a truncation. The new axes (PC1, PC2, …) are mixes of your old columns. **They are not product features.** Do not tell a PM “we should improve PC3.” Tell them “PC1 is mostly engagement (usage + events + features).”

```python
df = load_customer_360(DATA)
cols = ["mrr", "tenure_days", "log_usage", "features_adopted", "total_events", "avg_session", "n_support"]
sample = df.sample(6000, random_state=0)
X = StandardScaler().fit_transform(sample[cols])

# Cartoon: a diagonal cloud, then the axes PCA would pick
rng = np.random.default_rng(1)
t = rng.normal(size=400)
cloud = np.c_[t, 0.3 * t + rng.normal(scale=0.25, size=400)]
pca_c = PCA(n_components=2).fit(cloud)

fig, ax = plt.subplots(figsize=(5.2, 4.2))
ax.scatter(cloud[:, 0], cloud[:, 1], s=10, alpha=0.4, c="#64748b")
origin = pca_c.mean_
for vec, name, color in zip(pca_c.components_, ["PC1 — the stretch", "PC2 — the leftover"],
                            ["#dc2626", "#2563eb"]):
    ax.arrow(*origin, *(vec * 2), color=color, width=0.03, head_width=0.15)
    ax.text(*(origin + vec * 2.2), name, color=color, fontsize=9)
ax.set_title("PCA = rotate until axis 1 follows the cigar")
ax.set_aspect("equal")
plt.tight_layout()
plt.show()
```

## Scree plot + a 2-D map of CloudWave

!!! math "Math, translated"

    **Explained variance ratio** = “what fraction of the total stretch lives on this axis?” If PC1+PC2 hold 70%, a 2-D scatter still looks like the data. If they hold 20%, you flattened a ball into a pancake and lost the plot.

```python
pca = PCA().fit(X)
evr = pca.explained_variance_ratio_

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(range(1, len(evr) + 1), evr, color="#6366f1")
axes[0].plot(range(1, len(evr) + 1), np.cumsum(evr), marker="o", color="#b45309")
axes[0].axhline(0.8, ls="--", color="#64748b")
axes[0].set_xlabel("component")
axes[0].set_ylabel("fraction of stretch")
axes[0].set_title("Scree — bars=each axis, line=cumulative")

xy = PCA(n_components=2).fit_transform(X)
plan_codes = sample["plan_type"].astype("category")
scatter = axes[1].scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.35, c=plan_codes.cat.codes, cmap="Set2")
handles = [plt.Line2D([0], [0], marker="o", ls="", color=plt.cm.Set2(i / max(len(plan_codes.cat.categories)-1, 1)),
                      label=cat) for i, cat in enumerate(plan_codes.cat.categories)]
axes[1].legend(handles=handles, title="plan", fontsize=8)
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
axes[1].set_title("Same customers, two axes")
plt.tight_layout()
plt.show()

print("Variance kept by first 2 / 3 components:",
      f"{evr[:2].sum():.0%} / {evr[:3].sum():.0%}")
```

## Loadings — what *is* PC1 in English?

A **loading** is how much each original column leans on that axis. Large absolute loading → that column is a big ingredient of the axis.

```python
loadings = pd.DataFrame(pca.components_[:3].T, index=cols, columns=["PC1", "PC2", "PC3"])
print(loadings.round(2).to_string())
print("\nRead the biggest numbers in PC1 and name it in a Slack message.")

# Reconstruction error as 'does not fit the usual recipe'
pca3 = PCA(n_components=3).fit(X)
recon = pca3.inverse_transform(pca3.transform(X))
err = ((X - recon) ** 2).sum(axis=1)
sample = sample.copy()
sample["recon_error"] = err
print("\nOddballs (high reconstruction error):")
print(sample.nlargest(8, "recon_error")[["user_id", "plan_type", "mrr", "total_usage", "recon_error"]].to_string(index=False))
```

!!! warning "Watch out"

    PCA axes are not causes. An anomaly is “does not compress well,” which might be a new customer type, a data bug, or a whale. Do not auto-ban them.


!!! success "Ship / don’t ship"

    Use PCA to *look* and to compress a wide one-hot jungle before k-NN. Prefer original columns for a churn model you have to explain. Skip the autoencoder / t-SNE flex until you have a picture PCA cannot draw.


## ✍️ Exercise

When you can explain the week out loud, do the [exercises](exercises/week-09.md). Starter: `python exercises/ml/week-09/starter.py` from the repo root.

## 🤔 Reflection

1. Why did we scale before PCA? (Otherwise MRR, in dollars, owns the first axis.)
2. If two columns are almost the same (usage and events), what should PC1 do?
3. When is “we reduced to 2-D” a vanity metric?

## 🔗 Next week

Committees of models. Bagging vs boosting — a room of reviewers vs a sequence of specialists hunting the last miss.
