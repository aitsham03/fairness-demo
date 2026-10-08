"""Fairness demo (Adult dataset): one model, with and without one threshold per group."""
import pandas as pd
from fairlearn.datasets import fetch_adult
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LIMIT = 0.10  # arbitrary demo value: justify it with domain experts in a real project

# 1) Data: drop a useless column, look at missing values and category shares
data = fetch_adult(as_frame=True)
df = data.data.drop(columns=["fnlwgt"], errors="ignore")   # census sampling weight
y = (data.target == ">50K").astype(int)                    # 1 = earns more than 50K
print(f"{df.shape[0]} people, {df.shape[1]} variables\n")
missing = df.isna().sum()
print("Columns with missing values:\n", missing[missing > 0], "\n")
print("Share per sex:\n", df["sex"].value_counts(normalize=True).round(3), "\n")
print("Real share of >50K per sex:\n", y.groupby(df["sex"], observed=True).mean().round(3))

# 2) Prepare: sex is kept aside (sensitive attribute), text -> 0/1 columns, split, scale
A = df["sex"]
X = pd.get_dummies(df.drop(columns="sex")).astype(float)
X_tr, X_te, y_tr, y_te, A_tr, A_te = train_test_split(
    X, y, A, test_size=0.3, random_state=0, stratify=y)
scaler = StandardScaler().fit(X_tr)
X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
def report(name, pred):
    mf = MetricFrame(metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
                     y_true=y_te, y_pred=pred, sensitive_features=A_te)
    gap = demographic_parity_difference(y_te, pred, sensitive_features=A_te)
    print(f"\n=== {name} ===\n{mf.by_group.round(3)}\n"
          f"Overall accuracy: {mf.overall['accuracy']:.3f} | gap between groups: {gap:.3f}")
    return gap

# 3) Original model, then the same model with one decision threshold per group
model = LogisticRegression(random_state=42, max_iter=5000).fit(X_tr, y_tr)
gap_original = report("Original model", model.predict(X_te))
fixed = ThresholdOptimizer(estimator=model, constraints="demographic_parity",
                           prefit=True, predict_method="predict_proba")
fixed.fit(X_tr, y_tr, sensitive_features=A_tr)
gap_fixed = report("Fixed decisions", fixed.predict(X_te, sensitive_features=A_te, random_state=0))

# 4) Guardrail: the CI turns red if the fixed decisions still have a gap above the limit
for name, gap in [("Original", gap_original), ("Fixed", gap_fixed)]:
    print(f"{name}: {'PASS' if gap <= LIMIT else 'FAIL'} (gap {gap:.3f}, limit {LIMIT})")
if gap_fixed > LIMIT:
    raise SystemExit("FAIRNESS CHECK FAILED")
