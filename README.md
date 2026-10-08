# fairness-demo

MLOps course demo: measure and reduce unfairness with [Fairlearn](https://fairlearn.org).

## The dataset: Adult (UCI)

- Loaded with `fairlearn.datasets.fetch_adult`: **48,842 people, 14 variables** (numeric and categorical, for example age, education, occupation, hours worked per week).
- **Cleaning**: the column `fnlwgt` (a census sampling weight, not about the person) is dropped. The script then lists the **columns with missing values** and prints the **share of each category** (for example per sex).
- **Goal**: predict whether a person earns **more than $50,000 a year** (`>50K` or `<=50K`). In the code it becomes `1` / `0`.
- **Sensitive attribute**: `sex`. It is **removed from the model's inputs** and used only to compare groups.
- **Split**: 70 % to train, 30 % to test (same people for `X`, `y` and `sex`), stratified so both parts have the same share of `>50K`.
- Downloaded on the first run, then cached by Fairlearn (default folder `~/.fairlearn-data`), so a second run should work offline.
- The script prints first: number of people and variables, columns with missing values, share per sex, and the real share of `>50K` per sex.

## The model: logistic regression

A simple linear model for **classification** (despite its name). According to the [scikit-learn user guide](https://scikit-learn.org/stable/modules/linear_model.html):

1. It computes a **weighted sum** of the variables (the weights are learned from the training data).
2. It passes the sum through the **logistic function**, which gives a **probability between 0 and 1** of being `>50K`. This is the *score* (`predict_proba`).
3. To get a decision, a **threshold** is applied: **0.5 by default**. Above = "yes", below = "no" (`predict`).

Parameters used: `LogisticRegression(random_state=42, max_iter=5000)`, default solver. `max_iter=5000` lets the solver iterate long enough to converge (default 100). `random_state` only matters for the solvers `sag`, `saga` and `liblinear` (scikit-learn docs), so it has no effect with the default solver; it is kept for reproducibility.

All the fairness work below is about **that threshold**.

## The fix: ThresholdOptimizer (Fairlearn)

```
Original model       score >= 0.50       -> yes      (same threshold for everyone)
ThresholdOptimizer   women: score >= t_w -> yes      (one threshold per group,
                     men:   score >= t_m -> yes       chosen by the tool)
```

According to the [Fairlearn user guide](https://fairlearn.org/main/user_guide/mitigation/postprocessing.html):

- The model is **kept as it is** (`prefit=True`): nothing is retrained. Only the decision rule changes.
- For each group, it **creates separate thresholds** and applies them to the model's scores.
- To choose them, it **generates all possible thresholds** and keeps the best combination for the **objective** (accuracy by default) under the **fairness constraint** (`demographic_parity`: both groups get the same share of "yes").
- At prediction time it may **pick between two thresholds at random**, which is why the script sets `random_state=0`.
- It satisfies the constraint **exactly**, often **at the expense of accuracy**.
- It **needs the sensitive attribute at prediction time** (`sensitive_features` is required in `predict`). Check that using it is possible and allowed in your context.

## What the script does

1. Loads the data, drops `fnlwgt`, prints missing values and category shares.
2. Trains the logistic regression and prints, with `MetricFrame`, accuracy and share of "yes" **per sex**, plus the gap between groups (demographic parity difference).
3. Applies `ThresholdOptimizer` to the same model and prints the same table.
4. **Guardrail**: prints PASS / FAIL for both versions against a limit of `0.10` (arbitrary demo value). The script exits with an error, and the CI turns red, if the fixed decisions exceed the limit.

## Files

- `demo.py`: the whole demo in 50 lines.
- `fairness_demo.ipynb`: the same demo as a **step-by-step notebook** with detailed explanations (English, plus French speaker hints). Run all cells before presenting so the outputs are visible.
- `requirements.txt`: `fairlearn`, `scikit-learn`, `pandas`.
- `.github/workflows/fairness.yml`: runs `demo.py` on every push and pull request (GitHub Actions).

## Run

```bash
pip install -r requirements.txt jupyter
python demo.py                # the script
jupyter notebook fairness_demo.ipynb   # the notebook
```

## Sources

- Fairlearn: [fetch_adult](https://fairlearn.org/main/api_reference/generated/fairlearn.datasets.fetch_adult.html), [ThresholdOptimizer](https://fairlearn.org/main/api_reference/generated/fairlearn.postprocessing.ThresholdOptimizer.html), [MetricFrame](https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.MetricFrame.html)
- scikit-learn: [Logistic regression](https://scikit-learn.org/stable/modules/linear_model.html)
- Hardt, Price, Srebro, *Equality of Opportunity in Supervised Learning*, NIPS 2016 (basis of `ThresholdOptimizer`)
