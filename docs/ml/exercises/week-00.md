# Exercises — Week 0 — Strong Python for AI Engineers

Do these after reading [Week 0 — Strong Python for AI Engineers](../week-00.md).

**1. Plan report.** Using only `csv` + `Counter`, print churn rate per `plan_type` from `subscriptions.csv`.

**2. Dataclass round-trip.** Build a `CustomerFeatures` from a subscription row. Write `to_payload(self) -> dict` that a JSON API could accept.

**3. MeanBaseline tests.** `assert` that `fit([2, 4, 6]).predict(2)` returns `[4.0, 4.0]`. `assert` that `predict` before `fit` raises.

**4. Foot-gun hunt.** Deliberately write the mutable-default version of `add_tag` and show the second call is dirty. Then fix it.
