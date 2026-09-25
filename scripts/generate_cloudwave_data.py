"""Regenerate CloudWave's churn, telemetry, usage, and feedback. Run from the repo root.

    python scripts/generate_cloudwave_data.py

Customers, plans, and MRR in `data/subscriptions.csv` are kept. Everything
else is re-simulated from one seeded story:

- Signups grow about 2% a month (Jan 2022 → Nov 2024), like a healthy SMB SaaS.
- Each customer has a hidden engagement level and a hidden friction level.
- Churn is a monthly hazard (about 2% of the active base a month): higher on
  free plans, in the first three months, for low-engagement and high-friction
  customers. Churn dates run to the observation end, 2024-11-30.
- Telemetry and usage only happen between signup and churn, inside the log
  window (2023-01-01 → 2024-11-30). Churners fade over their last ~8 weeks and
  write more support messages and downgrades on the way out; a `cancel` event
  lands on the churn date.
- Each customer has one home region and one or two devices.
- Feedback is dated, written while the customer is active, and leans negative
  for customers heading for the door.

Deterministic for a given SEED. `product_catalog.csv` is not touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SEED = 7
LOG_START = pd.Timestamp("2023-01-01")
OBSERVATION_END = pd.Timestamp("2024-11-30")

TARGET_EVENTS = 400_000
TARGET_USAGE_ROWS = 250_000
N_FEEDBACK = 10_000

# Monthly churn hazard before engagement/friction/tenure multipliers.
BASE_HAZARD = {"free": 0.016, "starter": 0.010, "pro": 0.006, "enterprise": 0.003}
PLAN_ACTIVITY = {"free": 0.6, "starter": 1.0, "pro": 1.4, "enterprise": 1.8}
REGIONS = ["NAMER", "EMEA", "APAC", "LATAM"]  # not "NA": pandas reads that as missing
REGION_P = [0.35, 0.30, 0.20, 0.15]
REGION_HAZARD = {"NAMER": 1.0, "EMEA": 0.95, "APAC": 1.05, "LATAM": 1.15}
DEVICES = ["web", "ios", "android"]
DEVICE_P = [0.55, 0.25, 0.20]
FEATURES = ["dashboard", "reports", "analytics", "search", "export", "collaboration", "automation", "api_call"]
FEATURE_P = [0.22, 0.16, 0.14, 0.13, 0.11, 0.10, 0.08, 0.06]

EVENT_TYPES = ["login", "page_view", "click", "feature_use", "payment", "support_message", "upgrade", "downgrade"]
EVENT_BASE_P = np.array([0.35, 0.18, 0.13, 0.09, 0.03, 0.02, 0.012, 0.008])
FADE_DAYS = 56  # churners' activity fades over their last eight weeks


def draw_signups(n: int, rng: np.random.Generator) -> pd.Series:
    """Signup dates on a steady growth curve, a uniform day inside each month."""
    months = pd.date_range("2022-01-01", OBSERVATION_END, freq="MS")
    weights = 1.02 ** np.arange(len(months))
    month = rng.choice(len(months), size=n, p=weights / weights.sum())
    start = months[month]
    span = np.array([(min(m + pd.offsets.MonthBegin(1), OBSERVATION_END + pd.Timedelta(days=1)) - m).days
                     for m in months])[month]
    return pd.Series(start + pd.to_timedelta((rng.random(n) * span).astype(int), unit="D"))


def simulate_churn(subs: pd.DataFrame, rng: np.random.Generator, engagement, friction, region) -> pd.Series:
    """Walk each customer forward a month at a time; draw churn from the hazard."""
    base = subs["plan_type"].map(BASE_HAZARD).to_numpy()
    region_mult = pd.Series(region).map(REGION_HAZARD).to_numpy()
    person = np.exp(-1.1 * engagement + 0.5 * friction) * region_mult

    signup = subs["signup_date"].to_numpy(dtype="datetime64[D]")
    end = np.datetime64(OBSERVATION_END.date())
    churn = np.full(len(subs), np.datetime64("NaT", "D"), dtype="datetime64[D]")
    alive = np.ones(len(subs), dtype=bool)
    for month in range(40):
        start = signup + np.timedelta64(int(month * 30.44), "D")
        stop = signup + np.timedelta64(int((month + 1) * 30.44), "D")
        at_risk = alive & (start <= end)
        if not at_risk.any():
            break
        tenure_mult = 1.8 if month < 3 else (1.3 if month < 6 else 1.0)
        h = np.clip(base * tenure_mult * person, 0, 0.6)
        leaves = at_risk & (rng.random(len(subs)) < h)
        span = (stop - start).astype(int)
        day = start + (rng.random(len(subs)) * span).astype(int).astype("timedelta64[D]")
        leaves &= day <= end
        churn[leaves] = day[leaves]
        alive &= ~leaves
    return pd.Series(pd.to_datetime(churn), index=subs.index)


def activity_grid(subs, churn, engagement, rng):
    """Users × weeks intensity: engagement × plan × onboarding ramp × fade before churn."""
    weeks = pd.date_range(LOG_START, OBSERVATION_END, freq="7D")
    w_start = weeks.to_numpy(dtype="datetime64[s]")
    w_end = np.minimum(w_start + np.timedelta64(7, "D"), np.datetime64(OBSERVATION_END + pd.Timedelta(days=1)))

    a_start = np.maximum(subs["signup_date"].to_numpy(dtype="datetime64[s]"), np.datetime64(LOG_START))
    a_end = churn.fillna(OBSERVATION_END + pd.Timedelta(days=1)).to_numpy(dtype="datetime64[s]")
    lo = np.maximum(a_start[:, None], w_start[None, :])
    hi = np.minimum(a_end[:, None], w_end[None, :])
    frac = np.clip((hi - lo).astype("timedelta64[s]").astype(float) / (7 * 86400), 0, 1)

    tenure_weeks = (w_start[None, :] - subs["signup_date"].to_numpy(dtype="datetime64[s]")[:, None]).astype(
        "timedelta64[D]"
    ).astype(float) / 7
    ramp = np.where(tenure_weeks < 4, 0.6, 1.0)
    days_left = (a_end[:, None] - w_start[None, :]).astype("timedelta64[D]").astype(float)
    churner = churn.notna().to_numpy()[:, None]
    fade = np.where(churner, 0.1 + 0.9 * np.clip(days_left / FADE_DAYS, 0, 1), 1.0)
    week_noise = rng.lognormal(0, 0.35, size=frac.shape)

    person = np.exp(0.9 * engagement) * subs["plan_type"].map(PLAN_ACTIVITY).to_numpy()
    intensity = person[:, None] * ramp * fade * week_noise * frac
    near_churn = churner & (days_left <= 45)
    return intensity, lo, hi, near_churn, fade


def draw_times(counts, lo, hi, rng):
    """Expand a users × weeks count grid into (user_idx, timestamp) rows."""
    u, w = np.nonzero(counts)
    reps = counts[u, w]
    u, w = np.repeat(u, reps), np.repeat(w, reps)
    start, stop = lo[u, w], hi[u, w]
    span = (stop - start).astype("timedelta64[s]").astype(float)
    ts = start + (rng.random(len(u)) * span).astype("timedelta64[s]")
    return u, w, ts


def make_events(subs, churn, engagement, friction, region, devices, intensity, lo, hi, near_churn, rng):
    scale = TARGET_EVENTS * 0.96 / intensity.sum()
    counts = rng.poisson(intensity * scale)
    u, w, ts = draw_times(counts, lo, hi, rng)

    probs = np.tile(EVENT_BASE_P, (len(u), 1))
    nc = near_churn[u, w]
    probs[:, 5] *= np.exp(1.0 * friction[u]) * np.where(nc, 4.0, 1.0)  # support_message
    probs[:, 7] *= np.where(nc, 5.0, 1.0)  # downgrade
    probs[:, 6] *= np.exp(0.4 * engagement[u])  # upgrade
    probs[:, 4] *= (subs["plan_type"].to_numpy()[u] != "free")  # free plans do not pay
    probs /= probs.sum(axis=1, keepdims=True)
    kind = (probs.cumsum(axis=1) > rng.random(len(u))[:, None]).argmax(axis=1)
    etype = np.array(EVENT_TYPES)[kind]

    ids = subs["user_id"].to_numpy()
    frames = [pd.DataFrame({"user_id": ids[u], "event_type": etype, "timestamp": ts})]

    # Lifecycle markers, on the day they happened, when the log was running.
    signed = subs["signup_date"] >= LOG_START
    frames.append(pd.DataFrame({"user_id": ids[signed], "event_type": "signup",
                                "timestamp": subs.loc[signed, "signup_date"] + pd.to_timedelta(
                                    rng.integers(8 * 3600, 18 * 3600, signed.sum()), unit="s")}))
    left = churn.notna() & (churn >= LOG_START)
    frames.append(pd.DataFrame({"user_id": ids[left], "event_type": "cancel",
                                "timestamp": churn[left] + pd.to_timedelta(
                                    rng.integers(8 * 3600, 18 * 3600, left.sum()), unit="s")}))
    events = pd.concat(frames, ignore_index=True)

    idx = pd.Series(np.arange(len(subs)), index=ids).loc[events["user_id"]].to_numpy()
    travel = rng.random(len(events)) < 0.03
    reg = np.where(travel, rng.choice(REGIONS, len(events), p=REGION_P), region[idx])
    second = rng.random(len(events)) < 0.2
    dev = np.where(second, devices[idx, 1], devices[idx, 0])

    session = np.round(rng.lognormal(5.0 + 0.25 * engagement[idx], 0.9)).clip(1, 3600)
    no_session = events["event_type"].isin(["payment", "signup", "cancel", "upgrade", "downgrade"]).to_numpy()
    session = np.where(no_session, 0, session)
    events["device"] = dev
    events["region"] = reg
    events["session_duration"] = pd.array(session.astype(int), dtype="Int64")
    events.loc[rng.random(len(events)) < 0.18, "session_duration"] = pd.NA  # client did not report it

    events = events.sort_values("timestamp", kind="stable").reset_index(drop=True)
    events.insert(0, "event_id", [f"evt_{i:07d}" for i in range(1, len(events) + 1)])
    events["timestamp"] = events["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return events[["event_id", "user_id", "event_type", "timestamp", "device", "region", "session_duration"]]


def make_usage(subs, engagement, intensity, fade, lo, hi, rng):
    scale = TARGET_USAGE_ROWS * 1.08 / intensity.sum()
    counts = rng.poisson(intensity * scale)
    u, w, ts = draw_times(counts, lo, hi, rng)

    # Engaged customers adopt more of the product.
    n_adopted = np.clip(np.round(2 + 1.6 * engagement + rng.normal(0, 1, len(subs))), 1, len(FEATURES)).astype(int)
    order = np.argsort(rng.random((len(subs), len(FEATURES))) / np.array(FEATURE_P), axis=1)
    slot = (rng.random(len(u)) * n_adopted[u]).astype(int)
    feat = np.array(FEATURES)[order[u, slot]]

    usage = pd.DataFrame({
        "user_id": subs["user_id"].to_numpy()[u],
        "feature_name": feat,
        "usage_count": 1 + rng.poisson(np.exp(1.2 + 0.8 * engagement[u]) * fade[u, w]),
        "session_seconds": np.clip(rng.normal(120 + 25 * engagement[u], 55), 10, 400),
        "date": pd.to_datetime(ts).normalize(),
    })
    usage = usage.groupby(["user_id", "feature_name", "date"], as_index=False, sort=False).agg(
        usage_count=("usage_count", "sum"), avg_session_seconds=("session_seconds", "mean")
    )
    usage["avg_session_seconds"] = usage["avg_session_seconds"].round().astype(int)
    usage["date"] = usage["date"].dt.strftime("%Y-%m-%d")
    usage = usage.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return usage[["user_id", "feature_name", "usage_count", "avg_session_seconds", "date"]]


FEEDBACK_TEMPLATES = {
    "praise": [
        "Great update — the {f} is much faster now!",
        "Love the new {f}. Saves my team an hour a week.",
        "The {f} redesign is excellent, thank you.",
        "Support sorted my question in minutes. Impressed.",
        "{F} just works. Best tool we pay for.",
        "Really happy with how {f} handles our volume.",
    ],
    "bug": [
        "The app crashes when I upload a file larger than {n}MB.",
        "{F} shows a blank page since yesterday's release.",
        "Getting a 500 error every time I open {f}.",
        "{F} times out when the date range is over {n} days.",
        "Numbers in {f} don't match the CSV export.",
        "Login loops back to the sign-in page on {d}.",
    ],
    "billing": [
        "I was charged twice for my subscription this month.",
        "Why did my invoice go up by ${n} without notice?",
        "Can't update my card — the billing page errors out.",
        "Refund for the duplicate charge still hasn't arrived.",
        "We downgraded but were billed at the old rate.",
        "The price is hard to justify for how little we use {f}.",
    ],
    "feature_request": [
        "Would love an API to export user timelines.",
        "Please add scheduled delivery for {f}.",
        "Any plans for SSO on the {p} plan?",
        "It would help if {f} supported custom fields.",
        "Could {f} get a dark mode?",
        "We need webhooks when {f} finishes.",
    ],
    "other": [
        "Not sure how to enable the new feature.",
        "Where do I find the {f} settings?",
        "Is there a way to invite a teammate on {d}?",
        "Just checking whether {f} is included in our plan.",
        "Who should I contact about a data processing agreement?",
        "Trying {f} for the first time — any tutorial?",
    ],
}
CATEGORY_SENTIMENT = {"praise": 0.75, "bug": -0.65, "billing": -0.55, "feature_request": 0.25, "other": 0.0}


def make_feedback(subs, churn, engagement, friction, events, rng):
    active_events = events[events["event_type"].isin(["login", "feature_use", "page_view", "support_message"])]
    pick = active_events.sample(N_FEEDBACK, random_state=SEED, weights=None).reset_index(drop=True)
    idx = pd.Series(np.arange(len(subs)), index=subs["user_id"]).loc[pick["user_id"]].to_numpy()
    days_left = (churn.iloc[idx].reset_index(drop=True) - pd.to_datetime(pick["timestamp"])).dt.days
    leaving = (days_left <= 60).fillna(False).to_numpy(dtype=bool)  # never-churned: NaN → not leaving

    cats = list(FEEDBACK_TEMPLATES)
    logits = np.zeros((len(pick), len(cats)))
    logits[:, cats.index("praise")] = 0.5 * engagement[idx] - 1.2 * leaving
    logits[:, cats.index("bug")] = 0.5 * friction[idx] + 0.6 * leaving
    logits[:, cats.index("billing")] = 0.4 * friction[idx] + 1.0 * leaving
    p = np.exp(logits); p /= p.sum(axis=1, keepdims=True)
    cat_idx = (p.cumsum(axis=1) > rng.random(len(pick))[:, None]).argmax(axis=1)

    rows = []
    for i, ci in enumerate(cat_idx):
        cat = cats[ci]
        f = FEATURES[rng.choice(len(FEATURES), p=FEATURE_P)].replace("_", " ")
        text = rng.choice(FEEDBACK_TEMPLATES[cat]).format(
            f=f, F=f.capitalize(), n=int(rng.choice([5, 10, 25, 30, 90])),
            d=str(rng.choice(["Safari", "Android", "iOS", "Firefox"])), p=str(subs["plan_type"].iat[idx[i]]),
        )
        score = float(np.clip(CATEGORY_SENTIMENT[cat] + rng.normal(0, 0.15), -1, 1))
        rows.append({"user_id": pick["user_id"].iat[i], "created_at": pick["timestamp"].iat[i][:10],
                     "category": cat, "sentiment_score": round(score, 2), "feedback_text": text})
    rows.sort(key=lambda r: r["created_at"])
    return rows


def main() -> None:
    rng = np.random.default_rng(SEED)
    subs = pd.read_csv(DATA / "subscriptions.csv")
    subs = subs[["user_id", "plan_type", "mrr"]].copy()
    n = len(subs)
    subs["signup_date"] = draw_signups(n, rng).to_numpy()
    engagement = rng.normal(0, 1, n) + subs["plan_type"].map({"free": -0.3, "starter": 0, "pro": 0.2,
                                                              "enterprise": 0.3}).to_numpy()
    friction = rng.normal(0, 1, n)
    region = rng.choice(REGIONS, n, p=REGION_P)
    first = rng.choice(len(DEVICES), n, p=DEVICE_P)
    second = (first + rng.integers(1, len(DEVICES), n)) % len(DEVICES)
    devices = np.array(DEVICES)[np.stack([first, second], axis=1)]

    churn = simulate_churn(subs, rng, engagement, friction, region)
    subs["churn_date"] = churn.dt.strftime("%Y-%m-%d")
    subs["is_churned"] = churn.notna().astype(int)
    subs["tenure_days"] = (churn.fillna(OBSERVATION_END) - subs["signup_date"]).dt.days.clip(lower=0)
    subs["signup_date"] = subs["signup_date"].dt.strftime("%Y-%m-%d")
    subs_dates = subs.assign(signup_date=pd.to_datetime(subs["signup_date"]))

    intensity, lo, hi, near_churn, fade = activity_grid(subs_dates, churn, engagement, rng)
    events = make_events(subs_dates, churn, engagement, friction, region, devices,
                         intensity, lo, hi, near_churn, rng)
    usage = make_usage(subs_dates, engagement, intensity, fade, lo, hi, rng)
    feedback = make_feedback(subs_dates, churn, engagement, friction, events, rng)

    subs.to_csv(DATA / "subscriptions.csv", index=False)
    events.to_csv(DATA / "user_events.csv", index=False)
    usage.to_csv(DATA / "feature_usage.csv", index=False)
    with (DATA / "feedback.json").open("w") as fh:
        for row in feedback:
            fh.write(json.dumps(row) + "\n")

    print(f"subscriptions {len(subs):,}  lifetime churn {subs['is_churned'].mean():.3f}")
    print(f"user_events   {len(events):,}")
    print(f"feature_usage {len(usage):,}")
    print(f"feedback      {len(feedback):,}")


if __name__ == "__main__":
    main()
