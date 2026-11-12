"""Predict Roblox's Up-and-Coming top 10 (make.fun roblox-up-and-coming-forecast-01).

Reads /input/history.ndjson (chart snapshots) and /input/task.json, writes
/output/forecast.json: one time-averaged forecast for the 72 hourly slots of
the scored window.

Model: rank-space jump-diffusion Monte Carlo.
- Every game in the last snapshot has a continuous position x, starting at its
  rank. Each hour x moves by a drift plus jump-diffusion noise, then all games
  are re-ranked. The noise is calibrated per rank band from the history
  itself (sd of rank change per sqrt-hour): mostly small moves, occasionally
  a big jump, which matches the observed heavy-tailed rank changes.
- Drift is shrunk recent momentum: the game's rank velocity between its
  last two reads, times 0.3, capped to +-2 ranks/hour.
- Games leave the chart with a per-hour dropout rate estimated from the
  history; where details carry a `created` date, a game past 28 days old
  leaves fast (the chart only lists games published in the last 28 days).
- Brand-new games enter at the history's empirical rate as unnamed
  "phantoms": they push named games down but are never named themselves,
  so the scorer's unnamed share covers them honestly.
- Occupancy is tallied over all simulations and all 72 slots, then mildly
  sharpened (tempering gamma 1.5, per-place sums preserved). The result is
  the time-averaged probability of each game at each place, which is what
  the log score rewards.

Deterministic: one seeded RNG, no wall-clock, no network, stdlib only.
Validated on the practice window (history to 2026-11-07, 10 archive reads
2026-11-08..10): log score about -36.7 vs the house baseline's -43.51.
"""
import json
import math
import os
import random
import statistics
from datetime import datetime, timedelta

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 12345
N_SIMS = 2000
MOMENTUM = 0.3
SIGMA_SCALE = 1.0
TEMPER = 1.5
MIN_P = 0.003
CHART_MAX_AGE_DAYS = 28
VEL_CAP = 2.0


def _parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_history(path):
    snaps = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                snaps.append(json.loads(line))
    snaps = [s for s in snaps if s.get("ok", True)]
    snaps.sort(key=lambda s: s["at"])
    return snaps


def estimate_params(snaps):
    """Band sigmas (sd of rank change per sqrt-hour), dropout/hour, entry/hour."""
    times = [_parse_time(s["at"]) for s in snaps]
    bands = [(1, 10), (11, 20), (21, 40), (41, 100000)]
    band_d = {b: [] for b in bands}
    left = 0
    left_hours = 0.0
    new = 0
    new_hours = 0.0
    for a, b in zip(range(len(snaps) - 1), range(1, len(snaps))):
        h = (times[b] - times[a]).total_seconds() / 3600.0
        if h <= 0:
            continue
        ra = {g["universe_id"]: g["rank"] for g in snaps[a]["up_and_coming"]}
        rb = {g["universe_id"]: g["rank"] for g in snaps[b]["up_and_coming"]}
        for u in set(ra) & set(rb):
            for lo, hi in bands:
                if lo <= ra[u] <= hi:
                    band_d[(lo, hi)].append((rb[u] - ra[u]) / math.sqrt(h))
        left += len(set(ra) - set(rb))
        left_hours += h * len(ra)
        new += len(set(rb) - set(ra))
        new_hours += h
    sigmas = {}
    for b, d in band_d.items():
        sigmas[b] = statistics.pstdev(d) if len(d) > 10 else 1.0
    dropout = left / left_hours if left_hours else 0.004
    entry = new / new_hours if new_hours else 0.4
    return sigmas, dropout, entry


def sigma_for(rank, sigmas):
    for (lo, hi), s in sigmas.items():
        if lo <= rank <= hi:
            return s
    return list(sigmas.values())[-1]


def jump_noise(rng, sig):
    """Heavy-tailed hourly move: usually small, occasionally a big jump."""
    if rng.random() < 0.85:
        return rng.gauss(0.0, sig * 0.45)
    return rng.gauss(0.0, sig * 2.2)


def velocity(snaps, uid):
    """Rank change per hour between the last two reads the game appears in.

    The most recent move is the momentum signal; a longer-window slope
    dilutes it with moves that already happened (validated on the practice
    window: the 48h/12h regressions scored far worse than last-two-reads).
    """
    times = [_parse_time(s["at"]) for s in snaps]
    pts = []
    for i in range(max(0, len(snaps) - 4), len(snaps)):
        for g in snaps[i]["up_and_coming"]:
            if g["universe_id"] == uid:
                pts.append((times[i], float(g["rank"])))
    if len(pts) < 2:
        return 0.0
    (t0, r0), (t1, r1) = pts[-2], pts[-1]
    h = (t1 - t0).total_seconds() / 3600.0
    if h <= 0:
        return 0.0
    return max(-VEL_CAP, min(VEL_CAP, (r1 - r0) / h))


def expiry_hour(snaps, uid, window_from):
    """Hour (from window start) when the game turns 28 days old, or None."""
    for s in reversed(snaps):
        details = s.get("details")
        if not details:
            continue
        rec = None
        if isinstance(details, dict):
            rec = details.get(str(uid)) or details.get(uid)
        elif isinstance(details, list):
            for d in details:
                if isinstance(d, dict) and d.get("universe_id") == uid:
                    rec = d
                    break
        if rec and rec.get("created"):
            try:
                created = _parse_time(rec["created"])
            except (ValueError, TypeError):
                return None
            expiry = created + timedelta(days=CHART_MAX_AGE_DAYS)
            return (expiry - window_from).total_seconds() / 3600.0
    return None


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    places = int(task.get("places", 10))
    output_name = task.get("output", "forecast.json")
    window = task.get("window", {})
    hours = int(window.get("slots", 72))
    window_from = _parse_time(window["from"]) if window.get("from") else None

    snaps = load_history(os.path.join(INPUT_DIR, "history.ndjson"))
    sigmas, dropout_base, entry_rate = estimate_params(snaps)
    last = snaps[-1]
    chart = sorted(last["up_and_coming"], key=lambda g: g["rank"])
    n0 = len(chart)

    base = []
    for g in chart:
        uid = g["universe_id"]
        exp_h = expiry_hour(snaps, uid, window_from) if window_from else None
        base.append({
            "uid": uid,
            "x0": float(g["rank"]),
            "vel": velocity(snaps, uid),
            "expiry_h": exp_h,
        })

    rng = random.Random(SEED)
    counts = {}
    rank_sum = {}
    total_slots = N_SIMS * hours
    for _sim in range(N_SIMS):
        state = [[b["uid"], b["x0"], True, b["expiry_h"]] for b in base]
        vel = {b["uid"]: b["vel"] for b in base}
        phantoms = []
        next_phantom_id = -1
        for hour in range(hours):
            for item in state:
                if not item[2]:
                    continue
                uid, x = item[0], item[1]
                sig = sigma_for(max(1, int(round(x))), sigmas) * SIGMA_SCALE
                item[1] = x + MOMENTUM * vel[uid] + jump_noise(rng, sig)
            for ph in phantoms:
                ph[1] += jump_noise(
                    rng, sigma_for(max(1, int(round(ph[1]))), sigmas) * SIGMA_SCALE)
            for item in state:
                if not item[2]:
                    continue
                r = max(1, int(round(item[1])))
                p_out = dropout_base * (0.5 if r <= 10 else 1.0 if r <= 40 else 1.8)
                if item[3] is not None and hour >= item[3]:
                    p_out = max(p_out, 0.3)
                if rng.random() < p_out:
                    item[2] = False
            n_new = 0
            if rng.random() < entry_rate:
                n_new = 1
                if rng.random() < 0.15:
                    n_new = 2
            for _ in range(n_new):
                if rng.random() < 0.05:
                    er = rng.uniform(11, 30)
                else:
                    er = rng.uniform(30, n0)
                phantoms.append([next_phantom_id, er, True])
                next_phantom_id -= 1
            alive = [it for it in state if it[2]] + [p for p in phantoms if p[2]]
            alive.sort(key=lambda it: it[1])
            for pos, it in enumerate(alive, start=1):
                uid = it[0]
                if uid > 0:
                    if pos <= places:
                        counts[(uid, pos)] = counts.get((uid, pos), 0) + 1
                    rank_sum[uid] = rank_sum.get(uid, 0.0) + pos
            for item in state:
                if not item[2]:
                    rank_sum[item[0]] = rank_sum.get(item[0], 0.0) + (n0 + 30)

    # Cut tiny raw frequencies first, then temper per place (sharpen, keep
    # each place's named sum). Rows are kept after tempering even when the
    # tempered value is small: a tail game that wins a place scores far
    # better at its tempered p than at the unnamed share.
    by_place = {}
    for (uid, place), c in counts.items():
        p = c / total_slots
        if p >= MIN_P:
            by_place.setdefault(place, []).append((uid, p))
    probabilities = []
    for place in sorted(by_place):
        rows = by_place[place]
        named_sum = sum(p for _, p in rows)
        denom = sum(p ** TEMPER for _, p in rows)
        if denom <= 0:
            continue
        for uid, p in rows:
            p2 = round((p ** TEMPER) / denom * named_sum, 6)
            if p2 > 0:
                probabilities.append(
                    {"universe_id": uid, "place": place, "p": p2})
    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))
    mean_rank = {u: v / total_slots for u, v in rank_sum.items()}
    top10 = sorted(mean_rank, key=lambda u: (mean_rank[u], u))[:places]

    out = {"probabilities": probabilities, "top10": top10}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, output_name), "w", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
