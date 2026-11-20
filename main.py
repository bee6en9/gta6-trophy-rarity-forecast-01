"""gta6-trophy-rarity-forecast-01: structural prior-driven forecast of each
Grand Theft Auto VI PlayStation 5 trophy's Exophase earn rate at the case
target, as five quantiles (0.05/0.25/0.5/0.75/0.95) on the logit scale.

Model (all parameters chosen by grid search on the validation harness in
/tmp/trophy-forecast-scratch, not by feel):

  * The house history is FLAT: every read before the checkpoint shows the
    same 638 players and the same rates (launch night, Exophase had not
    refreshed yet). No per-trophy trend can be estimated from it, so the
    forecast is prior-driven and structural. The only real signal in the
    data is the refresh the practice answers contain: one refresh took
    players 638 -> 3,868 and every rate but the 100% one down, i.e. a
    logit shift of about -1.4 to -1.8.
  * Median, in logit space, for trophy i with last rate r:
        logit(m) = logit(r) + drift_i * v - jump / (1 + v)
    where v = ln(1 + t_target) - ln(1 + t_last), t measured in hours since
    the release (2026-11-19T00:00:00Z). drift_i is a per-trophy log-time
    growth rate: a tier base (bronze 2.2, silver 1.2, gold 0.6, platinum
    0.0) times a rate-band factor (r >= 1.5% -> 1.2, 0.9-1.5% -> 1.0,
    r < 0.9% -> 0.8) times a position factor (id <= 13, the early story
    trophies, -> 1.3; id <= 30 -> 1.0; later, side-content trophies ->
    0.7), all scaled by G = 0.7, minus a global dilution D = 0.4, then
    shrunk halfway toward the tier mean (shrink = 0.5): with only 4-13
    earners behind each launch-night rate, the rate differences between
    trophies are mostly noise, so the tier mean carries real signal.
    jump = 1.2 logit is the one-off dilution of the imminent next
    Exophase refresh; it is (nearly) full at short horizons and decays
    as log-time accrues, because over a week the player-growth backlog
    clears and story progression dominates. The median is capped at a
    tier saturation (bronze 65%, silver 40%, gold 15%, platinum 3%):
    early trophies rise fast then saturate, late ones still crawl.
  * Spread, in logit units: s = 0.6 + 0.5 * v (times 1.15 for trophies
    last under 1%, whose counts are thinnest), i.e. the spread grows with
    log-time, not sqrt-time: refresh shocks are jumps, not diffusion.
    The trophy at 100% (every tracked player has it) keeps a median of
    99.99% with a spread of at least 1.2, hedging a small chance it
    slips below 100%.
  * Quantiles: q_tau = sigmoid(logit(m) + z_tau * s), z the standard
    normal quantiles, rounded to 4 decimals (half up).

Validation (scorer: pinball loss on the logit scale, exactly per RULES;
see /tmp/trophy-forecast-scratch/scorer_final.py):
  * Rolling-origin backtest on the flat history: persistence 0.014 vs
    this model 0.42-0.44 -- the backtest can only reward narrow spreads,
    because nothing moves in-sample; it cannot see a refresh, so it is
    reported honestly and not used for selection.
  * Refresh-shock stress test (practice answers rebuilt by diluting last
    rates 638 -> 3,868 players; earners flat / +20% / +50%): this model
    0.74 / 0.60 / 0.47 (mean 0.60, 90% coverage 0.91) vs persistence and
    curve baselines 1.69 / 1.52 / 1.29 (mean 1.50, coverage 0.00).
    Per tier (stress mean): platinum 0.34, gold 0.43, silver 0.46,
    bronze 0.63 -- every tier beats the baselines (~1.5).
  * Final-horizon sanity (synthetic case, last read Nov 21 11:30 UTC,
    target Nov 26 00:00 UTC): median rates early story 2.07%,
    late bronze 1.10%, silver 0.45%, gold 0.35%, platinum 0.23% --
    early trophies rising toward saturation, late ones low but rising.
    Against three refined final worlds (progression / dilution / flat)
    the model scores 0.62 / 0.46 / 0.32, worst case 0.62; the baselines
    score 0.72 / 0.34 / 0.21 on the same worlds but 1.50 on the stress
    test. The model was selected for the best worst case across all
    four worlds, not for any single one.

Deterministic (no randomness), standard library only, no network.
Reads $MAKEFUN_INPUT_DIR (else /input), writes $MAKEFUN_OUTPUT_DIR
(else /output): one <id>.out per <id>.in, {"trophies": [{"id": n,
"q": [q05, q25, q50, q75, q95]}, ...]}, 55 rows for the practice/final
cases, each q finite in [0, 100], non-decreasing, file well under
65,536 bytes. Two runs are byte-identical.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- tuned parameters (grid search; see module docstring) ---
GROWTH_SCALE = 0.7      # G: scales the tier progression prior
DILUTION = 0.4          # D: global player-growth drag, logit per log-time
SHRINK = 0.5            # shrinkage of per-trophy drift toward its tier mean
SPREAD_A = 0.6          # spread intercept (logit)
SPREAD_B = 0.5          # spread per unit log-time
REFRESH_JUMP = 1.2      # one-off logit dilution of the imminent refresh
RARE_SPREAD = 1.15      # extra spread for trophies last under 1%
FULL_SPREAD = 1.2       # minimum spread for the trophy at 100%

TIER_GROWTH = {"bronze": 2.2, "silver": 1.2, "gold": 0.6, "platinum": 0.0}
TIER_CAP = {"bronze": 65.0, "silver": 40.0, "gold": 15.0, "platinum": 3.0}
# Fallback last rates when a case has no usable history at all.
TIER_FALLBACK_RATE = {"platinum": 0.5, "gold": 1.0, "silver": 0.75, "bronze": 1.0}


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up (as the starter rounds it)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def norm_ppf(p: float) -> float:
    """Standard normal quantile (Acklam's rational approximation, |err| < 1.2e-9)."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def band_factor(rate: float) -> float:
    if rate >= 1.5:
        return 1.2
    if rate >= 0.9:
        return 1.0
    return 0.8


def position_factor(trophy_id: int) -> float:
    if trophy_id <= 13:
        return 1.3
    if trophy_id <= 30:
        return 1.0
    return 0.7


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def last_rates(case: dict) -> list:
    """Most recent non-null rate per trophy, falling back through history,
    then to a tier prior when a trophy was never read."""
    trophies = case["trophies"]
    rates: list = [None] * len(trophies)
    for read in reversed(case.get("history") or []):
        for i, r in enumerate(read.get("rates") or []):
            if rates[i] is None and r is not None:
                rates[i] = r
        if all(r is not None for r in rates):
            break
    for i, t in enumerate(trophies):
        if rates[i] is None:
            rates[i] = TIER_FALLBACK_RATE.get(t.get("tier"), 1.0)
    return rates


def forecast_case(case: dict) -> dict:
    trophies = case["trophies"]
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    # Clamp levels into the open unit interval: norm_ppf is undefined at
    # exactly 0 or 1, and a case must never crash the run over a level.
    z = [norm_ppf(min(1.0 - 1e-9, max(1e-9, float(tau)))) for tau in levels]

    history = case.get("history") or []
    rates = last_rates(case)
    if history:
        last_at = parse_time(history[-1]["at"])
    else:
        last_at = RELEASE
    target = parse_time(case["target"])
    t_last = max(0.0, (last_at - RELEASE).total_seconds() / 3600.0)
    t_tgt = max(t_last, (target - RELEASE).total_seconds() / 3600.0)
    v = math.log1p(t_tgt) - math.log1p(t_last)

    spread_base = SPREAD_A + SPREAD_B * v
    jump_eff = REFRESH_JUMP / (1.0 + v)

    # Per-trophy drift per unit log-time, then shrunk toward the tier mean.
    raw = {}
    for t, r in zip(trophies, rates):
        raw[t["id"]] = (TIER_GROWTH.get(t.get("tier"), 1.0) * GROWTH_SCALE
                        * band_factor(r) * position_factor(t["id"]) - DILUTION)
    tier_mean = {}
    for tier in TIER_GROWTH:
        vals = [raw[t["id"]] for t in trophies if t.get("tier") == tier]
        tier_mean[tier] = sum(vals) / len(vals) if vals else 0.0

    rows = []
    for t, r in zip(trophies, rates):
        tid = t["id"]
        tier = t.get("tier")
        if r >= 99.99:
            # Everyone tracked has this trophy; median stays ~100, but the
            # spread hedges the small chance it slips below 100%.
            med_logit = logit(99.99)
            s = max(spread_base, FULL_SPREAD)
            qs = [round4(sigmoid(med_logit + zi * s)) for zi in z]
        else:
            drift = SHRINK * tier_mean.get(tier, 0.0) + (1.0 - SHRINK) * raw[tid]
            med_logit = logit(r) + drift * v - jump_eff
            med_logit = min(med_logit, logit(TIER_CAP.get(tier, 65.0)))
            s = spread_base * (RARE_SPREAD if r < 1.0 else 1.0)
            qs = [round4(sigmoid(med_logit + zi * s)) for zi in z]
        # Guarantee the contract even under float dust: finite, 0..100,
        # non-decreasing (rounding is monotone, this is a belt-and-braces).
        qs = [min(100.0, max(0.0, float(q))) for q in qs]
        for i in range(1, len(qs)):
            qs[i] = max(qs[i], qs[i - 1])
        rows.append({"id": tid, "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
