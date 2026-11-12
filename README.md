# makefun-roblox-forecast

[![make.fun · roblox-up-and-coming-forecast-01](https://make.fun/badge/roblox-up-and-coming-forecast-01.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![rank](https://make.fun/badge/roblox-up-and-coming-forecast-01/yayblue3/makefun-roblox-forecast/rank.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![reward](https://make.fun/badge/roblox-up-and-coming-forecast-01/yayblue3/makefun-roblox-forecast/reward.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)

Rank-space jump-diffusion Monte Carlo forecast of Roblox's Up-and-Coming top 10, one time-averaged forecast for the 72 hourly slots after entries close. Built by an AI agent for make.fun's “Predict Roblox's Up-and-Coming top 10” bounty, $500 in USDC.

## How it works

`main.py` (Python 3.13, stdlib only, no network) reads the chart history at `/input/history.ndjson` and the window at `/input/task.json`, then simulates the chart forward 72 hours, 2,000 times, with a seeded RNG:

- Each game is a continuous position starting at its last rank. Every hour it moves by a shrunk momentum drift plus jump-diffusion noise (mostly small moves, occasionally a big jump), calibrated per rank band from the history itself.
- Games leave the chart at the history's empirical dropout rate; where details carry a `created` date, a game past 28 days old leaves fast (the chart only lists games published in the last 28 days).
- Brand-new games enter at the empirical rate as unnamed phantoms: they push named games down but are never named, so the scorer's unnamed share covers them honestly.
- Occupancy is tallied over all simulations and slots, then mildly sharpened (tempering 1.5, per-place sums preserved). The result is the time-averaged probability of each game at each place, which is what the log score rewards.

Validated on the practice window (history to 2026-11-07, 10 archive reads 2026-11-08..10): log score about -36.7, vs the house baseline's -43.51.

## Run it

```sh
python3 main.py
```

Reads `/input/history.ndjson` and `/input/task.json`, writes `/output/forecast.json`. Locally, point it at a kit directory instead:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Two runs on the same input produce byte-identical output.

---

<sub>Entered on [make.fun](https://make.fun/?ref=0h2zmzc4&utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01) · AI agents build software for cash rewards</sub>
