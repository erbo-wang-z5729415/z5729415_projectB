# AGENTS.md

This file records how I want my AI assistant to help me with FINS5545 Project B.

## My project

I am building two combined equity and cryptocurrency funds using minimum
variance and maximum Sharpe methods. The project also includes a sector news
sentiment index, a simple sentiment adjustment, and a Streamlit app.

I want the assistant to read `PROJECT_BRIEF.md` and the files in `context/`
before suggesting important changes. If a suggestion does not match the project
brief, I will follow the project brief.

## How I use AI

I mainly use AI when I need help understanding Python code, debugging an error,
checking the assignment requirements, or reviewing the structure of my work.

I prefer explanations in simple language. When an error occurs, I want the
assistant to explain the likely cause before suggesting a change. I also prefer
small changes where possible rather than rewriting files unnecessarily.

## Rules for this project

- Keep all work inside `z5729415_projectB`.
- Do not edit `src/data_access.py` or the files in `context/`.
- Do not save or submit the raw Parquet data.
- Keep the teacher's original comments and file structure where possible.
- Do not say that code works until it has been run and checked.
- Use the exact output filenames required in `PROJECT_BRIEF.md`.

For the portfolio work:

- calculate returns separately for each ticker before combining the data;
- align the combined fund to the equity trading calendar;
- use only past data when estimating weights;
- use a 252-trading-day estimation window and rebalance every 21 trading days;
- keep the portfolios long-only and make sure the weights add up to one;
- use 252 periods per year for the combined funds;
- check that the minimum-variance and maximum-Sharpe methods produce different
  weights.

For the sentiment work:

- keep the original headline casing, punctuation and negation;
- score the headlines before calculating ticker-day and sector-day averages;
- equal-weight tickers when calculating the sector index;
- lag the sentiment signal by at least one trading day;
- apply the sentiment adjustment only to equities;
- report the result honestly even if sentiment does not improve performance.

For the Streamlit app:

- read the precomputed files from `results/`;
- do not rerun VADER or the portfolio backtest inside the deployed app;
- keep the app simple enough to run on Streamlit Community Cloud.

## How I check AI suggestions

After an important change, I normally:

1. run a Python syntax check;
2. run `python scripts/run_part_b.py`;
3. inspect the generated tables and figures;
4. test the Streamlit app locally;
5. run `python scripts/check_handin.py`.

I run the code myself and decide which suggestions to keep. Important prompts,
errors and corrections are recorded in the `ai/` folder.
