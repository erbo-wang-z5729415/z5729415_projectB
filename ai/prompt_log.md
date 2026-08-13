# Prompt Log

This file records the main tasks where I used AI during Project B. I mainly used
AI for debugging, checking assignment requirements, and reviewing whether my
implementation was working as expected.

---

## Task 1 - Portfolio and OOS Backtest

### What I wanted

I wanted to build the required combined equity and cryptocurrency funds and make
sure the backtest did not use future information.

I used minimum variance and maximum Sharpe as the two portfolio methods.

### Prompt(s)

Some of the prompts I used were:

> “这个 portfolios.py 是老师给的模板，我应该怎么完成 minimum variance 和 maximum Sharpe？”

> “walk-forward out-of-sample 应该怎么做，怎样确保没有 look-ahead？”

> “这个回测结果正常吗？两个方法的权重是不是确实不同？”

### What the assistant produced

The assistant explained how to use a rolling estimation window and how to
rebalance using only past returns.

It suggested a 252-trading-day estimation window, rebalancing every 21 trading
days, and long-only weights that sum to one.

### What was wrong or risky

The main risk was look-ahead bias. If the current day's return was included when
forming the current day's weights, the backtest would not have been valid.

There was also a risk that the optimiser could appear to run without producing
meaningfully different portfolios.

### What I changed and why

I checked that each estimation window ended before the return being tested.

I also compared the final holdings from the two methods. The minimum-variance
portfolio was more diversified, while the maximum-Sharpe portfolio was much more
concentrated in assets such as GE, NVDA and BTC.

This confirmed that the two optimisation methods were producing different
portfolio weights.

---

## Task 2 - Sentiment and Debugging

### What I wanted

I wanted to build the sector sentiment index and fix the errors that appeared
when aligning the news data and running VADER.

### Prompt(s)

Some of the prompts I used were:

> “为什么 merge_asof 会出现 datetime64[us] 和 datetime64[s] 不一致的错误？”

> “VADER 为什么提示 vader_lexicon not found？”

> “NLTK 下载 vader_lexicon 一直失败，应该怎么办？”

### What the assistant produced

The assistant explained that the two datetime columns used for `merge_asof`
needed to have exactly the same dtype.

It also suggested using the standalone `vaderSentiment` package after the NLTK
lexicon download failed because of a proxy or security restriction.

### What was wrong or risky

Using `pd.to_datetime()` alone did not fully solve the merge problem because the
two columns still had different datetime resolutions.

There was also a reproducibility risk if the project depended on manually
downloading the VADER lexicon before every build.

### What I changed and why

I converted both merge keys to `datetime64[ns]` before using `merge_asof`.

For the sentiment model, I used `vaderSentiment`, which includes the VADER
lexicon with the package, instead of relying on the separate NLTK download.

After making these changes, I reran the Part B workflow and confirmed that the
sector sentiment index was created successfully.

---

## Task 3 - Streamlit App and Deployment

### What I wanted

I wanted the Streamlit app to show the fund comparison, fund fact sheets,
allocation controls and sentiment analytics using the precomputed result files.

I also wanted to deploy the app online and check that the public version worked
the same way as the local version.

### Prompt(s)

Some of the prompts I used were:

> “这个 App 应该怎样满足老师要求的 investor journey？”

> “为什么本地下拉框暂时不能点击？”

> “怎么把本地 Project B 推到 GitHub 并部署到 Streamlit？”

> “公网 Streamlit 为什么出现 KeyError，但本地 CSV 里面明明有这一列？”

### What the assistant produced

The assistant suggested organising the app into three main sections: Funds,
Allocation and Sentiment.

It also helped me check the fund selector, allocation slider, current holdings,
sector selector and public deployment.

For the deployed `KeyError`, the assistant suggested checking the actual columns
inside `fund_returns.csv` and then checking whether Streamlit was still using an
older cached version of the file.

### What was wrong or risky

The deployed app initially failed after I added the sentiment shock extension.
The local CSV already contained the new columns, but the online app was still
using older cached result data.

There was also a risk of running the portfolio backtest or VADER inside the
deployed app. This would make the app slower and would not follow the project
brief.

### What I changed and why

I kept the portfolio and sentiment calculations inside `run_part_b.py` and made
the Streamlit app read the precomputed CSV files from `results/`.

I checked the `fund_returns.csv` header directly and confirmed that the new
sentiment shock columns were present.

I then removed the result-loading cache that was keeping the older data and
pushed the correction to GitHub.

After redeployment, I tested the public app again. I checked the fund selector,
the 0% and 100% allocation cases, multiple sector selections, and the empty
sector-selection case.
---

## Task 4 - Sentiment Shock Extension

### What I wanted

I wanted to test whether relative changes in sector sentiment could be more
useful than using the absolute sentiment level alone.

My idea was to compare the latest lagged sector sentiment with its recent
history and use unusually high or low values as a separate signal.

### Prompt(s)

Some of the prompts I used were:

> “我想测试行业情绪相对于过去一段时间的异常变化，而不是只看绝对 sentiment，这个思路在 Part B 里是否合理？”

> “如果用过去 20 个交易日作为参考，sentiment shock 应该怎样计算才能避免 look-ahead？”

> “这个 sentiment shock 的结果和 basic sentiment tilt 相比怎么样？”

### What the assistant produced

The assistant helped me turn the idea into a simple testable rule.

It explained how to compare lagged sector sentiment with its recent rolling
mean and standard deviation, and how to make sure the calculation only used
information available before the trading decision.

It also helped me compare the shock-based strategy with the base portfolio and
the basic sentiment tilt.

### What was wrong or risky

A large sentiment shock does not necessarily mean that the signal has useful
predictive information.

There was also a risk of overfitting if I kept changing the lookback window or
tilt strength after seeing the results.

Another issue was that the signal was calculated at sector level, so it could
lose information about differences between individual stocks in the same
sector.

### What I changed and why

I kept the rule simple and used a 20-day rolling reference window.

I also kept the tilt strength fixed rather than repeatedly adjusting it after
seeing the performance results.

The extension did not improve the portfolios. For the maximum-Sharpe fund,
annualised return fell from about 29.48% for the base portfolio to about 28.75%
with the sentiment shock, while the Sharpe ratio fell from about 1.05 to about
1.02.

I kept the result because the purpose of the extension was to test whether the
relative sentiment signal added useful information. The negative result was
still useful because it showed that the additional signal did not improve the
portfolio in this sample.
