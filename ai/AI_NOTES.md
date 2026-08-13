# AI Notes

I used AI mainly to help me understand errors, check the assignment requirements,
and review whether my implementation was working as expected.

I did not assume that every AI suggestion was correct. During the project, I
checked the code by running it locally, looking at the generated CSV files and
figures, and comparing the results with `PROJECT_BRIEF.md`.

There were several cases where I had to correct or further check the AI output.
For example, the news merge still failed until I checked the actual datetime
types and converted both merge keys to `datetime64[ns]`. I also changed from the
NLTK VADER setup to the standalone `vaderSentiment` package after the lexicon
download failed.

For the portfolio work, I checked that the backtest used only past data and I
compared the latest minimum-variance and maximum-Sharpe weights to make sure the
two optimisation methods were producing different portfolios.

I also tested the Streamlit app both locally and after deployment. One public-app
error came from an older cached version of `fund_returns.csv`, so I checked the
actual CSV columns, removed the result-loading cache, and deployed the corrected
version again.

For the sentiment shock extension, I kept the negative result instead of changing
the rule repeatedly until the performance improved. I treated the result as
evidence that the additional signal did not add value in this sample.

I ran the code myself, checked the outputs, and made the final decisions about
which changes to keep.
