# SignalHarbour - FINS5545 Project B

SignalHarbour is a multi-asset investment dashboard built for Project B. It
compares Minimum Variance and Maximum Sharpe portfolios using equities and
cryptocurrencies, with a walk-forward out-of-sample backtest.

The project also includes a VADER sector news sentiment index, a basic sentiment
tilt, and a Sentiment Shock extension. The sentiment strategies did not improve
performance in this sample, so the negative results are retained and discussed
in the report.

## Public Links

Live Streamlit app:
https://z5729415projectb-wmusfsgjcu23zsmkttophv.streamlit.app/

GitHub repository:
https://github.com/erbo-wang-z5729415/z5729415_projectB

## How to Run

```bash
pip install -r requirements.txt -r requirements-dev.txt
python scripts/run_part_b.py
streamlit run streamlit_app.py
