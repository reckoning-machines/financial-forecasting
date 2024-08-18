A repo to facilitate data pulls from https://site.financialmodelingprep.com/ and https://fred.stlouisfed.org/

Initial pulls are shown in the example notebook.

Fred api key and FMP api key must be placed in a keys.toml file in the root.

To do: 

1. Forecast earnings vs the Street

2. establish daily returns as the 'target' for the machine learning module.

3. Rolling join (using functools) module to create the final ML dataset.

4. ML modeling, output, and daily roll foward backtest.