# 04 - Continuous OptBinning for Amount + Explained Variance

This notebook applies continuous optimal binning against deal amount and reports:
- binning-based IV-like score from continuous tables
- explained variance from bin-mean predictions
- ANOVA F-statistic over bin groups


```python
import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from sklearn.metrics import explained_variance_score
from optbinning import ContinuousOptimalBinning

pd.set_option('display.max_columns', 200)
```


```python
df = pd.read_excel('../cars.xlsx')
df['target_amount'] = df['Opportunity Amount USD'].astype(float)
df['target_log_amount'] = np.log1p(df['target_amount'])
print('shape:', df.shape)
print(df[['target_amount', 'target_log_amount']].describe())
```

    shape: (78025, 21)
            target_amount  target_log_amount
    count    78025.000000       78025.000000
    mean     91637.260750          10.314292
    std     133161.029156           2.291696
    min          0.000000           0.000000
    25%      15000.000000           9.615872
    50%      49000.000000          10.799596
    75%     105099.000000          11.562668
    max    1000000.000000          13.815512


## Fit continuous optimal binning and compute explained variance


```python
excluded = {'Opportunity Amount USD', 'target_amount', 'target_log_amount', 'Opportunity Number'}
features = [c for c in df.columns if c not in excluded]

rows = []
failed = []
y = df['target_log_amount']
for col in features:
    x = df[col]
    dtype = 'numerical' if pd.api.types.is_numeric_dtype(x) else 'categorical'

    try:
        cob = ContinuousOptimalBinning(name=col, dtype=dtype)
        cob.fit(x, y)

        bt = cob.binning_table.build()
        iv_total = float(bt.iloc[-1]['IV']) if 'IV' in bt.columns else np.nan

        idx = cob.transform(x, metric='indices')
        work = pd.DataFrame({'idx': idx, 'y': y})
        bin_mean = work.groupby('idx', dropna=False)['y'].mean()
        yhat = work['idx'].map(bin_mean).astype(float).to_numpy()
        ev = float(explained_variance_score(y, yhat))

        groups = [g['y'].values for _, g in work.groupby('idx') if len(g) > 1]
        if len(groups) >= 2:
            f_stat = float(f_oneway(*groups).statistic)
        else:
            f_stat = np.nan

        rows.append({
            'variable': col,
            'dtype': dtype,
            'status': cob.status,
            'iv_continuous': iv_total,
            'explained_variance': ev,
            'anova_f': f_stat
        })
    except Exception as e:
        failed.append({'variable': col, 'error': str(e)[:200]})

amount_bin_df = pd.DataFrame(rows)
if not amount_bin_df.empty:
    amount_bin_df = amount_bin_df.sort_values('explained_variance', ascending=False).reset_index(drop=True)
failed_df = pd.DataFrame(failed)
print('processed:', len(amount_bin_df), 'failed:', len(failed_df))
amount_bin_df.head(20)
```

    processed: 17 failed: 0





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>dtype</th>
      <th>status</th>
      <th>iv_continuous</th>
      <th>explained_variance</th>
      <th>anova_f</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Deal Size Category (USD)</td>
      <td>categorical</td>
      <td>OPTIMAL</td>
      <td>1.444898</td>
      <td>0.673988</td>
      <td>32258.820970</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Total Days Identified Through Qualified</td>
      <td>numerical</td>
      <td>OPTIMAL</td>
      <td>0.266347</td>
      <td>0.021850</td>
      <td>174.271322</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Total Days Identified Through Closing</td>
      <td>numerical</td>
      <td>OPTIMAL</td>
      <td>0.262242</td>
      <td>0.020670</td>
      <td>164.655558</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Route To Market</td>
      <td>categorical</td>
      <td>OPTIMAL</td>
      <td>0.321507</td>
      <td>0.019698</td>
      <td>1567.772906</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Supplies Subgroup</td>
      <td>categorical</td>
      <td>OPTIMAL</td>
      <td>0.250268</td>
      <td>0.018890</td>
      <td>214.586543</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Competitor Type</td>
      <td>categorical</td>
      <td>OPTIMAL</td>
      <td>0.190901</td>
      <td>0.013290</td>
      <td>1050.891894</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Opportunity Result</td>
      <td>categorical</td>
      <td>OPTIMAL</td>
      <td>0.205392</td>
      <td>0.011483</td>
      <td>906.368208</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Ratio Days Identified To Total Days</td>
      <td>numerical</td>
      <td>OPTIMAL</td>
      <td>0.205773</td>
      <td>0.010200</td>
      <td>200.996800</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ratio Days Validated To Total Days</td>
      <td>numerical</td>
      <td>OPTIMAL</td>
      <td>0.203219</td>
      <td>0.009202</td>
      <td>120.768005</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Ratio Days Qualified To Total Days</td>
      <td>numerical</td>
      <td>OPTIMAL</td>
      <td>0.127896</td>
      <td>0.005017</td>
      <td>98.344976</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Sales Stage Change Count</td>
      <td>numerical</td>
      <td>OPTIMAL</td>
      <td>0.155516</td>
      <td>0.004933</td>
      <td>96.695925</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Client Size By Revenue (USD)</td>
      <td>categorical</td>
      <td>OPTIMAL</td>
      <td>0.092308</td>
      <td>0.004104</td>
      <td>107.175620</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Client Size By Employee Count</td>
      <td>categorical</td>
      <td>OPTIMAL</td>
      <td>0.081818</td>
      <td>0.003784</td>
      <td>74.086113</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Region</td>
      <td>categorical</td>
      <td>OPTIMAL</td>
      <td>0.110277</td>
      <td>0.003119</td>
      <td>40.686784</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Revenue From Client Past Two Years (USD)</td>
      <td>categorical</td>
      <td>OPTIMAL</td>
      <td>0.055682</td>
      <td>0.002483</td>
      <td>194.217526</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Supplies Group</td>
      <td>categorical</td>
      <td>OPTIMAL</td>
      <td>0.093594</td>
      <td>0.001814</td>
      <td>141.810865</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Elapsed Days In Sales Stage</td>
      <td>numerical</td>
      <td>OPTIMAL</td>
      <td>0.030930</td>
      <td>0.000632</td>
      <td>16.453816</td>
    </tr>
  </tbody>
</table>
</div>




```python
amount_bin_df.to_csv('../data/processed/amount_optbinning_explained_variance.csv', index=False)
if not failed_df.empty:
    failed_df.to_csv('../data/processed/amount_optbinning_failed.csv', index=False)
print('saved: ../data/processed/amount_optbinning_explained_variance.csv')
```

    saved: ../data/processed/amount_optbinning_explained_variance.csv


## Top variable continuous binning table


```python
if amount_bin_df.empty:
    print('No successful continuous optbinning fits to display.')
else:
    top_var = amount_bin_df.loc[0, 'variable']
    dtype = 'numerical' if pd.api.types.is_numeric_dtype(df[top_var]) else 'categorical'
    cob = ContinuousOptimalBinning(name=top_var, dtype=dtype)
    cob.fit(df[top_var], df['target_log_amount'])
    print('Top variable:', top_var)
    display(cob.binning_table.build())
```

    Top variable: Deal Size Category (USD)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bin</th>
      <th>Count</th>
      <th>Count (%)</th>
      <th>Sum</th>
      <th>Std</th>
      <th>Mean</th>
      <th>Min</th>
      <th>Max</th>
      <th>Zeros count</th>
      <th>WoE</th>
      <th>IV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[10K or less]</td>
      <td>12095</td>
      <td>0.155014</td>
      <td>78898.973493</td>
      <td>3.263002</td>
      <td>6.523272</td>
      <td>0.000000</td>
      <td>9.210340</td>
      <td>2047</td>
      <td>-3.791020</td>
      <td>0.587663</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[10K to 20K]</td>
      <td>15123</td>
      <td>0.193822</td>
      <td>145466.360740</td>
      <td>0.292173</td>
      <td>9.618883</td>
      <td>9.210440</td>
      <td>10.126631</td>
      <td>0</td>
      <td>-0.695410</td>
      <td>0.134786</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[20K to 30K]</td>
      <td>11968</td>
      <td>0.153387</td>
      <td>124528.893866</td>
      <td>0.197771</td>
      <td>10.405155</td>
      <td>10.126671</td>
      <td>10.819778</td>
      <td>0</td>
      <td>0.090863</td>
      <td>0.013937</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[30K to 40K]</td>
      <td>13628</td>
      <td>0.174662</td>
      <td>150423.506464</td>
      <td>0.215005</td>
      <td>11.037827</td>
      <td>10.819798</td>
      <td>11.512925</td>
      <td>0</td>
      <td>0.723535</td>
      <td>0.126374</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[40K to 50K]</td>
      <td>18074</td>
      <td>0.231644</td>
      <td>213454.567664</td>
      <td>0.273613</td>
      <td>11.810035</td>
      <td>11.512935</td>
      <td>12.429216</td>
      <td>0</td>
      <td>1.495743</td>
      <td>0.346479</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[50K to 60K, More than 60K]</td>
      <td>7137</td>
      <td>0.091471</td>
      <td>92000.347383</td>
      <td>0.387359</td>
      <td>12.890619</td>
      <td>12.429220</td>
      <td>13.815512</td>
      <td>0</td>
      <td>2.576327</td>
      <td>0.235658</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Special</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>-10.314292</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Missing</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>-10.314292</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Totals</th>
      <td></td>
      <td>78025</td>
      <td>1.000000</td>
      <td>804772.649609</td>
      <td></td>
      <td>10.314292</td>
      <td>0.000000</td>
      <td>13.815512</td>
      <td>2047</td>
      <td>30.001481</td>
      <td>1.444898</td>
    </tr>
  </tbody>
</table>
</div>

