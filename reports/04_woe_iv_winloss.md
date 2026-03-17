# 03 - WoE and IV for Win/Loss

This notebook runs variable-wise optimal binning for `Opportunity Result` (Won/Loss), computes WoE bins, and reports Information Value (IV) for all candidate predictors.


```python
import pandas as pd
import numpy as np
from optbinning import OptimalBinning

pd.set_option('display.max_columns', 200)
```


```python
df = pd.read_excel('../cars.xlsx')
df['target_win'] = df['Opportunity Result'].map({'Won': 1, 'Loss': 0})
print('shape:', df.shape)
print('target nulls:', df['target_win'].isna().sum())
```

    shape: (78025, 20)
    target nulls: 0


## Fit WoE/IV by variable


```python
excluded = {'Opportunity Result', 'target_win', 'Opportunity Number'}
features = [c for c in df.columns if c not in excluded]

results = []
failed = []
for col in features:
    x = df[col]
    y = df['target_win']
    dtype = 'numerical' if pd.api.types.is_numeric_dtype(x) else 'categorical'

    try:
        optb = OptimalBinning(name=col, dtype=dtype)
        optb.fit(x, y)
        bt = optb.binning_table.build()
        iv_total = float(bt.iloc[-1]['IV']) if 'IV' in bt.columns else np.nan
        js_total = float(bt.iloc[-1]['JS']) if 'JS' in bt.columns else np.nan
        results.append({
            'variable': col,
            'dtype': dtype,
            'status': optb.status,
            'iv': iv_total,
            'js': js_total
        })
    except Exception as e:
        failed.append({'variable': col, 'error': str(e)[:200]})

iv_df = pd.DataFrame(results)
if not iv_df.empty:
    iv_df = iv_df.sort_values('iv', ascending=False).reset_index(drop=True)
failed_df = pd.DataFrame(failed)
print('processed:', len(iv_df), 'failed:', len(failed_df))
iv_df.head(20)
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
      <th>iv</th>
      <th>js</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Total Days Identified Through Qualified</td>
      <td>numerical</td>
      <td>OPTIMAL</td>
      <td>0.801425</td>
      <td>0.094950</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Total Days Identified Through Closing</td>
      <td>numerical</td>
      <td>OPTIMAL</td>
      <td>0.692787</td>
      <td>0.082768</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Revenue From Client Past Two Years (USD)</td>
      <td>categorical</td>
      <td>OPTIMAL</td>
      <td>0.570736</td>
      <td>0.064250</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ratio Days Identified To Total Days</td>
      <td>numerical</td>
      <td>OPTIMAL</td>
      <td>0.397489</td>
      <td>0.044744</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Opportunity Amount USD</td>
      <td>numerical</td>
      <td>OPTIMAL</td>
      <td>0.332396</td>
      <td>0.039679</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ratio Days Qualified To Total Days</td>
      <td>numerical</td>
      <td>OPTIMAL</td>
      <td>0.315554</td>
      <td>0.038723</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Deal Size Category (USD)</td>
      <td>categorical</td>
      <td>OPTIMAL</td>
      <td>0.274840</td>
      <td>0.033554</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Ratio Days Validated To Total Days</td>
      <td>numerical</td>
      <td>OPTIMAL</td>
      <td>0.229913</td>
      <td>0.028174</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Sales Stage Change Count</td>
      <td>numerical</td>
      <td>OPTIMAL</td>
      <td>0.119074</td>
      <td>0.014761</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Supplies Subgroup</td>
      <td>categorical</td>
      <td>OPTIMAL</td>
      <td>0.065706</td>
      <td>0.008162</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Route To Market</td>
      <td>categorical</td>
      <td>OPTIMAL</td>
      <td>0.064877</td>
      <td>0.008088</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Elapsed Days In Sales Stage</td>
      <td>numerical</td>
      <td>OPTIMAL</td>
      <td>0.064456</td>
      <td>0.007833</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Competitor Type</td>
      <td>categorical</td>
      <td>OPTIMAL</td>
      <td>0.035146</td>
      <td>0.004361</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Region</td>
      <td>categorical</td>
      <td>OPTIMAL</td>
      <td>0.012827</td>
      <td>0.001601</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Supplies Group</td>
      <td>categorical</td>
      <td>OPTIMAL</td>
      <td>0.003531</td>
      <td>0.000441</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Client Size By Employee Count</td>
      <td>categorical</td>
      <td>OPTIMAL</td>
      <td>0.003246</td>
      <td>0.000405</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Client Size By Revenue (USD)</td>
      <td>categorical</td>
      <td>OPTIMAL</td>
      <td>0.002744</td>
      <td>0.000343</td>
    </tr>
  </tbody>
</table>
</div>



## IV interpretation bands


```python
def iv_band(iv):
    if iv < 0.02:
        return 'Not useful'
    if iv < 0.1:
        return 'Weak'
    if iv < 0.3:
        return 'Medium'
    if iv < 0.5:
        return 'Strong'
    return 'Suspicious/Very strong'

iv_df['iv_band'] = iv_df['iv'].apply(iv_band)
iv_df
```




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
      <th>iv</th>
      <th>js</th>
      <th>iv_band</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Total Days Identified Through Qualified</td>
      <td>numerical</td>
      <td>OPTIMAL</td>
      <td>0.801425</td>
      <td>0.094950</td>
      <td>Suspicious/Very strong</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Total Days Identified Through Closing</td>
      <td>numerical</td>
      <td>OPTIMAL</td>
      <td>0.692787</td>
      <td>0.082768</td>
      <td>Suspicious/Very strong</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Revenue From Client Past Two Years (USD)</td>
      <td>categorical</td>
      <td>OPTIMAL</td>
      <td>0.570736</td>
      <td>0.064250</td>
      <td>Suspicious/Very strong</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ratio Days Identified To Total Days</td>
      <td>numerical</td>
      <td>OPTIMAL</td>
      <td>0.397489</td>
      <td>0.044744</td>
      <td>Strong</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Opportunity Amount USD</td>
      <td>numerical</td>
      <td>OPTIMAL</td>
      <td>0.332396</td>
      <td>0.039679</td>
      <td>Strong</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ratio Days Qualified To Total Days</td>
      <td>numerical</td>
      <td>OPTIMAL</td>
      <td>0.315554</td>
      <td>0.038723</td>
      <td>Strong</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Deal Size Category (USD)</td>
      <td>categorical</td>
      <td>OPTIMAL</td>
      <td>0.274840</td>
      <td>0.033554</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Ratio Days Validated To Total Days</td>
      <td>numerical</td>
      <td>OPTIMAL</td>
      <td>0.229913</td>
      <td>0.028174</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Sales Stage Change Count</td>
      <td>numerical</td>
      <td>OPTIMAL</td>
      <td>0.119074</td>
      <td>0.014761</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Supplies Subgroup</td>
      <td>categorical</td>
      <td>OPTIMAL</td>
      <td>0.065706</td>
      <td>0.008162</td>
      <td>Weak</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Route To Market</td>
      <td>categorical</td>
      <td>OPTIMAL</td>
      <td>0.064877</td>
      <td>0.008088</td>
      <td>Weak</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Elapsed Days In Sales Stage</td>
      <td>numerical</td>
      <td>OPTIMAL</td>
      <td>0.064456</td>
      <td>0.007833</td>
      <td>Weak</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Competitor Type</td>
      <td>categorical</td>
      <td>OPTIMAL</td>
      <td>0.035146</td>
      <td>0.004361</td>
      <td>Weak</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Region</td>
      <td>categorical</td>
      <td>OPTIMAL</td>
      <td>0.012827</td>
      <td>0.001601</td>
      <td>Not useful</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Supplies Group</td>
      <td>categorical</td>
      <td>OPTIMAL</td>
      <td>0.003531</td>
      <td>0.000441</td>
      <td>Not useful</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Client Size By Employee Count</td>
      <td>categorical</td>
      <td>OPTIMAL</td>
      <td>0.003246</td>
      <td>0.000405</td>
      <td>Not useful</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Client Size By Revenue (USD)</td>
      <td>categorical</td>
      <td>OPTIMAL</td>
      <td>0.002744</td>
      <td>0.000343</td>
      <td>Not useful</td>
    </tr>
  </tbody>
</table>
</div>




```python
iv_df.to_csv('../data/processed/win_loss_iv_report.csv', index=False)
if not failed_df.empty:
    failed_df.to_csv('../data/processed/win_loss_iv_failed.csv', index=False)
print('saved: ../data/processed/win_loss_iv_report.csv')
print('saved failures:', (not failed_df.empty))
```

    saved: ../data/processed/win_loss_iv_report.csv
    saved failures: False


## Top variable WoE table example


```python
if iv_df.empty:
    print('No successful WoE/IV fits to display.')
else:
    top_var = iv_df.loc[0, 'variable']
    x = df[top_var]
    y = df['target_win']
    dtype = 'numerical' if pd.api.types.is_numeric_dtype(x) else 'categorical'
    optb = OptimalBinning(name=top_var, dtype=dtype)
    optb.fit(x, y)
    woe_table = optb.binning_table.build()
    print('Top variable:', top_var)
    display(woe_table)
```

    Top variable: Total Days Identified Through Qualified



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
      <th>Non-event</th>
      <th>Event</th>
      <th>Event rate</th>
      <th>WoE</th>
      <th>IV</th>
      <th>JS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(-inf, 0.50)</td>
      <td>9559</td>
      <td>0.122512</td>
      <td>5193</td>
      <td>4366</td>
      <td>0.456742</td>
      <td>-1.05806</td>
      <td>0.171097</td>
      <td>0.020442</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0.50, 3.50)</td>
      <td>9394</td>
      <td>0.120397</td>
      <td>5193</td>
      <td>4201</td>
      <td>0.447200</td>
      <td>-1.019535</td>
      <td>0.155324</td>
      <td>0.018616</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[3.50, 5.50)</td>
      <td>5403</td>
      <td>0.069247</td>
      <td>3509</td>
      <td>1894</td>
      <td>0.350546</td>
      <td>-0.614884</td>
      <td>0.030345</td>
      <td>0.003734</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[5.50, 7.50)</td>
      <td>4854</td>
      <td>0.062211</td>
      <td>3495</td>
      <td>1359</td>
      <td>0.279975</td>
      <td>-0.28694</td>
      <td>0.005518</td>
      <td>0.000687</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[7.50, 9.50)</td>
      <td>4786</td>
      <td>0.061339</td>
      <td>3655</td>
      <td>1131</td>
      <td>0.236314</td>
      <td>-0.05853</td>
      <td>0.000213</td>
      <td>0.000027</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[9.50, 12.50)</td>
      <td>6041</td>
      <td>0.077424</td>
      <td>4845</td>
      <td>1196</td>
      <td>0.197980</td>
      <td>0.16744</td>
      <td>0.002071</td>
      <td>0.000259</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[12.50, 15.50)</td>
      <td>5650</td>
      <td>0.072413</td>
      <td>4889</td>
      <td>761</td>
      <td>0.134690</td>
      <td>0.628586</td>
      <td>0.023744</td>
      <td>0.002920</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[15.50, 18.50)</td>
      <td>5003</td>
      <td>0.064120</td>
      <td>4383</td>
      <td>620</td>
      <td>0.123926</td>
      <td>0.724245</td>
      <td>0.027083</td>
      <td>0.003313</td>
    </tr>
    <tr>
      <th>8</th>
      <td>[18.50, 23.50)</td>
      <td>7347</td>
      <td>0.094162</td>
      <td>6675</td>
      <td>672</td>
      <td>0.091466</td>
      <td>1.064342</td>
      <td>0.077052</td>
      <td>0.009201</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[23.50, 28.50)</td>
      <td>5606</td>
      <td>0.071849</td>
      <td>5115</td>
      <td>491</td>
      <td>0.087585</td>
      <td>1.111964</td>
      <td>0.063197</td>
      <td>0.007516</td>
    </tr>
    <tr>
      <th>10</th>
      <td>[28.50, 33.50)</td>
      <td>4018</td>
      <td>0.051496</td>
      <td>3703</td>
      <td>315</td>
      <td>0.078397</td>
      <td>1.232802</td>
      <td>0.053552</td>
      <td>0.006300</td>
    </tr>
    <tr>
      <th>11</th>
      <td>[33.50, 42.50)</td>
      <td>4697</td>
      <td>0.060199</td>
      <td>4397</td>
      <td>300</td>
      <td>0.063871</td>
      <td>1.453371</td>
      <td>0.081071</td>
      <td>0.009327</td>
    </tr>
    <tr>
      <th>12</th>
      <td>[42.50, inf)</td>
      <td>5667</td>
      <td>0.072631</td>
      <td>5346</td>
      <td>321</td>
      <td>0.056644</td>
      <td>1.581139</td>
      <td>0.111157</td>
      <td>0.012607</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Special</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Missing</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Totals</th>
      <td></td>
      <td>78025</td>
      <td>1.000000</td>
      <td>60398</td>
      <td>17627</td>
      <td>0.225915</td>
      <td></td>
      <td>0.801425</td>
      <td>0.094950</td>
    </tr>
  </tbody>
</table>
</div>

