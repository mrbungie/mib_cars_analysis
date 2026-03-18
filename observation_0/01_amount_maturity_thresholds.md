# Observation 0.1 - Amount Maturity Threshold Diagnostics

This parallel-level Observation 0 notebook evaluates amount-target maturity using three heuristics:

1. Cross-sectional predictability (R^2 plateau)
2. Round-number heuristic (scope proxy)
3. Probability-driven maturity (CV stabilization by P(win))

Data source: `../1_iteration1_exploration/cars.xlsx`


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, cross_val_predict

pd.set_option('display.max_columns', 200)
sns.set_theme(style='whitegrid')
```


```python
df = pd.read_excel('../1_iteration1_exploration/cars.xlsx')

# Core targets
df['target_amount'] = df['Opportunity Amount USD'].astype(float)
df['target_win'] = df['Opportunity Result'].map({'Won': 1, 'Loss': 0})

# Age buckets (proxy for stage maturity)
bins = [-1, 15, 30, 45, 60, 75, 90, 120, np.inf]
labels = ['0-15', '16-30', '31-45', '46-60', '61-75', '76-90', '91-120', '121+']
df['age_bucket'] = pd.cut(df['Elapsed Days In Sales Stage'], bins=bins, labels=labels)

print('shape:', df.shape)
print('age bucket counts:')
print(df['age_bucket'].value_counts(dropna=False).sort_index())
```

    shape: (78025, 22)
    age bucket counts:
    age_bucket
    0-15      11752
    16-30     18370
    31-45     13018
    46-60     10814
    61-75     11744
    76-90     10649
    91-120     1640
    121+         38
    Name: count, dtype: int64


## Method 1 - Cross-sectional predictability (R^2 plateau)


```python
stable_base_features = [
    'Supplies Group',
    'Supplies Subgroup',
    'Region',
    'Route To Market',
    'Client Size By Revenue (USD)',
    'Client Size By Employee Count',
    'Revenue From Client Past Two Years (USD)',
    'Competitor Type',
]

# Additional stable non-funnel covariates (interaction-style composites)
df['supply_route_combo'] = (
    df['Supplies Group'].astype('string').fillna('Unknown') + '|' +
    df['Route To Market'].astype('string').fillna('Unknown')
)
df['region_route_combo'] = (
    df['Region'].astype('string').fillna('Unknown') + '|' +
    df['Route To Market'].astype('string').fillna('Unknown')
)
df['client_size_combo'] = (
    df['Client Size By Revenue (USD)'].astype('string').fillna('Unknown') + '|' +
    df['Client Size By Employee Count'].astype('string').fillna('Unknown')
)

creation_features = stable_base_features + [
    'supply_route_combo',
    'region_route_combo',
    'client_size_combo',
]

work = df[creation_features + ['target_amount', 'age_bucket']].copy()
for c in creation_features:
    work[c] = work[c].astype('string').fillna('Unknown')

pre = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), creation_features)],
    remainder='drop'
)

reg = Pipeline(steps=[
    ('pre', pre),
    ('ridge', Ridge(alpha=3.0, random_state=42) if hasattr(Ridge(), 'random_state') else Ridge(alpha=3.0))
])

rows = []
for b in labels:
    sub = work[work['age_bucket'].astype(str) == b].dropna(subset=['target_amount'])
    n = len(sub)
    if n < 200:
        rows.append({'age_bucket': b, 'n_rows': n, 'r2_oof_mean': np.nan, 'r2_oof_std': np.nan, 'mae_oof_mean': np.nan, 'mae_oof_std': np.nan})
        continue

    x = sub[creation_features]
    y = sub['target_amount']
    n_splits = min(5, max(3, n // 100))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_validate(reg, x, y, cv=cv, scoring={'r2': 'r2', 'mae': 'neg_mean_absolute_error'})

    rows.append({
        'age_bucket': b,
        'n_rows': n,
        'r2_oof_mean': float(np.mean(scores['test_r2'])),
        'r2_oof_std': float(np.std(scores['test_r2'])),
        'mae_oof_mean': float(-np.mean(scores['test_mae'])),
        'mae_oof_std': float(np.std(-scores['test_mae'])),
    })

m1 = pd.DataFrame(rows)

valid = m1['r2_oof_mean'].dropna()
if len(valid) > 0:
    best = valid.max()
    # first bucket near best and materially positive
    cand = m1[(m1['r2_oof_mean'] >= best - 0.02) & (m1['r2_oof_mean'] >= 0.03)]
    maturity_bucket_m1 = cand['age_bucket'].iloc[0] if len(cand) else 'No clear threshold'
else:
    maturity_bucket_m1 = 'No clear threshold'

print('Method 1 maturity threshold bucket:', maturity_bucket_m1)
m1

```

    Method 1 maturity threshold bucket: 61-75





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
      <th>age_bucket</th>
      <th>n_rows</th>
      <th>r2_oof_mean</th>
      <th>r2_oof_std</th>
      <th>mae_oof_mean</th>
      <th>mae_oof_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0-15</td>
      <td>11752</td>
      <td>0.106235</td>
      <td>0.016507</td>
      <td>78145.478848</td>
      <td>1905.741800</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16-30</td>
      <td>18370</td>
      <td>0.121089</td>
      <td>0.011353</td>
      <td>75423.887369</td>
      <td>1522.334572</td>
    </tr>
    <tr>
      <th>2</th>
      <td>31-45</td>
      <td>13018</td>
      <td>0.107404</td>
      <td>0.005898</td>
      <td>75633.359934</td>
      <td>2326.348700</td>
    </tr>
    <tr>
      <th>3</th>
      <td>46-60</td>
      <td>10814</td>
      <td>0.118724</td>
      <td>0.010332</td>
      <td>75925.032521</td>
      <td>2136.704284</td>
    </tr>
    <tr>
      <th>4</th>
      <td>61-75</td>
      <td>11744</td>
      <td>0.144868</td>
      <td>0.012083</td>
      <td>70723.203216</td>
      <td>1722.793756</td>
    </tr>
    <tr>
      <th>5</th>
      <td>76-90</td>
      <td>10649</td>
      <td>0.134045</td>
      <td>0.013649</td>
      <td>68357.622333</td>
      <td>1552.019140</td>
    </tr>
    <tr>
      <th>6</th>
      <td>91-120</td>
      <td>1640</td>
      <td>0.106867</td>
      <td>0.022592</td>
      <td>81075.244273</td>
      <td>2880.507849</td>
    </tr>
    <tr>
      <th>7</th>
      <td>121+</td>
      <td>38</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.lineplot(data=m1, x='age_bucket', y='r2_oof_mean', marker='o', ax=axes[0])
axes[0].set_title('R^2 by Age Bucket')
axes[0].axhline(0, color='gray', lw=1, ls='--')

sns.lineplot(data=m1, x='age_bucket', y='mae_oof_mean', marker='o', ax=axes[1])
axes[1].set_title('MAE by Age Bucket')

for ax in axes:
    ax.set_xlabel('Age bucket')
plt.tight_layout()
```


    
![png](01_amount_maturity_thresholds_files/01_amount_maturity_thresholds_5_0.png)
    


## Method 2 - Round-number heuristic (scope proxy)


```python
amt = df['target_amount'].abs()

# stricter roundness proxy: only coarse planning anchors (10k/50k multiples)
df['is_round_10k'] = (amt % 10000 == 0)
df['is_round_50k'] = (amt % 50000 == 0)
df['is_round_amount'] = df['is_round_10k'] | df['is_round_50k']

round_by_bucket = (df.groupby('age_bucket', observed=False)['is_round_amount']
                   .mean()
                   .mul(100)
                   .rename('round_pct')
                   .reset_index())

closed_won_baseline = float(df.loc[df['Opportunity Result'] == 'Won', 'is_round_amount'].mean() * 100)

# threshold: first bucket where round% reaches baseline (+2pp tolerance)
tol = 2.0
cand2 = round_by_bucket[round_by_bucket['round_pct'] <= closed_won_baseline + tol]
maturity_bucket_m2 = cand2['age_bucket'].astype(str).iloc[0] if len(cand2) else 'No clear threshold'

print('Closed-Won round% baseline:', round(closed_won_baseline, 2))
print('Method 2 maturity threshold bucket:', maturity_bucket_m2)
round_by_bucket
```

    Closed-Won round% baseline: 19.7
    Method 2 maturity threshold bucket: 121+





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
      <th>age_bucket</th>
      <th>round_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0-15</td>
      <td>47.106875</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16-30</td>
      <td>47.425150</td>
    </tr>
    <tr>
      <th>2</th>
      <td>31-45</td>
      <td>46.658473</td>
    </tr>
    <tr>
      <th>3</th>
      <td>46-60</td>
      <td>48.853338</td>
    </tr>
    <tr>
      <th>4</th>
      <td>61-75</td>
      <td>50.953678</td>
    </tr>
    <tr>
      <th>5</th>
      <td>76-90</td>
      <td>51.187905</td>
    </tr>
    <tr>
      <th>6</th>
      <td>91-120</td>
      <td>35.182927</td>
    </tr>
    <tr>
      <th>7</th>
      <td>121+</td>
      <td>18.421053</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(8, 4))
sns.lineplot(data=round_by_bucket, x='age_bucket', y='round_pct', marker='o')
plt.axhline(closed_won_baseline, color='red', ls='--', label=f'Closed-Won baseline ({closed_won_baseline:.1f}%)')
plt.title('Round-number % by Age Bucket')
plt.xlabel('Age bucket')
plt.ylabel('Round amount %')
plt.legend()
plt.tight_layout()
```


    
![png](01_amount_maturity_thresholds_files/01_amount_maturity_thresholds_8_0.png)
    


## Method 3 - Probability-driven maturity (CV stabilization)


```python
# Build P(win) model with broad snapshot features (excluding amount/deal-size to avoid circularity)
exclude = {
    'Opportunity Number', 'Opportunity Result', 'target_win',
    'Opportunity Amount USD', 'target_amount', 'Deal Size Category (USD)'
}
feat_cols = [c for c in df.columns if c not in exclude and c != 'age_bucket' and c != 'is_round_amount']

X = df[feat_cols].copy()
y = df['target_win'].copy()

for c in X.select_dtypes(include=['object', 'string', 'category']).columns:
    X[c] = X[c].astype('string').fillna('Unknown')

cat_cols = X.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
num_cols = X.select_dtypes(include=['number', 'bool']).columns.tolist()

pre_clf = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', StandardScaler(), num_cols),
    ],
    remainder='drop'
)

clf = Pipeline(steps=[
    ('pre', pre_clf),
    ('lr', LogisticRegression(max_iter=1500, class_weight='balanced'))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
df['p_win_oof'] = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]

# Probability bins
df['pwin_bin'] = pd.qcut(df['p_win_oof'], q=10, duplicates='drop')

cv_tbl = (df.groupby('pwin_bin', observed=False)['target_amount']
          .agg(['count', 'mean', 'std'])
          .reset_index())
cv_tbl['cv'] = cv_tbl['std'] / cv_tbl['mean'].replace(0, np.nan)
cv_tbl['pwin_left'] = cv_tbl['pwin_bin'].apply(lambda x: float(x.left) if pd.notna(x) else np.nan)
cv_tbl['pwin_right'] = cv_tbl['pwin_bin'].apply(lambda x: float(x.right) if pd.notna(x) else np.nan)
cv_tbl = cv_tbl.sort_values('pwin_left').reset_index(drop=True)

# stricter stabilization heuristic: local smoothness + low CV + forward stability
cv_tbl['rolling_range_3'] = cv_tbl['cv'].rolling(3, min_periods=3).apply(
    lambda s: float(np.max(s) - np.min(s)), raw=True
)
cv_tbl['forward_range'] = [
    float(cv_tbl.loc[i:, 'cv'].max() - cv_tbl.loc[i:, 'cv'].min())
    for i in range(len(cv_tbl))
]
low_cv_cutoff = float(cv_tbl['cv'].quantile(0.35))

cand3 = cv_tbl[(cv_tbl['rolling_range_3'] <= 0.10) & (cv_tbl['cv'] <= low_cv_cutoff) & (cv_tbl['forward_range'] <= 0.12)]
if len(cand3):
    pwin_threshold = float(cand3.iloc[0]['pwin_left'])
    maturity_threshold_m3 = f'p_win > {pwin_threshold:.3f}'
else:
    pwin_threshold = np.nan
    maturity_threshold_m3 = 'No clear threshold'

print('Method 3 maturity threshold:', maturity_threshold_m3)
cv_tbl
```

    Method 3 maturity threshold: No clear threshold





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
      <th>pwin_bin</th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>cv</th>
      <th>pwin_left</th>
      <th>pwin_right</th>
      <th>rolling_range_3</th>
      <th>forward_range</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(-0.0009997790000000001, 0.0311]</td>
      <td>7803</td>
      <td>136209.003845</td>
      <td>149748.423463</td>
      <td>1.099402</td>
      <td>-0.0010</td>
      <td>0.0311</td>
      <td>NaN</td>
      <td>0.642766</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(0.0311, 0.0692]</td>
      <td>7802</td>
      <td>121338.484619</td>
      <td>142157.924105</td>
      <td>1.171582</td>
      <td>0.0311</td>
      <td>0.0692</td>
      <td>NaN</td>
      <td>0.570586</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(0.0692, 0.12]</td>
      <td>7803</td>
      <td>106683.022043</td>
      <td>136068.454901</td>
      <td>1.275446</td>
      <td>0.0692</td>
      <td>0.1200</td>
      <td>0.176044</td>
      <td>0.466722</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(0.12, 0.193]</td>
      <td>7802</td>
      <td>89798.617149</td>
      <td>128590.436787</td>
      <td>1.431987</td>
      <td>0.1200</td>
      <td>0.1930</td>
      <td>0.260405</td>
      <td>0.310181</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(0.193, 0.289]</td>
      <td>7803</td>
      <td>90890.035884</td>
      <td>136095.718394</td>
      <td>1.497367</td>
      <td>0.1930</td>
      <td>0.2890</td>
      <td>0.221921</td>
      <td>0.244801</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(0.289, 0.409]</td>
      <td>7802</td>
      <td>88765.660728</td>
      <td>136081.530591</td>
      <td>1.533043</td>
      <td>0.2890</td>
      <td>0.4090</td>
      <td>0.101056</td>
      <td>0.209125</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(0.409, 0.551]</td>
      <td>7802</td>
      <td>72678.804409</td>
      <td>116814.292515</td>
      <td>1.607268</td>
      <td>0.4090</td>
      <td>0.5510</td>
      <td>0.109901</td>
      <td>0.134900</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(0.551, 0.734]</td>
      <td>7803</td>
      <td>67073.986415</td>
      <td>114374.065137</td>
      <td>1.705193</td>
      <td>0.5510</td>
      <td>0.7340</td>
      <td>0.172150</td>
      <td>0.036975</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(0.734, 0.867]</td>
      <td>7802</td>
      <td>70384.477057</td>
      <td>122581.121412</td>
      <td>1.741593</td>
      <td>0.7340</td>
      <td>0.8670</td>
      <td>0.134325</td>
      <td>0.000575</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(0.867, 1.0]</td>
      <td>7803</td>
      <td>72548.564783</td>
      <td>126391.785109</td>
      <td>1.742168</td>
      <td>0.8670</td>
      <td>1.0000</td>
      <td>0.036975</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(9, 4))
plot_tbl = cv_tbl.copy()
plot_tbl['bin_label'] = plot_tbl['pwin_bin'].astype(str)
sns.lineplot(data=plot_tbl, x='bin_label', y='cv', marker='o')
if np.isfinite(pwin_threshold):
    plt.axvline(
        x=max(0, (plot_tbl['pwin_left'] < pwin_threshold).sum() - 1),
        color='red',
        ls='--',
        label=f'Threshold ~ {pwin_threshold:.2f}'
    )
plt.xticks(rotation=30, ha='right')
plt.title('Amount CV by P(win) Bin')
plt.xlabel('P(win) bins')
plt.ylabel('Coefficient of Variation')
if np.isfinite(pwin_threshold):
    plt.legend()
plt.tight_layout()
```


    
![png](01_amount_maturity_thresholds_files/01_amount_maturity_thresholds_11_0.png)
    


## Save artifacts


```python
m1.to_csv('observation0_amount_method1_r2_mae_by_age_bucket.csv', index=False)
round_by_bucket.to_csv('observation0_amount_method2_roundpct_by_age_bucket.csv', index=False)
cv_tbl.to_csv('observation0_amount_method3_cv_by_pwin_bin.csv', index=False)
# Primary maturity flag based on Method 1 (age bucket >= 61-90)
if maturity_bucket_m1 in labels:
    maturity_start_idx = labels.index(maturity_bucket_m1)
    mature_labels = labels[maturity_start_idx:]
    df['is_mature'] = df['age_bucket'].astype(str).isin(mature_labels)
else:
    mature_labels = []
    df['is_mature'] = False
# Alternative stability metric: Median Absolute Deviation (MAD) per P(win) bin
mad_tbl = (df.groupby('pwin_bin', observed=False)['target_amount']
            .apply(lambda s: np.median(np.abs(s - np.median(s))))
            .reset_index(name='mad'))
cv_tbl = cv_tbl.merge(mad_tbl, on='pwin_bin', how='left')

summary = pd.DataFrame([
    {'method': 'Method 1 (R2 plateau)', 'threshold': str(maturity_bucket_m1)},
    {'method': 'Method 2 (round-number baseline)', 'threshold': str(maturity_bucket_m2)},
    {'method': 'Method 3 (CV by P(win))', 'threshold': maturity_threshold_m3},
])
summary.to_csv('observation0_amount_maturity_threshold_summary.csv', index=False)
summary
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
      <th>method</th>
      <th>threshold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Method 1 (R2 plateau)</td>
      <td>61-75</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Method 2 (round-number baseline)</td>
      <td>121+</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Method 3 (CV by P(win))</td>
      <td>No clear threshold</td>
    </tr>
  </tbody>
</table>
</div>



## Decision usage

Use the three thresholds together (intersection or consensus rule) to define a high-confidence mature subset for amount modeling.
