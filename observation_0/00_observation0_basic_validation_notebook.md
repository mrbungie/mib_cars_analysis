# Observation 0 - Opportunity ID Validation

Parallel-level analysis (outside iteration folders).

Goal:
- Check if `Opportunity Number` is unique
- Separate exact duplicate rows vs state-change rows for repeated IDs


```python
import pandas as pd

pd.set_option('display.max_columns', 200)
```


```python
df = pd.read_excel('../1_iteration1_exploration/cars.xlsx')
ID = 'Opportunity Number'
print('shape:', df.shape)
print('unique IDs:', df[ID].nunique(dropna=False))
print('is unique:', df[ID].nunique(dropna=False) == len(df))
```

    shape: (78025, 19)
    unique IDs: 77829
    is unique: False



```python
mask = df[ID].duplicated(keep=False)
dup = df[mask].copy()
non_id = [c for c in df.columns if c != ID]

summary = []
for oid, g in dup.groupby(ID, sort=False):
    distinct_state = g[non_id].drop_duplicates().shape[0]
    summary.append({
        ID: int(oid),
        'rows_for_id': len(g),
        'distinct_states_excl_id': distinct_state,
        'is_exact_repeat_only': distinct_state == 1
    })

s = pd.DataFrame(summary)
print('duplicated IDs:', len(s))
print('duplicate rows:', len(dup))
print('exact-repeat IDs:', int(s['is_exact_repeat_only'].sum()))
print('state-change IDs:', int((~s['is_exact_repeat_only']).sum()))

s.head()
```

    duplicated IDs: 183
    duplicate rows: 379
    exact-repeat IDs: 46
    state-change IDs: 137





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
      <th>Opportunity Number</th>
      <th>rows_for_id</th>
      <th>distinct_states_excl_id</th>
      <th>is_exact_repeat_only</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4947042</td>
      <td>2</td>
      <td>2</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5629727</td>
      <td>2</td>
      <td>2</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5799657</td>
      <td>2</td>
      <td>2</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5934206</td>
      <td>2</td>
      <td>2</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5943944</td>
      <td>2</td>
      <td>2</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
exact_ids = s.loc[s['is_exact_repeat_only'], ID].tolist()
state_ids = s.loc[~s['is_exact_repeat_only'], ID].tolist()

exact_rows = dup[dup[ID].isin(exact_ids)].copy().sort_values([ID])
state_rows = dup[dup[ID].isin(state_ids)].copy().sort_values([ID])

exact_rows.to_csv('observation0_exact_repeat_rows.csv', index=False)
state_rows.to_csv('observation0_state_change_rows.csv', index=False)

split_summary = pd.DataFrame({
    'group': ['exact_repeat_ids', 'state_change_ids', 'exact_repeat_rows', 'state_change_rows'],
    'count': [len(exact_ids), len(state_ids), len(exact_rows), len(state_rows)]
})
split_summary.to_csv('observation0_duplicate_split_summary.csv', index=False)

print('saved: observation0_exact_repeat_rows.csv')
print('saved: observation0_state_change_rows.csv')
print('saved: observation0_duplicate_split_summary.csv')
```

    saved: observation0_exact_repeat_rows.csv
    saved: observation0_state_change_rows.csv
    saved: observation0_duplicate_split_summary.csv


## Outputs

- `observation0_exact_repeat_rows.csv`
- `observation0_state_change_rows.csv`
- `observation0_duplicate_split_summary.csv`
- `observation0_opportunity_duplicates.csv`
- `observation0_duplicate_state_analysis.csv`
- `observation0_duplicate_state_sample.csv`
