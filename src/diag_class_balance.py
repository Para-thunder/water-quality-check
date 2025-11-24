import pandas as pd
import os
p = os.path.join('data','processed','train.csv')
if not os.path.exists(p):
    print('train.csv not found at', p)
else:
    df = pd.read_csv(p)
    if 'Potability' in df.columns:
        print(df['Potability'].value_counts(dropna=False).to_dict())
    else:
        print('Potability column not present')
