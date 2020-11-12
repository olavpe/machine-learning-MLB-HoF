import pandas as pd
import numpy as np
from numpy.random import RandomState
from sklearn.preprocessing import StandardScaler


### Hyper-parameters
NAME = "80-20"
PERCENTAGE = 0.8


### ---------------- Helper Functions ---------------- ###

def num_sample(percentage, hof_df):
    hof_length = len(hof_df['G_all'])
    return int((percentage * hof_length)/(1 - percentage))


### ---------------- Importing and cleaning the data ---------------- ###

df = pd.read_csv('../data/final_data.csv')#,
                        # usecols=['playerID','nameFirst','nameLast','bats','throws','debut','finalGame'])

### Fixing Last Game Played
# Splits the final Game date and convert to string
finalGameYear = df['finalGame'].str.split(pat="-", expand=True)[0]
df['finalGame'] = finalGameYear
# Removing NaN fields from the dataset
df = df[finalGameYear.notnull()]
# Adding back to dataset as ints
df['finalGame'] = df['finalGame'].astype('float32')


### Fixing parts that cannot have NaN

##Awards parts
# Replacing NaN in each column with 0 and adding back as type
award_names = ['Most Valuable Player', 'World Series MVP', 'AS_games', 'Gold Glove',
               'Rookie of the Year', 'Silver Slugger','G', 'AB', 'R', 'H', '2B', '3B',
               'HR', 'RBI', 'SB', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF']
for col in award_names:
    df[col] = df[col].fillna(0)
    df[col] = df[col].astype('float32')


### ---------------- Creating additional statistics ---------------- ###

### Creating 1B, SLG, OBS, OPS
df['1B'] = df['H'] - df['2B'] - df['3B'] - df['HR']
df['SLG'] = (df['1B'] + 2*df['2B'] + 3*df['3B'] + 4*df['HR'])/df['AB']
df['OBS'] = (df['H'] + df['BB'] + df['HBP'])/(df['AB'] + df['BB']+ df['HBP']+ df['SF'])
df['OPS'] = df['SLG'] + df['OBS']

new_stats = ['1B', 'SLG', 'OBS', 'OPS']
for col in new_stats:
    df[col] = df[col].fillna(0)
    df[col] = df[col].astype('float32')

stat_names = ['G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB',
              'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF', '1B', 'OPS', 'SLG', 'OBS']
for name in stat_names:
    string = 'number of NaN values in '+ name
    print(string, df[name].isna().sum())


### ---------------- Fulfilling criteria of HoF eligibility ---------------- ###

### Removing players who have played less than 10 years
df = df[df['Years_Played'] >= 10] # 10 years


### ---------------- Preparation for input into model---------------- ###

### Selecting variables to use in training
pre_std_df = df
df = df[['G_all', 'finalGame', 'OPS',
         'Years_Played', 'Most Valuable Player', 'AS_games',
         'Gold Glove', 'Rookie of the Year', 'World Series MVP', 'Silver Slugger',]]

### Standardizing the data
df = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)
### Adding back the HoF data
df.insert(df.shape[1], 'HoF', pre_std_df['HoF'].to_numpy())

### Converting HoF back to strings
df['HoF'] = df['HoF'].replace(1.0, 'Y', regex=True)
df['HoF'] = df['HoF'].replace(np.nan, 'N', regex=True)
df['HoF'] = df['HoF'].replace(0.0, 'N', regex=True)

### Splitting dataset by HoF players
reg_df = df[df['HoF'] == 'N']
hof_df = df[df['HoF'] == 'Y']

### Under-sampling the non-HoF players
reg_seeded_random = RandomState(1)
sampled_reg_df = reg_df.sample(n = num_sample(PERCENTAGE, hof_df), random_state=reg_seeded_random)

### Splitting reg and HoF into train and test
sep_seeded_random = RandomState(1)
train_frac = 0.8
train_reg_df = sampled_reg_df.sample(frac=train_frac, random_state=sep_seeded_random)
test_reg_df = sampled_reg_df.drop(train_reg_df.index)
train_hof_df = hof_df.sample(frac=train_frac, random_state=sep_seeded_random)
test_hof_df = hof_df.drop(train_hof_df.index)

print('length of train_reg_df: ', len(train_reg_df))
print('length of test_reg_df: ', len(test_reg_df))
print('length of train_hof_df: ', len(train_hof_df))
print('length of test_hof_df: ', len(test_hof_df))

### Merging the test and training datasets
train_df = pd.concat([train_hof_df, train_reg_df])
test_df = pd.concat([test_hof_df, test_reg_df])
# Shuffling
train_df = train_df.sample(frac = 1, random_state=sep_seeded_random)
test_df = test_df.sample(frac = 1, random_state=sep_seeded_random)

# ### Directly separating into test and training data
# sep_seeded_random = RandomState(1)
# train_frac = 0.8
# train_df = df.sample(frac=train_frac, random_state=sep_seeded_random)
# test_df = df.drop(train_df.index)


### ---------------- Saving data ---------------- ###

########## Save the different testing and validation data sets. !!!!!!!
train_df.to_csv('../data/train_data_' + NAME + '.csv', index=False)
test_df.to_csv('../data/test_data_' + NAME + '.csv', index=False)
