import pandas as pd
import numpy as np
from numpy.random import RandomState
from sklearn.preprocessing import StandardScaler

### ---------------- Importing the data ---------------- ###

df = pd.read_csv('../data/final_data.csv')#,
                        # usecols=['playerID','nameFirst','nameLast','bats','throws','debut','finalGame'])

print(df.columns)

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

# df['HoF'] = df['HoF'].fillna('N')
# df['HoF'] = df['HoF'].astype(string)


### Creating 1B, SLG, OBS, OPS

df['1B'] = df['H'] - df['2B'] - df['3B'] - df['HR']
df['SLG'] = (df['1B'] + 2*df['2B'] + 3*df['3B'] + 4*df['HR'])/df['AB']
df['OBS'] = (df['H'] + df['BB'] + df['HBP'])/(df['AB'] + df['BB']+ df['HBP']+ df['SF'])
df['OPS'] = df['SLG'] + df['OBS']

new_stats = ['1B', 'SLG', 'OBS', 'OPS']
for col in new_stats:
    df[col] = df[col].fillna(0)
    df[col] = df[col].astype('float32')

stat_names = ['G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF', '1B', 'OPS', 'SLG', 'OBS']

for name in stat_names:
    string = 'number of NaN values in '+ name
    print(string, df[name].isna().sum())
# print('number of NaN values in SLG: ', df['SLG'].isna().sum())
# print('number of NaN values in OBS: ', df['OBS'].isna().sum())

### Removing players who have played less than 10 years

df = df[df['Years_Played'] >= 10] # 10 years
# df = df[df['G_all'] >= ] # Testing
print("length after removing players less than 10 yrs", len(df['playerID']))

print(df[['playerID','HoF']].head())

# df.to_csv('../data/examine_data.csv')

pre_std_df = df

# ### Separating HoF with non-HoF players
# reg_df = df[df['HoF'].isnull()]
# print("Eligible players not in Hall of Fame", len(reg_df['playerID']))
# print(reg_df[['playerID','HoF']].head())
# hof_df = df[df['HoF'].notnull()]
# print("Players in the hall of fame", len(hof_df['playerID']))
# print(hof_df[['playerID','HoF']].head())


### Selecting variables to use in training

# df = df[['playerID', 'HoF', 'G_all', 'finalGame', 'OPS',
df = df[['G_all', 'finalGame', 'OPS',
         'Years_Played', 'Most Valuable Player', 'AS_games',
         'Gold Glove', 'Rookie of the Year', 'World Series MVP', 'Silver Slugger',]]

print('df: ', df)
print()
### Standardizing the data

df = pd.DataFrame(StandardScaler().fit_transform(df), columns=df.columns)
print('df: ', df)
df['HoF'] = pre_std_df['HoF']
print('df: ', df)

### Converting HoF back to strings
df['HoF'] = df['HoF'].replace(np.nan, 'N', regex=True)
df['HoF'] = df['HoF'].replace(1.0, 'Y', regex=True)
df['HoF'] = df['HoF'].replace(0.0, 'N', regex=True)

### Separating into test and training data

seeded_random = RandomState(1)
train_frac = 0.8
train_df = df.sample(frac=train_frac, random_state=seeded_random)
test_df = df.drop(train_df.index)


########## Save the different testing and validation data sets. !!!!!!!
train_df.to_csv('../data/train_data.csv', index=False)
test_df.to_csv('../data/test_data.csv', index=False)

print("train_df : ", train_df)
print("test_df : ", test_df)
print("train_df length: ", len(train_df))
print("test_df length: ", len(test_df))


# print("dataframe row at: ", df.iloc[2])
# print("Should be at 2 x 2: ", type(df['Gold Glove'].iloc[2]))
# print("length of final game data: ", len(finalGameYear))
# print("number of values that are NaN: ", finalGameYear.isnull().sum())
# print(finalGameYear.head())
# print("length of master dataframe: ", len(df['playerID']))
# # Attempting to drop the values that are not ints
# # df.dropna()
# # print(df['Most Valuable Player'].describe(include='all'))
# print(df['Most Valuable Player'].head())
# print(df['G_all'].head())
# print(df['Years_Played'].head())
# print("length of master dataframe: ", len(df['playerID']))
# print("columns: ", df.columns)
# print(df['Gold Glove'].head())
