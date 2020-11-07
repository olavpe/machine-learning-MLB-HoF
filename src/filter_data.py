import pandas as pd
import numpy as np
from numpy.random import RandomState

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
# print(df.head())
# Adding back to dataset as ints
df['finalGame'] = df['finalGame'].astype(int)


### Fixing awards parts

# Replacing NaN in each column with 0 and adding back as type
award_names = ['Most Valuable Player', 'World Series MVP', 'AS_games', 'Gold Glove',
               'Rookie of the Year', 'Silver Slugger']
for award in award_names:
    df[award] = df[award].fillna(0)
    df[award] = df[award].astype(int)


### Creating 1B, SLG, OBS, OPS

df['1B'] = df['H'] - df['2B'] - df['3B'] - df['HR']
df['SLG'] = (df['1B'] + 2*df['2B'] + 3*df['3B'] + 4*df['HR'])/df['AB']
df['OBS'] = (df['H'] + df['BB'] + df['HBP'])/(df['AB'] + df['BB']+ df['HBP']+ df['SF'])
df['OPS'] = df['SLG'] + df['OBS']


### Removing players who have played less than 10 years

df = df[df['Years_Played'] >= 10] # 10 years
# df = df[df['G_all'] >= ] # Testing
print("length after removing players less than 10 yrs", len(df['playerID']))

print(df[['playerID','HoF']].head())

# ### Separating HoF with non-HoF players
# reg_df = df[df['HoF'].isnull()]
# print("Eligible players not in Hall of Fame", len(reg_df['playerID']))
# print(reg_df[['playerID','HoF']].head())
# hof_df = df[df['HoF'].notnull()]
# print("Players in the hall of fame", len(hof_df['playerID']))
# print(hof_df[['playerID','HoF']].head())

### Separating into test and training data

# Creating proper random seed number
seeded_random = RandomState(1)
train_frac = 0.8
train_df = df.sample(frac=train_frac, random_state=seeded_random)
test_df = df.drop(train_df.index)


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
