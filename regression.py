import numpy as np
import pandas as pd
import random

# Read Data
prefix = 'Data/'
dt = {'WTeamID':'str', 'LTeamID':'str'}
ss = pd.read_csv(prefix + 'WSampleSubmissionStage1.csv')
sd = pd.read_csv(prefix + 'WRegularSeasonCompactResults.csv', dtype=dt)
td = pd.read_csv(prefix + 'WNCAATourneyCompactResults.csv', dtype=dt)
ts = pd.read_csv(prefix + 'WNCAATourneySeeds.csv', dtype={'TeamID':'str'})

ts['Seed'] = ts['Seed'].map(lambda s: s[1:])
sd['DScore'] = sd['WScore'] - sd['LScore']

for i in range(len(td.index)):
    if random.choices([0,1]) == [1]:
        td.at[i, 'Team1'] = td.at[i, 'WTeamID']
        td.at[i, 'Team2'] = td.at[i, 'LTeamID']
        td.at[i, 'target'] = 1.0
    else:
        td.at[i, 'Team1'] = td.at[i, 'LTeamID']
        td.at[i, 'Team2'] = td.at[i, 'WTeamID']
        td.at[i, 'target'] = 0.0

ss['Team1'] = ss['ID'].map(lambda s: s[5:9])
ss['Team2'] = ss['ID'].map(lambda s: s[10:])
ss['Season'] = ss['ID'].map(lambda s: s[:4])

stat = {}
def calculate_stat(season, team):
    if (season, team) in stat.keys():
        return
    t_w = sd.loc[(sd['Season']==season)&(sd['WTeamID']==team),'DScore']
    t_l = sd.loc[(sd['Season']==season)&(sd['LTeamID']==team),'DScore']
    t_wc = len(t_w.index)
    t_lc = len(t_l.index)
    t_ws = t_w.sum()
    t_ls = t_l.sum()
    stat[(season, team)] = {}
    stat[(season, team)]['WinRate'] = t_wc / (t_wc+t_lc)
    stat[(season, team)]['ScoreDiff'] = t_ws - t_ls
    stat[(season, team)]['Seed'] = int(ts.loc[(ts['Season']==season)&(ts['TeamID']==team),'Seed'].any())

def feat(df):
    for i in df.index:
        season = int(df.at[i, 'Season'])
        team1 = df.at[i, 'Team1']
        team2 = df.at[i, 'Team2']
        calculate_stat(season, team1)
        calculate_stat(season, team2)
        df.at[i, 'T1WinRate'] = stat[(season, team1)]['WinRate']
        df.at[i, 'T2WinRate'] = stat[(season, team2)]['WinRate']
        df.at[i, 'T1ScoreDiff'] = stat[(season, team1)]['ScoreDiff']
        df.at[i, 'T2ScoreDiff'] = stat[(season, team2)]['ScoreDiff']
        df.at[i, 'T1Seed'] = stat[(season, team1)]['Seed']
        df.at[i, 'T2Seed'] = stat[(season, team2)]['Seed']
    return df

td = feat(td)
ss = feat(ss)

print(type(td))
print(type(ss))
print(td.head(10))
print(ss.head(10))

cols = ['T1ScoreDiff','T2ScoreDiff','T1WinRate','T2WinRate','T1Seed','T2Seed']
def get_train_test(df, test_season):
    train_df = df.loc[df['Season']!=test_season, cols+['target']]
    test_df = df.loc[df['Season']==test_season, cols+['target']]
    return train_df, test_df

seasons = [2015, 2016, 2017, 2018, 2019]
for season in seasons:
    train, test = get_train_test(td, season)
    print(season)
    print(train)
    print(test)

'''
for season in seasons:
    train, test = get_train_test(td, season)
    model.fit(train.drop('target', axis=1), train['target'])
    pred = model.predict(test.drop('target', axis=1))
    loss = log_loss(test['target'], pred)
    print(season, loss)
    gloss += loss

print('average', gloss/len(seasons))

model.fit(td[cols], td['target'])
pred = model.predict(ss[cols])
ss['Pred'] = pred.clip(0, 1)
ss.to_csv('submission.csv', columns=['ID','Pred'], index=None)
ss
'''
