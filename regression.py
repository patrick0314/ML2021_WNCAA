import numpy as np
import pandas as pd
import random

# Read Data
print('=== ReadData ===')
prefix = 'Data/WDataFiles_Stage1'
#prefix = 'Data/WDataFiles_Stage2'
dt = {'WTeamID':'str', 'LTeamID':'str'}
#ss = pd.read_csv(prefix + 'WSampleSubmissionStage1.csv')
ss = pd.read_csv(prefix + 'WSampleSubmissionStage2.csv')
sd1 = pd.read_csv(prefix + 'WRegularSeasonCompactResults.csv', dtype=dt)
sd2 = pd.read_csv(prefix + 'WRegularSeasonDetailedResults.csv', dtype=dt)
sd2 = sd2.drop(['WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF'], axis=1)
sd2 = sd2.drop(['LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF'], axis=1)
sd = pd.concat([sd1, sd2], axis=0, ignore_index=True)
td1 = pd.read_csv(prefix + 'WNCAATourneyCompactResults.csv', dtype=dt)
td2 = pd.read_csv(prefix + 'WNCAATourneyDetailedResults.csv', dtype=dt)
td2 = td2.drop(['WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF'], axis=1)
td2 = td2.drop(['LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF'], axis=1)
td = pd.concat([td1, td2], axis=0, ignore_index=True)
ts = pd.read_csv(prefix + 'WNCAATourneySeeds.csv', dtype={'TeamID':'str'})

# Dataframe  formation with feature engineering
print('=== Feature Engineering ===')
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

# Linear Regression
print('=== Linear Regression ===')
cols = ['T1ScoreDiff','T2ScoreDiff','T1WinRate','T2WinRate','T1Seed','T2Seed']
def get_train_test(df, test_season):
    train_df = df.loc[df['Season']!=test_season, cols+['target']]
    test_df = df.loc[df['Season']==test_season, cols+['target']]
    return train_df, test_df

seasons = [2015, 2016, 2017, 2018, 2019]
total_loss = 0
for season in seasons:
    print('Training by: ', season, ' with resulting loss: ', end='')
    train, test = get_train_test(td, season) # [1323, 7] [63, 7]
    
    train_x, train_y = train.drop('target', axis=1), train['target']
    train_x, train_y = train_x.to_numpy(), train_y.to_numpy()
    
    w = np.zeros(len(train_x[0]))
    l_rate = 20
    repeat = 100000
    prev_grad = np.zeros(len(train_x[0]))
    train_x_t = train_x.transpose()

    for i in range(repeat):
        tmp = np.dot(train_x, w)
        loss = tmp - train_y
        grad = 2 * np.dot(train_x_t, loss)
        prev_grad += grad ** 2
        ada = np.sqrt(prev_grad)
        w -= l_rate * grad / ada

    test_x, test_y = test.drop('target', axis=1).to_numpy(), test['target'].to_numpy()
    pred = np.dot(test_x, w)
    loss = np.absolute(pred - test_y)
    total_loss += np.sum(loss) / len(loss)
    print(np.sum(loss) / len(loss))

print('Average', total_loss / len(seasons), end='\n\n')

# Prediction
print('=== Prediction ===')
x, y = td[cols].to_numpy(), td['target'].to_numpy()
w = np.zeros(len(x[0]))
l_rate = 10
repeat = 10000
prev_grad = np.zeros(len(train_x[0]))
x_t = x.transpose()

for i in range(repeat):
    tmp = np.dot(x, w)
    loss = tmp - y
    grad = 2 * np.dot(x_t, loss)
    prev_grad += grad ** 2
    ada = np.sqrt(prev_grad)
    w -= l_rate * grad / ada

xx = ss[cols].to_numpy()
pred = np.dot(xx, w)
pred[pred<0.5] = 0
ss['Pred'] = pred.clip(0, 1)
print(ss.head(10))
ss.to_csv('regression.csv', columns=['ID','Pred'], index=None)
