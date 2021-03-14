from ReadData import *
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

team = Team() # [369, 2] = [TeamID, TeamName]
n = len(team['TeamID'])
VS = np.zeros((n, n)) # [369, 369], [i, j] record the winning probability for team i playing with team j

compactresult = CompactResult() # [1386, 8] = [Season, DayNum, WteamID, LteamID, LScore, WLoc, NumOT]
for i in range(len(compactresult)):
    W = team.index[team['TeamID']==compactresult['WTeamID'][i]].to_list()
    L = team.index[team['TeamID']==compactresult['LTeamID'][i]].to_list()
    VS[W[0], L[0]] += 1
regularcompactresult = RegularCompactResult() # [112183, 8] = [Season, DayNum, WteamID, WScore, LteamID, LScore, WLoc, NumOT]
for i in range(len(regularcompactresult) * 2 // 3):
    W = team.index[team['TeamID']==regularcompactresult['WTeamID'][i]].to_list()
    L = team.index[team['TeamID']==regularcompactresult['LTeamID'][i]].to_list()
    VS[W[0], L[0]] += 1
detailedresult = DetailedResult() # [630, 34]
for i in range(len(detailedresult)):
    W = team.index[team['TeamID']==detailedresult['WTeamID'][i]].to_list()
    L = team.index[team['TeamID']==detailedresult['LTeamID'][i]].to_list()
    VS[W[0], L[0]] += 1
'''
regulardetailedresult = RegularDetailedResult() # [56793, 34]
for i in range(len(regulardetailedresult)):
    W = team.index[team['TeamID']==regulardetailedresult['WTeamID'][i]].to_list()
    L = team.index[team['TeamID']==regulardetailedresult['LTeamID'][i]].to_list()
    VS[W[0], L[0]] += 1
'''

for i in range(VS.shape[0]):
    for j in range(i+1, VS.shape[1]):
        tmp = VS[i, j] + VS[j, i]
        if tmp != 0:
            VS[i, j] /= tmp
            VS[j, i] /= tmp

output = Output()
for i in range(len(output['ID'])):
    output['ID'][i] = output['ID'][i].split('_')

for i in range(len(output['ID'])):
    W = team.index[team['TeamID']==int(output['ID'][i][1])].to_list()
    L = team.index[team['TeamID']==int(output['ID'][i][2])].to_list()
    output['Pred'][i] = VS[W[0], L[0]]

output1 = Output()
output1['Pred'] = output['Pred']
output1.to_csv('tradition.csv', index=0)
