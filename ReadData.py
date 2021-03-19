import pandas as pd

#prefix = 'Data/WDataFiles_Stage1'
#prefix = 'Data/WDataFiles_Stage2'

def City(prefix):
    City = pd.read_csv(prefix + 'Cities.csv')
    #print(City, end='\n\n') # [453, 3] = [CityID, City, State]
    return City

def Conference(prefix):
    Conference = pd.read_csv(prefix + 'Conferences.csv')
    #print(Conference, end='\n\n') # [51, 2] = [ConfAbbrev, Description]
    return Conference

def GameCity(prefix):
    GameCity = pd.read_csv(prefix + 'WGameCities.csv')
    #print(GameCity, end='\n\n') # [57316, 6] = [Season, DayNum, WTeamId, LTeamID, CRType, CityID]
    return GameCity

def CompactResult(prefix):
    CompactResult = pd.read_csv(prefix + 'WNCAATourneyCompactResults.csv')
    #print(CompactResult, end='\n\n') # [1386, 8] = [Season, DayNum, WTeamID, LTeamID, LScore, WLoc, NumOT]
    return CompactResult

def DetailedResult(prefix):
    DetailedResult = pd.read_csv(prefix + 'WNCAATourneyDetailedResults.csv')
    #print(DetailedResult, end='\n\n') # [630, 34] = [Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc ... LOR, LDR, LAst, LTO, LSTl, LBlk, LPF]
    return DetailedResult

def Seed(prefix):
    Seed = pd.read_csv(prefix + 'WNCAATourneySeeds.csv')
    #print(Seed, end='\n\n') # [1408, 3] = [Season, Seed, TeamID]
    return Seed

def Slot(prefix):
    Slot = pd.read_csv(prefix + 'WNCAATourneySlots.csv')
    #print(Slot, end='\n\n') # [63, 3] = [Slot, StrongSeed, WeakSeed]
    return Slot

def RegularCompactResult(prefix):
    RegularCompactResult = pd.read_csv(prefix + 'WRegularSeasonCompactResults.csv')
    #print(RegularCompactResult, end='\n\n') # [112183, 8] = [Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc, NumOT]
    return RegularCompactResult

def RegularDetailedResult(prefix):
    RegularDetailedResult = pd.read_csv(prefix + 'WRegularSeasonDetailedResults.csv')
    #print(RegularDetailedResult, end='\n\n') # [56793, 34] = [Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc ... LOR, LDR, LAst, LTO, LSTl, LBlk, LPF]
    return RegularDetailedResult

def Season(prefix):
    Season = pd.read_csv(prefix + 'WSeasons.csv')
    #print(Season, end='\n\n') # [24, 6] = [Season, DayZero, RegionW, RegionX, RegionY, RegionZ]
    return Season

def TeamConference(prefix):
    TeamConference  = pd.read_csv(prefix + 'WTeamConferences.csv')
    #print(TeamConference, end='\n\n') # [8051, 3] = [Season, TeamID, ConfAbbrev]
    return TeamConference

def Team(prefix):
    Team = pd.read_csv(prefix + 'WTeams.csv')
    #print(Team, end='\n\n') # [369, 2] = [TeamID, TeamName]
    return Team

def TeamSpelling(prefix):
    TeamSpelling = pd.read_csv(prefix + 'WTeamSpellings.csv', encoding='cp1252')
    #print(TeamSpelling, end='\n\n') # [1140, 2] = [TeamNameSpelling, TeamID]
    return TeamSpelling

def Output(prefix):
    if prefix[-2] == '1':
        Output = pd.read_csv(prefix + 'WSampleSubmissionStage1.csv')
    if prefix[-2] == '2':
        Output = pd.read_csv(prefix + 'WSampleSubmissionStage2.csv')
    #print(Output, end='\n\n')
    return Output

def Rating(prefix):
    rating = pd.read_csv(prefix + '538ratingsWomen.csv')
    #print(rating, end='\n\n')
    rate = []
    for i in range(len(rating)):
        change = False
        for j in range(len(rate)):
            if rating['TeamID'][i] == rate[j][0]:
                rate[j][1] == rating['538rating'][i]
                change = True
        if not change:
            rate.append([rating['TeamID'][i], rating['538rating'][i]])

    rate.sort()
    df = pd.DataFrame(rate, columns=['TeamID', 'Rating'])
    #print(df)
    return df
