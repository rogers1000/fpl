##### Bring in base packages

import pandas as pd
import numpy as np
import requests
from google.colab import data_table
from matplotlib import pyplot
from scipy.stats import pearsonr
import plotly.express as px
from google.colab import files

data_table.enable_dataframe_formatter()

##### FPL General Information API Calling

fpl_general_info_url = 'https://fantasy.premierleague.com/api/bootstrap-static/'

fpl_general_info_r = requests.get(fpl_general_info_url)

fpl_general_info_json = fpl_general_info_r.json()

fpl_general_info_json.keys()

##### Creating a Teams Code Mapping DF

teams_df = pd.DataFrame(fpl_general_info_json['teams'])

teams_code_mapping = teams_df[['id','name']]

teams_code_mapping = teams_code_mapping.rename(columns={"name":"team","id":"team_id"})

##### Building base player metadata

element_types_df = pd.DataFrame(fpl_general_info_json['element_types'])

fpl_position_points_df = element_types_df[['id','singular_name_short']]

fpl_position_points_df = fpl_position_points_df.rename(columns={"id":"pos_id","singular_name_short":"pos"})

elements_df = pd.DataFrame(fpl_general_info_json['elements'])

# elements_df

# elements_df.columns

##### Building base for FPL points based off positional information

player_details_df = elements_df[['element_type','id','first_name','second_name','web_name','team']]
player_details_df = player_details_df.rename(columns={'element_type':'position_id',
                                            'id':'player_id',
                                            'team':'team_code'
                                            })

player_details_df['fpl_cleansheet_position_points'] = np.where(player_details_df['position_id'] == 1,4,
                                                               np.where(player_details_df['position_id'] == 2,4,
                                                                        np.where(player_details_df['position_id'] == 3,1,0)))

player_details_df['fpl_goals_position_points'] = np.where(player_details_df['position_id'] == 1,6,
                                                               np.where(player_details_df['position_id'] == 2,6,
                                                                        np.where(player_details_df['position_id'] == 3,5,4)))

##### Now loading individual players season history for game-by-game df.
### Probably need to build something which doesn't use a manual hard stop for element_summary_id and goes until there is no output

### Output Meaning
# Fixtures is future current season.
# History is previous current season.
# History Past is previous seasons

element_summary_id = 1

player_stats_game_by_game_url = 'https://fantasy.premierleague.com/api/element-summary/'+str(element_summary_id)+'/'
player_stats_game_by_game_r = requests.get(player_stats_game_by_game_url)
player_stats_game_by_game_json = player_stats_game_by_game_r.json()
player_game_stats_loop = pd.DataFrame(player_stats_game_by_game_json['history'])

element_summary_id = 0

for element_summary_id in range(1,652):
  element_summary_id = element_summary_id +1
  player_stats_game_by_game_player_url = 'https://fantasy.premierleague.com/api/element-summary/'+str(element_summary_id)+'/'
  player_stats_game_by_game_player_r = requests.get(player_stats_game_by_game_player_url)
  player_stats_game_by_game_player_json = player_stats_game_by_game_player_r.json()
  player_stats_game_by_game_player_json.keys()
  player_game_stats_player = pd.DataFrame(player_stats_game_by_game_player_json['history'])
  player_game_stats_loop = pd.concat([player_game_stats_loop, player_game_stats_player],ignore_index=True)

##### Building out individual game analysis output
### FPL is one a Gameweek basis (potential multiple games) though so will need aggregating later on.

player_game_stats = player_game_stats_loop.drop_duplicates()

player_game_stats = player_game_stats[['element','fixture','opponent_team','total_points','was_home','round','minutes','goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
       'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards',
       'red_cards', 'saves', 'bonus', 'bps','starts','value','expected_goals', 'expected_assists','expected_goals_conceded','defensive_contribution']]

player_game_stats = player_game_stats.rename(columns={"fixture":"fixture_id",
                                                      "opponent_team":"opponent_team_id",
                                                      "element":"player_code",
                                                      "total_points":"round_points"})

player_game_stats = player_game_stats.merge(teams_code_mapping, how='left',left_on ='opponent_team_id',right_on='team_id')

player_game_stats = player_game_stats.rename(columns={"team":"opposition_team"})

player_game_stats = player_game_stats[['player_code', 'fixture_id', 'opponent_team_id', 'round_points',
       'was_home', 'round', 'minutes', 'goals_scored', 'assists',
       'clean_sheets', 'goals_conceded', 'own_goals', 'penalties_saved',
       'penalties_missed', 'yellow_cards', 'red_cards', 'saves', 'bonus',
       'bps', 'starts', 'value', 'expected_goals', 'expected_assists',
       'expected_goals_conceded', 'opposition_team','defensive_contribution']]

player_game_stats = player_game_stats.merge(player_details_df,how='left',left_on='player_code',right_on='player_id')

player_game_stats = player_game_stats.merge(teams_code_mapping, how='left',left_on ='team_code',right_on='team_id')

player_game_stats['position'] = np.where(player_game_stats['position_id'] == 1,'GK',
                                         np.where(player_game_stats['position_id'] == 2,'DEF',
                                                  np.where(player_game_stats['position_id'] == 3,'MID',
                                                           np.where(player_game_stats['position_id'] == 4,'FWD','Fail'))))

player_game_stats['value'] = player_game_stats['value']*100000

player_game_stats['expected_goals'] = pd.to_numeric(player_game_stats['expected_goals'])
player_game_stats['expected_assists'] = pd.to_numeric(player_game_stats['expected_assists'])
player_game_stats['expected_goals_conceded'] = pd.to_numeric(player_game_stats['expected_goals_conceded'])

player_game_stats['home_or_away'] = np.where(player_game_stats['was_home'] == True,'H','A')

player_game_stats['fixture'] = player_game_stats['opposition_team']+' ('+player_game_stats['home_or_away']+')'

##### Building an Expected FPL Points Model for historical analysis
#### Split by Position (GK/DEF/MID/FWD)
### Expected Goals (6|6|5|4)
### Expected Assists (3|3|3|3)
### Expected Cleansheets (4|4|1|0)
### Expected Goals Conceded Per 2 goals (-1|-1|0|0)
### Minutes are the same
### Penalties Saved are zero
### Penalties Missed are zero
### Yellow cards is the actual FPL Points output
### Red Cards is the actual FPL Points output
### Own Goals are zero
### Saves is the actual FPL Points output
### Bonus is the top 3 xFPL Points scorers per Game
### Defcon, 10 for DEF. 12 for MID/FOR

# Goals
# player_game_stats['expected_fpl_points_goals'] = player_game_stats['expected_goals']*4
player_game_stats['expected_fpl_points_goals'] = player_game_stats['expected_goals']*player_game_stats['fpl_goals_position_points']
# Assists
player_game_stats['expected_fpl_points_assists'] = player_game_stats['expected_assists']*3
# Minutes
player_game_stats['fpl_points_minutes'] = np.where(player_game_stats['starts'] == 1,2,
                                                    np.where(player_game_stats['minutes'] > 0,1,0))


# Minutes
player_game_stats['fpl_points_expected_minutes'] = np.where(player_game_stats['minutes'] > 45,2,
                                                    np.where(player_game_stats['minutes'] > 0,1,0))


# # Goals Conceded
player_game_stats['expected_goals_conceded_rounded'] = round(player_game_stats['expected_goals_conceded'])
player_game_stats['expected_fpl_points_goals_conceded'] = np.where(player_game_stats['position'] == 'GK',np.floor(player_game_stats['expected_goals_conceded_rounded']/2),
                                                                   np.where(player_game_stats['position'] == 'DEF',np.floor(player_game_stats['expected_goals_conceded_rounded']/2),0))

# # Cleansheets
player_game_stats['expected_fpl_cleansheet_eligible'] = np.logical_and(player_game_stats['expected_goals_conceded_rounded'] == 0,
                                                                       player_game_stats['minutes'] > 60)

player_game_stats['expected_fpl_points_cleansheet'] = np.where(player_game_stats['expected_fpl_cleansheet_eligible'] == True, player_game_stats['fpl_cleansheet_position_points'],0)

# DEFCON

player_game_stats['defcon_fpl_pos_required'] = np.where(player_game_stats['position'] == 'DEF',10,12)

player_game_stats['defcon_fpl_points'] = np.where(player_game_stats['defensive_contribution'] >= player_game_stats['defcon_fpl_pos_required'],2,0)

# EXPECTED FPL POINTS
player_game_stats['expected_fpl_points'] = round((player_game_stats['expected_fpl_points_goals']
                                             +player_game_stats['expected_fpl_points_assists']
                                             +player_game_stats['fpl_points_expected_minutes']
                                             +(np.floor(player_game_stats['saves']/3))
                                             +player_game_stats['expected_fpl_points_cleansheet']
                                            #  +player_game_stats['bonus']
                                             -player_game_stats['expected_fpl_points_goals_conceded']
                                             -(player_game_stats['yellow_cards'])
                                             -(player_game_stats['red_cards']*3)
                                             +(player_game_stats['defcon_fpl_points'])
                                             ),2)

player_game_stats['expected_fpl_bonus'] = (player_game_stats.groupby('fixture_id')['expected_fpl_points'].rank(method='min', ascending=False).map({1: 3, 2: 2, 3: 1}).fillna(0).astype(int))

player_game_stats['expected_fpl_points'] = player_game_stats['expected_fpl_points']+player_game_stats['expected_fpl_bonus']

player_game_stats['fpl_points_actual_vs_expected'] = player_game_stats['round_points']-player_game_stats['expected_fpl_points']

player_game_stats = player_game_stats[['player_code','first_name','web_name','second_name','position','team','fixture','value','round','round_points','minutes','goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
       'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards',
       'red_cards', 'saves'
       , 'defensive_contribution'
       ,'bonus'
       , 'expected_goals'
       , 'expected_assists'
       , 'expected_fpl_points_goals_conceded'
       , 'expected_fpl_bonus'
       ,'expected_fpl_points'
       , 'fpl_points_actual_vs_expected']]

##### Cleaning Data for Grouping (sum), so fixtures and value needs additional cleaning

player_round_stats_added_info = player_game_stats.groupby(['round','player_code']).agg({'fixture':' / '.join}).reset_index()

player_round_stats_added_info = player_round_stats_added_info.rename(columns={'fixture':'fixture(s)'})

player_round_stats_added_info_value = player_game_stats[['round','player_code','value']].groupby(['round','player_code']).max('value').reset_index()

# player_round_stats_added_info_value

player_round_stats_added_info = player_round_stats_added_info.merge(player_round_stats_added_info_value,how='left',left_on=['round','player_code'],right_on=['round','player_code'])

# player_round_stats_added_info

##### Converting Game data into Gameweek Data (aka rounds)

player_round_stats = player_game_stats[['player_code','first_name','web_name','second_name','position','team','round','round_points','minutes','goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
       'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards',
       'red_cards', 'saves'
       , 'defensive_contribution'
       ,'bonus'
       , 'expected_goals'
       , 'expected_assists'
       , 'expected_fpl_points_goals_conceded'
       , 'expected_fpl_bonus'
       ,'expected_fpl_points'
       , 'fpl_points_actual_vs_expected']]

### Outside of metadata, everything should be summed.

player_round_stats = player_round_stats.groupby(['player_code','first_name','web_name','second_name','position','team','round']).sum().reset_index()

player_round_stats['expected_fpl_points'] = round(player_round_stats['expected_fpl_points'],2)

player_round_stats = player_round_stats.merge(player_round_stats_added_info,how='left',left_on=['round','player_code'],right_on=['round','player_code'])

player_round_stats = player_round_stats[['player_code','first_name','web_name','second_name','position','team','round','value','fixture(s)','round_points','minutes','goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
       'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards',
       'red_cards', 'saves'
       ,'bonus'
       , 'defensive_contribution'
       , 'expected_goals'
       , 'expected_assists'
       , 'expected_fpl_points_goals_conceded'
       , 'expected_fpl_bonus'
       ,'expected_fpl_points'
       , 'fpl_points_actual_vs_expected']]

player_round_stats = player_round_stats.sort_values(by=['round_points','expected_fpl_points','value'],ascending=[False,False,True])

pd.set_option('display.max_columns', None)
pd.reset_option('display.max_rows')

# player_round_stats

# display(player_round_stats)
