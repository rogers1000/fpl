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

##### Reading CSV for player stats


##### Team Selection Analysis
### Need to track teams GW-by-GW scores as a basis

### Manual Changes that need to be accounted for at this point.
current_game_week = 4

### First of Loop first

game_week_first_loop = 1

fpl_team_id = '2658055'

fpl_team_selection_info_url = 'https://fantasy.premierleague.com/api/entry/'+str(fpl_team_id)+'/event/'+str(game_week_first_loop)+'/picks/'

fpl_team_selection_info_r = requests.get(fpl_team_selection_info_url)

fpl_team_selection_info_json = fpl_team_selection_info_r.json()

team_selection_df_loop = pd.DataFrame(fpl_team_selection_info_json['picks'])

team_selection_df_loop['GW'] = game_week_first_loop

team_selection_df_loop['fpl_team_id'] = fpl_team_id

team_selection_df_loop['scenario'] = 'GW'+ str(game_week_first_loop) +' Starting 11'

for fpl_team_id in fpl_team_id_list:

  for game_week in range(1,current_game_week + 1):

      print(f"Processing team {fpl_team_id}, GW {game_week}")

      fpl_team_selection_info_url = 'https://fantasy.premierleague.com/api/entry/'+str(fpl_team_id)+'/event/'+str(game_week)+'/picks/'

      try:

        fpl_team_selection_info_r = requests.get(fpl_team_selection_info_url)

        fpl_team_selection_info_json = fpl_team_selection_info_r.json()

        team_selection_df = pd.DataFrame(fpl_team_selection_info_json['picks'])

        team_selection_df['GW'] = game_week

        team_selection_df['fpl_team_id'] = fpl_team_id

        team_selection_df['scenario'] = 'GW'+ str(game_week) +' Starting 11'

        team_selection_df_loop = pd.concat([team_selection_df_loop, team_selection_df],ignore_index=True)

      except Exception as e:
        print(e)




team_selection_df_loop = team_selection_df_loop.rename(columns={'position':'position_key',
                                                      'element':'player_code'})

team_selection_df_loop = team_selection_df_loop.drop_duplicates()

team_selection_df_loop = team_selection_df_loop[['player_code','multiplier','GW','position_key','fpl_team_id','scenario']]

# team_selection_df_loop

##### Scenario Analysis for manual scenarios.
### There is a github csv which has the template for starting 11 scenarios and transfer scenarios.
## Recommend to include whole squad in case of 0 minutes.
## I could update the template to include multiple teams but for the moment it's just my own team I have done scenarios for.

scenario_url = "https://raw.githubusercontent.com/rogers1000/fpl/refs/heads/main/FPL_2627_Starting_scenarios.csv"

scenario_df = pd.read_csv(scenario_url)

scenario_df = scenario_df.rename(columns={'Player_Code':'player_code',
                                          'Scenario':'scenario'})

scenario_df['fpl_team_id'] = '5891056'

scenario_df = scenario_df[['player_code','multiplier','GW','position_key','fpl_team_id','scenario']]

team_selection_df_loop = pd.concat([team_selection_df_loop, scenario_df],ignore_index=True)

# team_selection_df_loop

##### Adding key information to the team scenarios
### Adds the round analysis data to the team selections
### Updates round points for captains/bench using multiplier
### Creates an Expected Captain
## Using most expensive player. Will want to update to account for double game weeks.

team_selection_info = team_selection_df_loop

team_selection_info = team_selection_info[['fpl_team_id','GW','player_code','position_key','multiplier','scenario']]

team_selection_info = team_selection_info.merge(player_round_stats[['player_code','round','first_name','web_name','position','team','value','fixture(s)','round_points','expected_fpl_points']], how = 'left', left_on = ['player_code'], right_on = ['player_code'])

team_selection_info = team_selection_info[team_selection_info['GW'] <= team_selection_info['round']]

team_selection_info = team_selection_info[team_selection_info['round'] <= current_game_week]

team_selection_info['round_points'] = team_selection_info['round_points']*team_selection_info['multiplier']

team_selection_info = team_selection_info.sort_values(["fpl_team_id", "GW", "value"],ascending=[True, True, False]
)

team_selection_info["expected_captain"] = (
    team_selection_info
    .groupby(["fpl_team_id", "GW"])["value"]
    .transform(lambda x: np.where(x == x.max(), 2, 1))
)

team_selection_info['expected_fpl_points'] = team_selection_info['expected_fpl_points']*team_selection_info['multiplier']

team_selection_info['fpl_points_actual_vs_expected'] = round(team_selection_info['round_points']-team_selection_info['expected_fpl_points'],2)

team_selection_info = team_selection_info.sort_values(by = ['GW','round','position_key'], ascending = [True,True,True])

team_selection_info = team_selection_info.reset_index()

##### Building an expected starting 11 model. Expected model is the highest value starting 11 available to the team in a week
#### NGL this was done using a combo of myself and Chatgpt.
### Need to expand python knowledge to be able to rebuild as I'm sure this isn't as efficent as it could be.
### Need to look into expanding it for double game weeks.


formations = [
    (3, 4, 3),
    (3, 5, 2),
    (4, 3, 3),
    (4, 4, 2),
    (4, 5, 1),
    (5, 2, 3),
    (5, 3, 2),
    (5, 4, 1),
]


def find_best_formation(group):

    players = group.drop_duplicates("player_code").copy()

    gk = players[players["position"] == "GK"]
    defenders = players[players["position"] == "DEF"]
    midfielders = players[players["position"] == "MID"]
    forwards = players[players["position"] == "FWD"]

    best_value = -float("inf")
    best_formation = None
    best_players = None
    best_bench = None

    for d, m, f in formations:

        # Check if this formation is possible
        if (
            len(gk) >= 1
            and len(defenders) >= d
            and len(midfielders) >= m
            and len(forwards) >= f
        ):

            selected = pd.concat([
                gk.nlargest(1, "value"),
                defenders.nlargest(d, "value"),
                midfielders.nlargest(m, "value"),
                forwards.nlargest(f, "value")
            ])

            total_value = selected["value"].sum()

            if total_value > best_value:

                best_value = total_value
                best_formation = f"{d}-{m}-{f}"
                best_players = selected["player_code"].tolist()

                # --------------------------------------------------
                # BENCH
                # --------------------------------------------------

                # Players not selected in the starting XI
                remaining = players[
                    ~players["player_code"].isin(best_players)
                ].copy()

                # -----------------------------
                # OUTFIELD BENCH
                # -----------------------------

                outfield_bench = remaining[
                    remaining["position"].isin(["DEF", "MID", "FWD"])
                ].copy()

                # Rank ALL outfield players purely by value.
                # Position does NOT affect the ranking.
                outfield_bench = (
                    outfield_bench
                    .sort_values("value", ascending=False)
                    .reset_index(drop=True)
                )

                # Only the first 3 are the outfield substitutes
                outfield_bench = outfield_bench.head(3).copy()

                # FPL-style sub positions 1, 2, 3
                outfield_bench["sub_position"] = range(
                    1, len(outfield_bench) + 1
                )

                # -----------------------------
                # GOALKEEPER BENCH
                # -----------------------------

                # Remaining goalkeeper(s), ranked separately by value
                bench_gk = remaining[
                    remaining["position"] == "GK"
                ].copy()

                bench_gk = (
                    bench_gk
                    .sort_values("value", ascending=False)
                    .reset_index(drop=True)
                )

                # Take the best remaining GK
                bench_gk = bench_gk.head(1).copy()

                # GK is always sub_position 4
                bench_gk["sub_position"] = 4

                # -----------------------------
                # COMBINE BENCH
                # -----------------------------

                best_bench = pd.concat(
                    [
                        outfield_bench,
                        bench_gk
                    ],
                    ignore_index=True
                )

                # Ensure correct bench order
                best_bench = best_bench.sort_values(
                    "sub_position"
                ).reset_index(drop=True)

    return pd.Series({
        "best_formation": best_formation,
        "total_value": best_value,
        "players": best_players,

        # Bench player codes in sub_position order
        "bench_players": (
            best_bench["player_code"].tolist()
            if best_bench is not None else []
        ),

        # Useful if you want the actual sub-position mapping
        "bench_sub_positions": (
            best_bench["sub_position"].tolist()
            if best_bench is not None else []
        )
    })


expected_starters_list = (
    team_selection_info
    .groupby(["fpl_team_id", "round", "GW", "scenario"])
    .apply(find_best_formation)
    .reset_index()
    .sort_values(
        by=["total_value"],
        ascending=False
    )
)

# expected_starters_list

##### Build logic for bench player replacements for those that played 0 minutes.
### Expand to account for 2 starters not playing
## This is probably just a loop function for starters

bench_subs = expected_starters_list.explode('players')

bench_subs = bench_subs.rename(columns={'players':'player_code'})

bench_subs = bench_subs.explode(['bench_players','bench_sub_positions'])

bench_subs = bench_subs.merge(player_round_stats[['player_code','round','first_name','web_name','position','minutes']], how = 'left', left_on = ['player_code','round'], right_on = ['player_code','round'])

bench_subs = bench_subs.merge(player_round_stats[['player_code','round','first_name','web_name','position','minutes']], how = 'left', left_on = ['bench_players','round'], right_on = ['player_code','round'],suffixes = ['_starters','_bench'])

bench_subs = bench_subs[bench_subs['minutes_starters'] == 0]

bench_subs = bench_subs[bench_subs['minutes_bench'] != 0]

bench_subs['sub_same_position'] = np.where(bench_subs['position_starters'] == bench_subs['position_bench'],True,False)

bench_subs['sub_gk_position'] = np.where(((bench_subs['position_starters'] == 'GK') & (bench_subs['position_bench'] == 'GK')) | (bench_subs['position_starters'] != 'GK'),True,False)

bench_subs['starters_def_min_count'] = np.where((bench_subs['best_formation'].str[:1] == 3) & (bench_subs['position_bench'] != 'DEF'),False,True)

bench_subs['starters_mid_min_count'] = np.where((bench_subs['best_formation'].str[2:3] == 2) & (bench_subs['position_bench'] != 'MID'),False,True)
bench_subs['starters_FWD_min_count'] = np.where((bench_subs['best_formation'].str[4:5] == 1) & (bench_subs['position_bench'] != 'FWD'),False,True)

bench_subs['legal_sub'] = np.where((['sub_same_position']) & (bench_subs['sub_gk_position']) & (bench_subs['starters_def_min_count']) & (bench_subs['starters_mid_min_count']) & (bench_subs['starters_FWD_min_count']), True, False)


bench_subs = bench_subs[bench_subs['GW'] <= bench_subs['round']]

bench_subs['scenario'] = np.where('Expected Starting 11 for '+ bench_subs['scenario'] == 'Expected Starting 11 for Starting 11','Expected Starting 11 for GW Squad','Expected Starting 11 for '+ bench_subs['scenario'])


bench_subs = bench_subs[bench_subs['legal_sub'] == True]

bench_subs['bench_position_selected'] = bench_subs.groupby(['fpl_team_id','round','player_code_starters'])['bench_sub_positions'].transform(min)

bench_subs = bench_subs[bench_subs['bench_position_selected'] == bench_subs['bench_sub_positions']]

bench_subs = bench_subs[['fpl_team_id','round','GW','scenario','player_code_starters','bench_players']]

# bench_subs

##### Using info from Expected Starters and Bench Replacements to get an actual Expected Starting 11 for each team
### Brings in GW data to get their FPL scores too
### Expected Captain based off highest value starter

expected_starters = expected_starters_list.explode('players').reset_index()

expected_starters = expected_starters.rename(columns={'players':'player_code'})

expected_starters = expected_starters[expected_starters['GW'] <= expected_starters['round']]

expected_starters['scenario'] = np.where('Expected Starting 11 for '+ expected_starters['scenario'] == 'Expected Starting 11 for Starting 11','Expected Starting 11 for GW Squad','Expected Starting 11 for '+ expected_starters['scenario'])

expected_starters = expected_starters.merge(bench_subs, how = 'left', left_on = ['fpl_team_id','round','GW','scenario','player_code'],right_on = ['fpl_team_id','round','GW','scenario','player_code_starters'],suffixes = ['_starters','_bench'])

expected_starters['player_code'] = expected_starters['bench_players_bench'].combine_first(expected_starters['player_code'])


expected_starters = expected_starters[['fpl_team_id', 'round', 'GW', 'scenario','player_code',
       'total_value']]

expected_starters = expected_starters.merge(player_round_stats[['player_code','round','first_name','web_name','position','team','value','fixture(s)','minutes','round_points','expected_fpl_points']], how = 'left', left_on = ['player_code','round'], right_on = ['player_code','round'])


expected_starters["multiplier"] = (
    expected_starters
    .groupby(["fpl_team_id", "GW"])["value"]
    .transform(lambda x: np.where(x == x.max(), 2, 1))
)

expected_starters['round_points'] = expected_starters['round_points']*expected_starters['multiplier']

expected_starters['expected_fpl_points'] = expected_starters['expected_fpl_points']*expected_starters['multiplier']

expected_starters['fpl_points_actual_vs_expected'] = round(expected_starters['round_points']-expected_starters['expected_fpl_points'],2)


# expected_starters

# bench_subs

##### Team Selection Analysis, metadata so IDs and Names

fpl_team_id_list

fpl_team_meta_info_name_list = []

fpl_team_meta_info_player_name_list = []


team_count = 0

for fpl_team_id in fpl_team_id_list:

  fpl_team_meta_info_url = 'https://fantasy.premierleague.com/api/entry/'+str(fpl_team_id_list[team_count])+'/'

  fpl_team_meta_info_r = requests.get(fpl_team_meta_info_url)

  fpl_team_meta_info_json = fpl_team_meta_info_r.json()

  fpl_team_meta_info_name_list = fpl_team_meta_info_name_list+[fpl_team_meta_info_json['name']]

  fpl_team_meta_info_player_name_list = fpl_team_meta_info_player_name_list+[
    fpl_team_meta_info_json['player_first_name'] + ' ' +
    fpl_team_meta_info_json['player_last_name']]

  team_count = team_count + 1

fpl_team_meta_info_table = pd.DataFrame({
    'fpl_team_id': fpl_team_id_list,
    'fpl_team_name': fpl_team_meta_info_name_list,
    'fpl_player_name': fpl_team_meta_info_player_name_list
})

##### Chip Usage Analysis - Bench Boost.
### All players play

chip_bench_boost = team_selection_df_loop

# team_selection_info = team_selection_info[team_selection_info['multiplier'] > 0]

chip_bench_boost = chip_bench_boost[['fpl_team_id','GW','player_code','position_key','multiplier','scenario']]
chip_bench_boost['scenario'] = chip_bench_boost['scenario']+' - Chip (BB)'

chip_bench_boost = chip_bench_boost.merge(player_round_stats[['player_code','round','first_name','web_name','position','team','value','fixture(s)','round_points','expected_fpl_points']], how = 'left', left_on = ['player_code'], right_on = ['player_code'])

chip_bench_boost = chip_bench_boost[chip_bench_boost['GW'] <= chip_bench_boost['round']]

chip_bench_boost = chip_bench_boost[chip_bench_boost['round'] <= current_game_week]

chip_bench_boost['multiplier'] = np.where(chip_bench_boost['multiplier'] == 0,1,chip_bench_boost['multiplier'])
chip_bench_boost['multiplier'] = np.where(chip_bench_boost['multiplier'] == 3,2,chip_bench_boost['multiplier'])


chip_bench_boost['round_points'] = chip_bench_boost['round_points']*chip_bench_boost['multiplier']

chip_bench_boost = chip_bench_boost.sort_values(
    ["fpl_team_id", "GW", "value"],
    ascending=[True, True, False]
)

chip_bench_boost["expected_captain"] = (
    chip_bench_boost
    .groupby(["fpl_team_id", "GW"])["value"]
    .transform(lambda x: np.where(x == x.max(), 2, 1))
)

chip_bench_boost['expected_fpl_points'] = chip_bench_boost['expected_fpl_points']*chip_bench_boost['multiplier']

chip_bench_boost['fpl_points_actual_vs_expected'] = round(chip_bench_boost['round_points']-chip_bench_boost['expected_fpl_points'],2)

chip_bench_boost = chip_bench_boost.sort_values(by = ['GW','round','position_key'], ascending = [True,True,True])

chip_bench_boost = chip_bench_boost.reset_index()

# chip_bench_boost = chip_bench_boost['scenario'].unique()

# chip_bench_boost

##### Chip Usage Analysis - Triple Captain.
### Captain has multiplier x3 rather than x2
### Need to expand to not have it show TC + BB for the same week

chip_triple_captain = team_selection_df_loop

# team_selection_info = team_selection_info[team_selection_info['multiplier'] > 0]

chip_triple_captain = chip_triple_captain[['fpl_team_id','GW','player_code','position_key','multiplier','scenario']]
chip_triple_captain['scenario'] = chip_triple_captain['scenario']+' - Chip (TC)'

chip_triple_captain = chip_triple_captain.merge(player_round_stats[['player_code','round','first_name','web_name','position','team','value','fixture(s)','round_points','expected_fpl_points']], how = 'left', left_on = ['player_code'], right_on = ['player_code'])

chip_triple_captain = chip_triple_captain[chip_triple_captain['GW'] <= chip_triple_captain['round']]

chip_triple_captain = chip_triple_captain[chip_triple_captain['round'] <= current_game_week]

chip_triple_captain['multiplier'] = np.where(chip_triple_captain['multiplier'] == 2,3,chip_triple_captain['multiplier'])
# chip_triple_captain['multiplier'] = np.where(chip_triple_captain['multiplier'] == 3,2,chip_triple_captain['multiplier'])


chip_triple_captain['round_points'] = chip_triple_captain['round_points']*chip_triple_captain['multiplier']

chip_triple_captain = chip_triple_captain.sort_values(
    ["fpl_team_id", "GW", "value"],
    ascending=[True, True, False]
)

chip_triple_captain["expected_captain"] = (
    chip_triple_captain
    .groupby(["fpl_team_id", "GW"])["value"]
    .transform(lambda x: np.where(x == x.max(), 3, 1))
)

chip_triple_captain['expected_fpl_points'] = chip_triple_captain['expected_fpl_points']*chip_triple_captain['multiplier']

chip_triple_captain['fpl_points_actual_vs_expected'] = round(chip_triple_captain['round_points']-chip_triple_captain['expected_fpl_points'],2)

chip_triple_captain = chip_triple_captain.sort_values(by = ['GW','round','position_key'], ascending = [True,True,True])

chip_triple_captain = chip_triple_captain.reset_index()

# chip_triple_captain = chip_triple_captain['multiplier'].unique()

# chip_triple_captain

##### Bringing all different scenarios into one table
### This is on a team-player level

team_selection_grouped_gran = team_selection_info[['fpl_team_id','round','GW','player_code','first_name','web_name','position','team','value','fixture(s)','multiplier','scenario',"round_points", "expected_fpl_points", "fpl_points_actual_vs_expected"]]

team_selection_grouped_gran = team_selection_grouped_gran[team_selection_grouped_gran['multiplier'] > 0].reset_index()

expected_starters = expected_starters[['fpl_team_id','GW','round','player_code','first_name','web_name','position','team','value','fixture(s)','multiplier','scenario','round_points', 'expected_fpl_points', 'fpl_points_actual_vs_expected']]

team_selection_grouped_gran = pd.concat([team_selection_grouped_gran, expected_starters,chip_bench_boost,chip_triple_captain],ignore_index=True)

team_selection_grouped_gran = team_selection_grouped_gran.merge(fpl_team_meta_info_table, how = 'left', left_on = 'fpl_team_id', right_on = 'fpl_team_id')


team_selection_grouped_gran['expected_fpl_points'] = round(team_selection_grouped_gran['expected_fpl_points'],2)

team_selection_grouped_gran['fpl_points_actual_vs_expected'] = round(team_selection_grouped_gran['fpl_points_actual_vs_expected'],2)

##### Aggregation of Team GW Level
### Team level for each scenario

team_selection_grouped_agg = team_selection_grouped_gran.groupby(['fpl_team_id','fpl_player_name','fpl_team_name','scenario','GW','round'])[["round_points", "expected_fpl_points", "fpl_points_actual_vs_expected"]].sum().reset_index()

# team_selection_grouped_agg['expected_fpl_points'] = round(team_selection_grouped_agg['expected_fpl_points'],2)

team_selection_grouped_agg['fpl_points_actual_vs_expected'] = round(team_selection_grouped_agg['fpl_points_actual_vs_expected'],2)

team_selection_grouped_agg = team_selection_grouped_agg.sort_values(by = ['fpl_team_id','round','GW','scenario'], ascending = [True,True,True,False]).reset_index()

# team_selection_grouped_agg = team_selection_grouped_agg[team_selection_grouped_agg['fpl_team_id' == '2658055']]

team_selection_grouped_agg

##### Creating a baseline for each team
### Uses actual 11 scores for the baseline so for the graphs the changes show from the moment the manager did them rather than start at 0 points.

team_selection_grouped_baseline = team_selection_grouped_agg

team_selection_grouped_baseline['actual_11_scenario'] = np.where((team_selection_grouped_baseline['scenario'] == 'GW'+team_selection_grouped_baseline['GW'].astype(str)+' Starting 11') & (team_selection_grouped_baseline['GW'].astype(str) == team_selection_grouped_baseline['round'].astype(str)),True,False)

team_selection_grouped_baseline = team_selection_grouped_baseline[team_selection_grouped_baseline['actual_11_scenario'] == True]

team_selection_grouped_baseline = team_selection_grouped_baseline.merge(team_selection_grouped_agg[['fpl_team_id','GW','scenario']], how = 'left', left_on = ['fpl_team_id'], right_on = ['fpl_team_id'], suffixes = ['','_scenario'])

team_selection_grouped_baseline = team_selection_grouped_baseline[team_selection_grouped_baseline['GW'] < team_selection_grouped_baseline['GW_scenario']]

team_selection_grouped_baseline['scenario'] = team_selection_grouped_baseline['scenario_scenario']

team_selection_grouped_baseline = team_selection_grouped_baseline[['index', 'fpl_team_id', 'fpl_player_name', 'fpl_team_name', 'scenario',
       'GW', 'round', 'round_points', 'expected_fpl_points',
       'fpl_points_actual_vs_expected'
                                                                   ]]

team_selection_grouped_baseline = team_selection_grouped_baseline.drop_duplicates()

# team_selection_grouped_baseline

##### Creating an actual team score as a single scenario

team_selection_grouped_actual = team_selection_grouped_agg

team_selection_grouped_actual['actual_11_scenario'] = np.where((team_selection_grouped_actual['scenario'] == 'GW'+team_selection_grouped_actual['GW'].astype(str)+' Starting 11') & (team_selection_grouped_actual['GW'].astype(str) == team_selection_grouped_actual['round'].astype(str)),True,False)

team_selection_grouped_actual = team_selection_grouped_actual[team_selection_grouped_actual['actual_11_scenario'] == True]

team_selection_grouped_actual['scenario'] = 'Actual FPL Selection'

# team_selection_grouped_actual

##### Putting everything together for visualisations

team_selection_grouped_agg_graph = pd.concat([team_selection_grouped_agg, team_selection_grouped_baseline,team_selection_grouped_actual],ignore_index=True)

team_selection_grouped_agg_graph = team_selection_grouped_agg_graph.sort_values(by=['fpl_team_id', 'round']).reset_index(drop=True)


team_selection_grouped_agg_graph['expected_fpl_points'] = round(team_selection_grouped_agg_graph['expected_fpl_points'],2)

team_selection_grouped_agg_graph['fpl_points_actual_vs_expected'] = round(team_selection_grouped_agg_graph['fpl_points_actual_vs_expected'],2)

team_selection_grouped_agg_graph['scenario_points_running_sum'] = np.where(team_selection_grouped_agg_graph['GW'] <= team_selection_grouped_agg_graph['round'], team_selection_grouped_agg_graph['round_points'],0)
team_selection_grouped_agg_graph['scenario_expected_points_running_sum'] = np.where(team_selection_grouped_agg_graph['GW'] <= team_selection_grouped_agg_graph['round'], team_selection_grouped_agg_graph['expected_fpl_points'],0)
team_selection_grouped_agg_graph['scenario_actual_vs_expected_points_running_sum'] = np.where(team_selection_grouped_agg_graph['GW'] <= team_selection_grouped_agg_graph['round'], team_selection_grouped_agg_graph['fpl_points_actual_vs_expected'],0)

team_selection_grouped_agg_graph['scenario_points_running_sum'] = team_selection_grouped_agg_graph.groupby(['fpl_team_id','fpl_player_name','fpl_team_name','scenario'])['scenario_points_running_sum'].cumsum()
team_selection_grouped_agg_graph['scenario_expected_points_running_sum'] = team_selection_grouped_agg_graph.groupby(['fpl_team_id','fpl_player_name','fpl_team_name','scenario'])['scenario_expected_points_running_sum'].cumsum()
team_selection_grouped_agg_graph['scenario_actual_vs_expected_points_running_sum'] = team_selection_grouped_agg_graph.groupby(['fpl_team_id','fpl_player_name','fpl_team_name','scenario'])['scenario_actual_vs_expected_points_running_sum'].cumsum()

team_selection_grouped_agg_graph['scenario_expected_points_running_sum'] = round(team_selection_grouped_agg_graph['scenario_expected_points_running_sum'],2)

team_selection_grouped_agg_graph['scenario_actual_vs_expected_points_running_sum'] = round(team_selection_grouped_agg_graph['scenario_actual_vs_expected_points_running_sum'],2)

team_selection_grouped_agg_graph['graph_splitter'] = 'GW'+(team_selection_grouped_agg_graph['GW']).astype(str) +' | '+ team_selection_grouped_agg_graph['scenario']

# team_selection_grouped_agg_graph = team_selection_grouped_agg_graph[team_selection_grouped_agg_graph['fpl_player_name'] == 'Zac Rogers']

# team_selection_grouped_agg_graph = team_selection_grouped_agg_graph[team_selection_grouped_agg_graph['scenario'] == 'Actual FPL Selection']

team_selection_grouped_agg_graph = team_selection_grouped_agg_graph[team_selection_grouped_agg_graph['GW'] <= team_selection_grouped_agg_graph['round']]

team_selection_grouped_agg_graph

player_season_stats = player_round_stats_testing

player_season_stats = player_season_stats.groupby(['player_code','first_name','web_name','second_name','player_name','position','team']).sum()

player_season_stats = player_season_stats.sort_values(by =['round_points','expected_fpl_points','minutes'],ascending = [False,False,False]).reset_index()

player_season_stats
