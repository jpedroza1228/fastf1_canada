import pandas as pd
import numpy as np
import plotnine as pn
from janitor import clean_names
import seaborn as sns
import matplotlib.pyplot as plt
from great_tables import GT
import plotly.express as px
import fastf1
from matplotlib import rcParams

pd.set_option('mode.copy_on_write', True)
pd.set_option('display.max_columns', None)
rcParams.update({'savefig.bbox': 'tight'})

fastf1.Cache.enable_cache('/home/jon/Documents/github_repos/fastf1_canada')

can_22_fp1 = fastf1.get_session(2022, 'Canada', 'FP1')
can_22_fp2 = fastf1.get_session(2022, 'Canada', 'FP2')
can_22_fp3 = fastf1.get_session(2022, 'Canada', 'FP3')
can_22_q = fastf1.get_session(2022, 'Canada', 'Q')
can_22_r = fastf1.get_session(2022, 'Canada', 'R')

can_23_fp1 = fastf1.get_session(2023, 'Canada', 'FP1')
can_23_fp2 = fastf1.get_session(2023, 'Canada', 'FP2')
can_23_fp3 = fastf1.get_session(2023, 'Canada', 'FP3')
can_23_q = fastf1.get_session(2023, 'Canada', 'Q')
can_23_r = fastf1.get_session(2023, 'Canada', 'R')

can_24_fp1 = fastf1.get_session(2024, 'Canada', 'FP1')
can_24_fp2 = fastf1.get_session(2024, 'Canada', 'FP2')
can_24_fp3 = fastf1.get_session(2024, 'Canada', 'FP3')
can_24_q = fastf1.get_session(2024, 'Canada', 'Q')
can_24_r = fastf1.get_session(2024, 'Canada', 'R')

sessions = [
  can_22_fp1, can_22_fp2, can_22_fp3, can_22_q, can_22_r,
  can_23_fp1, can_23_fp2, can_23_fp3, can_23_q, can_23_r,
  can_24_fp1, can_24_fp2, can_24_fp3, can_24_q, can_24_r
  ]

for i in sessions:
  i.load()

can_track = can_22_fp1.get_circuit_info()

can_track.corners
can_track.marshal_lights
can_track.marshal_sectors
can_track.rotation

# CANADA 2022 FREE PRACTICE 1

# Lap Data


fp1_22_cardata_ver = can_22_fp1.car_data['1']
fp1_22_cardata_nor = can_22_fp1.car_data['4']
fp1_22_cardata_ver['Driver'] = 'VER'
fp1_22_cardata_nor['Driver'] = 'NOR'
fp1_22_cardata = pd.concat([fp1_22_cardata_ver, fp1_22_cardata_nor])

fp1_22_laps = can_22_fp1.laps.loc[can_22_fp1.laps['Driver'].isin(['VER', 'NOR'])]

fp1_22_weather = can_22_fp1.weather_data

fp1_22_pos_ver = can_22_fp1.pos_data['1']
fp1_22_pos_nor = can_22_fp1.pos_data['4']
fp1_22_pos_ver['Driver'] = 'VER'
fp1_22_pos_nor['Driver'] = 'NOR'
fp1_22_pos = pd.concat([fp1_22_pos_ver, fp1_22_pos_nor])

can_22_fp1.session_status

can_22_fp1.track_status

def rotate(xy, *, angle):
    """Rotate a DataFrame or Series of x/y points."""
    rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])
    # xy can be a DataFrame (n, 2) or Series/list (2,)
    if isinstance(xy, pd.DataFrame):
        return xy @ rot_mat
    else:  # assume it's Series or list-like with 2 elements
        return pd.Series(xy) @ rot_mat

fp1_pos
track = fp1_22_pos_ver.loc[:, ('X', 'Y')]

track_angle = can_track.rotation / 180 * np.pi

rotated_track = rotate(track, angle = track_angle)

offset_vector = pd.Series([500, 0]) 

rotated_track = rotated_track.loc[(rotated_track[0] != 0) & (rotated_track[1] != 0)]
rotated_track['x'] = rotated_track[0]
rotated_track['y'] = rotated_track[1]
rotated_track = rotated_track.drop(columns = [0, 1])

# track visual
pn.ggplot.show(
  pn.ggplot(rotated_track, pn.aes('x', 'y'))
  + pn.geom_path()
)

# fp1_22_laps['LapStartDate'] = fp1_22_laps['LapStartDate'].dt.round('10ms')
# fp1_22_laps['Time'] = fp1_22_laps['Time'].dt.round('10ms')


#Date = LapStartDate
#Time = 

# LapStartDate

# all dataframes created
fp1_22_cardata.head()
fp1_22_laps.head()
fp1_22_weather.head()
fp1_22_pos.head()

fp1_22_pos['Date'] = fp1_22_pos.loc[:, 'Date'].dt.round('100ms')
fp1_22_pos['Time'] = fp1_22_pos.loc[:, 'Time'].dt.round('100ms')
fp1_22_pos['SessionTime'] = fp1_22_pos.loc[:, 'SessionTime'].dt.round('100ms')

fp1_22_cardata['Date'] = fp1_22_cardata.loc[:, 'Date'].dt.round('100ms')
fp1_22_cardata['Time'] = fp1_22_cardata.loc[:, 'Time'].dt.round('100ms')
fp1_22_cardata['SessionTime'] = fp1_22_cardata.loc[:, 'SessionTime'].dt.round('100ms')

fp1_22_carpos = fp1_22_pos.merge(fp1_22_cardata, 'inner', ['Date', 'Time', 'SessionTime', 'Driver'])

fp1_22_carpos.head()

fp1_22_carpos = fp1_22_carpos.loc[(fp1_22_carpos['X'] != 0) & (fp1_22_carpos['Y'] != 0)]
fp1_22_carpos = fp1_22_carpos.loc[(fp1_22_carpos['nGear'] != 0)]

brake_region = (
  fp1_22_carpos.loc[fp1_22_carpos['Brake'] == True, ['X', 'Y', 'Driver']]
  .value_counts()
  .reset_index()
  .sort_values(['X', 'Y'], ascending = True)
)

from sklearn.cluster import KMeans

# ver_region = brake_region.loc[brake_region['Driver'] == 'VER']
# nor_region = brake_region.loc[brake_region['Driver'] == 'NOR']

# ver_region['region'] = kmeans.fit_predict(ver_region[['X', 'Y']])
# nor_region['region'] = kmeans.fit_predict(nor_region[['X', 'Y']])


# 14 spatial clusters
kmeans = KMeans(n_clusters = 17, n_init = 20, max_iter = 1000, random_state = 15062025)
brake_region['region'] = kmeans.fit_predict(brake_region[['X', 'Y']])

pn.ggplot.show(
  pn.ggplot(brake_region, pn.aes('X', 'Y', color = 'factor(region)'))
  + pn.geom_point()
  + pn.scale_color_manual(values = ['red', 'dodgerblue', 'darkgreen', 
  'gold', 'purple', 'cyan', 
  'orange', 'teal', 'limegreen', 
  'brown', 'black', 'gray', 
  'blue', 'pink', 'magenta',
  'navy', 'orchid'])
  + pn.theme_classic()
)

knn_reg = [
  (brake_region['region'] == 6),
  (brake_region['region'] == 15),
  (brake_region['region'] == 2),
  (brake_region['region'] == 9),
  (brake_region['region'] == 16),
  (brake_region['region'] == 5),
  (brake_region['region'] == 14),
  (brake_region['region'] == 0),
  (brake_region['region'] == 10),
  (brake_region['region'] == 8),
  (brake_region['region'] == 3),
  (brake_region['region'] == 11),
  (brake_region['region'] == 1),
  (brake_region['region'] == 12),
  (brake_region['region'] == 7),
  (brake_region['region'] == 13),
  (brake_region['region'] == 4)
]

turn_reg = np.arange(0, 17)

brake_region['region'] = np.select(knn_reg, turn_reg)

pn.ggplot.show(
  pn.ggplot(brake_region, pn.aes('X', 'Y', color = 'factor(region)'))
  + pn.geom_point()
  + pn.scale_color_manual(values = ['red', 'dodgerblue', 'darkgreen', 
  'gold', 'purple', 'cyan', 
  'orange', 'teal', 'limegreen', 
  'brown', 'black', 'gray', 
  'blue', 'pink', 'magenta',
  'navy', 'orchid'])
  + pn.theme_classic()
)

# brake_region = pd.concat([ver_region, nor_region]).drop(columns = 'count')

fp1_22_carpos = fp1_22_carpos.merge(brake_region, 'left', on = ['Driver', 'X', 'Y'])

fp1_22_carpos = fp1_22_carpos.drop(columns = 'count')

fp1_22_carpos.head()

fp1_22_carpos['region'] = pd.Categorical(fp1_22_carpos['region'], ordered = True)

fp1_22_carpos.head()


fp1_22_carpos.loc[fp1_22_carpos['Brake'] == True, ['region', 'X', 'Y', 'nGear', 'Driver']].groupby(['Driver', 'region']).last()
fp1_22_carpos.loc[fp1_22_carpos['Brake'] == True, ['X', 'Y', 'region', 'Brake']].groupby('region').last()

pn.ggplot.show(
  pn.ggplot(fp1_22_carpos, pn.aes('X', 'Y'))
  + pn.geom_path(alpha = .3)
  + pn.geom_point(fp1_22_carpos.loc[fp1_22_carpos['Brake'] == True, ['X', 'Y', 'region', 'Brake']].groupby('region').last(), pn.aes(shape = 'Brake'))
  + pn.facet_wrap('Driver')
  + pn.theme_classic()
)

fp1_22_laps.head()
fp1_22_weather.head()

fp1_22_weather['WeatherTime'] = fp1_22_weather.loc[:, 'Time'].dt.round('s')

fp1_22_laps['WeatherTime'] = fp1_22_laps.loc[:, 'Time'].dt.round('s')

fp1_22_laps.merge(fp1_22_weather, 'left', ['Time'])

fp1_22_laps.groupby('Driver')['LapNumber'].max().reset_index()


#max_pos = miami24q.pos_data['1']
#lando_pos = miami24q.pos_data['4']

# from sklearn.model_selection import train_test_split
# from sklearn.utils.class_weight import compute_class_weight
# from tensorflow.random import set_seed
# from sklearn.ensemble import RandomForestClassifier as rfc
# from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.inspection import permutation_importance
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import KFold, GridSearchCV