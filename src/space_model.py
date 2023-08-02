import numpy as np
import pandas as pd
import math
import torch
from collections import Counter
import warnings
from scipy.stats import multivariate_normal
from pandas.errors import SettingWithCopyWarning
import matplotlib.pyplot as plt 
from matplotlib.patches import Ellipse
from matplotlib.ticker import FormatStrFormatter
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from src.nn_model import nnModel
from src.plot_utils import *
from src.Influence_function import *
from matplotlib import animation
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)


class space_model():
    def __init__(self, df, e_df):
        self.df = self.preprocess(df, e_df)
        self.velocities = None
        self.positions = None
        self.points = None
        self.players = None
        self.ball_x = None
        self.ball_y = None
        self.set_frame_flag = False

    def set_frame(self, frame): 
        t_df = self.df[self.df['Frame'] == frame]
        self.period = t_df['Period'].values[0]
        t_df = t_df.drop(['Period', 'Ball_x', 'Ball_y', 'Ball_v_abs'], axis=1).iloc[0,:]
        self.positions = t_df[[i for i in t_df.index if (('_x' in i) or ('_y' in i)) and 'v' not in i]]
        self.positions.dropna(inplace=True)
        self.velocities = t_df[[i for i in t_df.index if '_v' in i]]
        self.velocities.dropna(inplace=True)
        self.points = np.array([[self.positions[2*i], self.positions[2*i+1]] for i in range(len(self.positions)//2)])
        self.velocities = np.array([[self.velocities[3*i], self.velocities[3*i+1]] for i in range(len(self.velocities)//3)])
        self.players = np.array([self.positions.index[2*i].split('_')[0] for i in range(len(self.points))])
        self.ball_x, self.ball_y = self.df.loc[self.df['Frame'] == frame, ['Ball_x', 'Ball_y']].values[0]

        if math.isnan(self.ball_x) :
            self.set_frame_flag = False
            return 
        else :
            self.set_frame_flag = True
            return
        
    def pitch_control(self, locations):
        s_h, s_a = 0, 0
        if not self.set_frame_flag:
            print('need to set frame')
            return
        for i, j, k in zip(self.players, self.points, self.velocities):
            if 'H' in i:
                s_h += influence_function2(j, locations, k, (self.ball_x, self.ball_y))
            else :
                s_a += influence_function2(j, locations, k, (self.ball_x, self.ball_y))
                
        z = 1 / (1 + np.exp(- s_h + s_a))
        return z
    
    
    def space_value(self, locations):
        nn_model = nnModel()
        nn_model.load_state_dict(torch.load('./SpaceValueModel/best_svmodel_sdict.pt'))

        x = locations.reshape(-1, 2)[:,0].reshape(-1, 1) /104
        y = locations.reshape(-1, 2)[:,1].reshape(-1, 1) /68
    
        ball_x = self.ball_x * np.ones_like(x) / 104
        ball_y = self.ball_y * np.ones_like(y) / 68

        input = torch.FloatTensor(np.concatenate([ball_x, ball_y, x, y], axis=1))
        output = nn_model(input)
        output = output.detach().numpy()

        z = self.distance_from_goal(x*104, y*68)
        output = output.reshape(-1) * z.reshape(-1) 

        return output
    
    def space_quality(self, attacking_direction):
        if not self.set_frame_flag:
            print('need to set frame')
            return
        
        space_quality = {}

        for i,j in zip(self.players, self.points):
            pc = self.pitch_control(j)
            if i.startswith('A'):
                pc = 1 - pc
                if attacking_direction == 1:
                    sv = self.space_value(np.array([104 - j[0], j[1]]))
                else :
                    sv = self.space_value(j)
            else :
                if attacking_direction == 1:
                    sv = self.space_value(j)
                else :
                    sv = self.space_value(np.array([104 - j[0], j[1]]))
            key = i
            key = key + '_sq'
            space_quality[key] = float(pc*sv)

        return space_quality


    def distance_from_goal(self, x, y):
        dist = np.sqrt((104-x)**2 + (34-y)**2)
        max_v = np.sqrt(104**2 + 34**2)
        return (max_v - dist)/ (max_v)


    def preprocess(self, df, e_df):
        t_df = e_df[['Team', 'Type', 'Start Frame', 'End Frame', 'Period']]
        t_df = pd.concat([t_df, t_df.shift(-1).rename(columns={'Team': 'Next Team', 
                                                            'Type' : 'Next Type',
                                                            'Start Frame' : 'Next Start Frame',
                                                            'End Frame' : 'Next End Frame',
                                                            'Period' : 'Next Period'})], axis=1)
        t_df = t_df[(t_df['Start Frame'] < t_df['Next End Frame']) &
            (t_df['Team'] == t_df['Next Team']) & 
            (t_df['Period'] == t_df['Next Period']) &
            ((t_df['Type'] == 'PASS') | (t_df['Type'] == 'RECOVERY'))]
        t_df.reset_index(drop=True, inplace=True)

        n_list = np.zeros(len(df))
        for i in range(len(t_df)):
            s_frame, f_frame = np.array(t_df.loc[i, ['Start Frame', 'Next End Frame']])
            if math.isnan(s_frame*f_frame) == False:
                if t_df.loc[i, 'Team'] == 'Home':
                    n_list[int(s_frame) :int(f_frame)] = 1
                if t_df.loc[i, 'Team'] == 'Away':
                    n_list[int(s_frame):int(f_frame)] = 2
        for i in range(len(t_df)):
            if t_df.loc[i,'Type'] == 'RECOVERY':
                s_frame = t_df.loc[i, 'Start Frame']
                n_list[int(s_frame-25):int(s_frame + 25*5)] = 0
        df['owning'] = n_list
        return df



