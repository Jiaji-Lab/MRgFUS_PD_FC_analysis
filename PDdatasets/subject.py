"""
Subject management module

    A subect that contains multiple observation accross time,
    each observation contains multimodal images including:
        1. T1
        2. DTI
        3. BOLD
        4. ASL
        5. ESWAN
    Perform operations including read, write and feature extraction
    subject dir structure example (load observation from center_info.csv):
        subject01_observation01
            |---t1
            |---dti
            |---asl
        subject01_observation02
            |---t1
            |---dti
            |---asl
    Attributes:
        name: string, name for the subject.
        observations: list of Observation, holds observations for this subject.
    Functions:
        load_personal_info: load personal info like age, sex, etc.
        load_observation: load all observations for this subject.
        get_label: get subject label.
        get_personal_info: return a dict of personal info.
        get_observation: get observation with certain name.

"""
import ast
import os
import csv

import nibabel as nib
import nilearn as nil
import numpy as np
import pandas as pd

from . import modal

def load_subjects(info_path):
    info_df = pd.read_csv(info_path, index_col=['subject', 'observation'])#取出文件的index
    subject_names = set(info_df.index.get_level_values(0)) # set不重复+有顺序，这里有两层index，get_level_values取第一个，就是subjects的信息
    subjects = [] #这个是关键
    for subject_name in subject_names:
        observations = info_df.loc[subject_name,:] #取片
        observations_dict = {}#建立一个字典
        for index, row in observations.iterrows():#iterrow是dataFrame的遍历--成对每行，由左侧index带出row，变成index series和row series
            observations_dict[str(index)] = row #将series的东西，存成dict，这个地方是index（key）+row series（value）
        subject = Subject(subject_name, observations_dict)
        subjects.append(subject)
    return subjects
    
class Subject(object):
    def __init__(self, name,
                 observations_dict):
        """
        Args:
            name: string, name for the subject.
            observations_dict: dict of observations, {name1:args1, name2:args2}
        """
        self.name = name
        self.observations = []
        self.load_observation(observations_dict)
        self.load_personal_info()
    
    def load_observation(self, observations_dict):
        self.observations = []
        for observation_name, observation_args in observations_dict.items():#item是dict的遍历-成对，由左侧index带出row series，进行row series的遍历，由上侧的index来带出值
            observation = Observation(observation_name, observation_args)#这里的结果是我们的series数据，argument; columne name 是固定得到的
            self.observations.append(observation)#存成dict
    
    def load_personal_info(self):
        observation = self.get_observation('base')
        self.lesion_side = observation.lesion_side

    def get_label(self):
        if not self.label:
            self.load_personal_info()
        return self.label

    def get_personal_info(self):
        pass

    def get_observation(self, observation_name):
        for observation in self.observations:
            if observation.name == observation_name:
                return observation
    
    def get_all_observation(self):
        return self.observations

class Observation(object):
    def __init__(self, name, args):
        self.name = name
        self.args = args#将series保留在其属性中
        
        self.directory = args['dir']#单独定义的常用的两个函数
        self.lesion_side = args['lesion_side']

        self.t1 = modal.T1(os.path.join(self.directory, 'T1_post')) #这里直接确定每个模态所在是路径
        self.dti = modal.Dti(os.path.join(self.directory, 'DTI_post'))
        self.dti_s = modal.Dti(os.path.join(self.directory, 'DTI_post_S200'))
        self.dti_sb = modal.Dti(os.path.join(self.directory, 'DTI_post_BN246'))
        self.asl = modal.Asl(os.path.join(self.directory, 'ASL'))
        self.bold = modal.Modal(os.path.join(self.directory, 'BOLD_post'), mark='Bold')

    def get_lesion_side(self):
        return self.lesion_side