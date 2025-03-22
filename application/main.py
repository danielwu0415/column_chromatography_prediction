import scipy
from sklearn import linear_model
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import argparse
import  os
import pymysql
import pandas as pd
from rdkit import Chem
import numpy as np
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw
import mordred.CPSA as CPSA
import mordred
from mordred import Calculator, descriptors,is_missing
from sklearn.metrics import mean_squared_error
from xgboost.sklearn import XGBClassifier,XGBRegressor
from xgboost import plot_importance
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
from rdkit.Chem import MACCSkeys
from PIL import Image
import torch.nn as nn
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import time
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import heapq
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import pandas
from collections import deque
import numpy as np
import random
from utils import *
from QGeoGNN import QGeoGNN_transfer_reversed, calculate_separation_probability, predict_separate
random.seed(525)
np.random.seed(1101)
torch.manual_seed(324)
batch_size=64
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default=os.getcwd()+'\TLC_dataset.xlsx', help='path of download dataset')
    parser.add_argument('--dipole_path', type=str, default=os.getcwd() + '\compound_list_带化合物分类.xlsx',
                        help='path of dipole file')
    parser.add_argument('--data_range', type=int, default=4944, help='utilized data range,robot:4114,manual:4458,new:4944')
    parser.add_argument('--automatic_divide', type=bool, default=False, help='automatically divide dataset by 80% train,10% validate and 10% test')
    parser.add_argument('--choose_total', type=int, default=387, help='train total num,robot:387,manual:530')
    parser.add_argument('--choose_train', type=int, default=308, help='train num,robot:387,manual:530')
    parser.add_argument('--choose_validate', type=int, default=38, help='validate num')
    parser.add_argument('--choose_test', type=int, default=38, help='test num')
    parser.add_argument('--seed', type=int, default=324, help='random seed for split dataset，best is 1101')
    parser.add_argument('--torch_seed', type=int, default=324, help='random seed for torch')
    parser.add_argument('--add_dipole', type=bool, default=False, help='add dipole into dataset')
    parser.add_argument('--add_molecular_descriptors', type=bool, default=True, help='add molecular_descriptors (分子量(MW)、拓扑极性表面积(TPSA)、可旋转键的个数(NROTB)、氢键供体个数(HBA)、氢键受体个数(HBD)、脂水分配系数值(LogP)) into dataset')
    parser.add_argument('--add_eluent_matrix', type=bool, default=True,help='add eluent matrix into dataset')
    parser.add_argument('--test_mode', type=str, default='split_by_data', help='manual data or robot data or fix, costum test data')
    parser.add_argument('--use_model', type=str, default='Ensemble',help='the utilized model (XGB,LGB,ANN,RF,Ensemble,Bayesian)')
    parser.add_argument('--download_data', type=bool, default=False, help='use local dataset or download from dataset')
    parser.add_argument('--use_sigmoid', type=bool, default=True, help='use sigmoid')
    parser.add_argument('--shuffle_array', type=bool, default=True, help='shuffle_array')
    #---------------parapmeters for plot---------------------
    parser.add_argument('--plot_col_num', type=int, default=4, help='The col_num in plot')
    parser.add_argument('--plot_row_num', type=int, default=4, help='The row_num in plot')
    parser.add_argument('--plot_importance_num', type=int, default=10, help='The max importance num in plot')
    #--------------parameters For LGB-------------------
    parser.add_argument('--LGB_max_depth', type=int, default=5, help='max_depth for LGB')
    parser.add_argument('--LGB_num_leaves', type=int, default=25, help='num_leaves for LGB')
    parser.add_argument('--LGB_learning_rate', type=float, default=0.007, help='learning_rate for LGB')
    parser.add_argument('--LGB_n_estimators', type=int, default=1000, help='n_estimators for LGB')
    parser.add_argument('--LGB_early_stopping_rounds', type=int, default=200, help='early_stopping_rounds for LGB')

    #---------------parameters for XGB-----------------------
    parser.add_argument('--XGB_n_estimators', type=int, default=200, help='n_estimators for XGB')
    parser.add_argument('--XGB_max_depth', type=int, default=3, help='max_depth for XGB')
    parser.add_argument('--XGB_learning_rate', type=float, default=0.1, help='learning_rate for XGB')

    #---------------parameters for RF------------------------
    parser.add_argument('--RF_n_estimators', type=int, default=1000, help='n_estimators for RF')
    parser.add_argument('--RF_random_state', type=int, default=1, help='random_state for RF')
    parser.add_argument('--RF_n_jobs', type=int, default=1, help='n_jobs for RF')

    #--------------parameters for ANN-----------------------
    parser.add_argument('--NN_hidden_neuron', type=int, default=128, help='hidden neurons for NN')
    parser.add_argument('--NN_optimizer', type=str, default='Adam', help='optimizer for NN (Adam,SGD,RMSprop)')
    parser.add_argument('--NN_lr', type=float, default=0.001, help='learning rate for NN')
    parser.add_argument('--NN_model_save_location', type=str, default=os.getcwd()+'\model_save_NN', help='learning rate for NN')
    parser.add_argument('--NN_max_epoch', type=int, default=20000, help='max training epoch for NN')
    parser.add_argument('--NN_add_sigmoid', type=bool, default=True, help='whether add sigmoid in NN')
    parser.add_argument('--NN_add_PINN', type=bool, default=False, help='whether add PINN in NN')
    parser.add_argument('--NN_epi', type=float, default=100.0, help='The coef of PINN Loss in NN')



    config = parser.parse_args()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.costum_array=[314,378]
    return config

class Dataset_process():
    '''
    For processing the data and split the dataset
    '''
    def __init__(self,config):
        super(Dataset_process, self).__init__()
        self.file_path=config.file_path
        self.dipole_path=config.dipole_path
        self.data_range=config.data_range
        self.choose_train=config.choose_train
        self.choose_validate=config.choose_validate
        self.choose_test=config.choose_test
        self.automatic_divide=config.automatic_divide
        self.seed=config.seed
        self.add_dipole=config.add_dipole
        self.add_molecular_descriptors=config.add_molecular_descriptors
        self.add_eluent_matrix = config.add_eluent_matrix
        self.test_mode=config.test_mode
        self.download_data = config.download_data
        self.shuffle_array=config.shuffle_array
        self.costum_array=config.costum_array
        self.characterization_mode=''
        self.Eluent_smiles=['CCCCCC','CC(OCC)=O','C(Cl)Cl','CO','CCOCC']

    def download_dataset(self,print_info=True):
        '''
        Download the dataset from mysql dataset
        :param print_info: whether print the download information
        :return: None
        '''
        dbconn = pymysql.connect(
            host='bj-cdb-k8stylt6.sql.tencentcdb.com',
            port=60474,
            user='xuhao',
            password='xuhao1101',
            database='TLC',
            charset='utf8',
        )

        # SQL query
        sqlcmd = "select * from tb_TLC"

        # Load MySQL data using pandas module
        a = pd.read_sql(sqlcmd, dbconn)

        a.to_excel(self.file_path)
        if print_info==True:
            print(f'Dataset has been downloaded, the file path is :{self.file_path}')

    def get_descriptor(self,smiles,ratio):
        compound_mol = Chem.MolFromSmiles(smiles)
        descriptor=[]
        descriptor.append(Descriptors.ExactMolWt(compound_mol))
        descriptor.append(Chem.rdMolDescriptors.CalcTPSA(compound_mol))
        descriptor.append(Descriptors.NumRotatableBonds(compound_mol))  # Number of rotable bonds
        descriptor.append(Descriptors.NumHDonors(compound_mol))  # Number of H bond donors
        descriptor.append(Descriptors.NumHAcceptors(compound_mol)) # Number of H bond acceptors
        descriptor.append(Descriptors.MolLogP(compound_mol)) # LogP
        descriptor=np.array(descriptor)*ratio
        return descriptor

    def get_eluent_descriptor(self,eluent_array):
        eluent=eluent_array
        des = np.zeros([6,])
        for i in range(eluent.shape[0]):
            if eluent[i] != 0:
                e_descriptors = Dataset_process.get_descriptor(self, self.Eluent_smiles[i], eluent[i])
                des+=e_descriptors
        return des

    def get_3D_conformer(self):
        data_range = self.data_range
        if self.download_data == True:
            Dataset_process.download_dataset(self)
        entire_info = (pd.read_excel(self.file_path, index_col=None, na_values=['NA']).fillna(-1))
        compound_info = entire_info.values[0:data_range]
        compound_ID = compound_info[:, 2]
        compound_smile = compound_info[:, 3]


        # Convert to molecule object and generate 167-dimensional vector
        compound_mol = compound_smile.copy()

        use_index = 0
        for i in tqdm(range(len(compound_smile))):
            compound_mol[i] = Chem.MolFromSmiles(compound_smile[i])
            try:
                Finger = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(compound_smile[i]))
            except Exception as e:
                print(f'the compound {compound_ID[i]} has no MACCkeys.')
                continue
            mol = AllChem.MolFromSmiles(compound_smile[i])
            new_mol = Chem.AddHs(mol)
            res = AllChem.EmbedMultipleConfs(new_mol)
            ### MMFF generates multiple conformations
            res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
            new_mol = Chem.RemoveHs(new_mol)
            Chem.MolToMolFile(new_mol, f'3D_conform/data_{i}.mol')

    def create_dataset(self,data_array,choose_num,compound_ID,dipole_ID,compound_Rf,compound_finger,compound_eluent,dipole_moment,
                       compound_MolWt,compound_TPSA,compound_nRotB,compound_HBD,compound_HBA,compound_LogP):
        '''
        create training/validate/test dataset
        add or not the molecular_descriptors and dipole moments can be controlled
        '''
        y = []
        database_finger = np.zeros([1, 167])
        database_eluent = np.zeros([1, 6])
        database_dipole = np.zeros([1, 1])
        database_descriptor = np.zeros([1, 6])
        for i in range(choose_num):
            index = int(data_array[i])
            ID_loc = np.where(compound_ID == index)[0]
            dipole_loc = np.where(dipole_ID == index)[0]
            for j in ID_loc:
                y.append([compound_Rf[j]])
                database_finger=np.vstack((database_finger,compound_finger[j]))
                database_eluent=np.vstack((database_eluent,compound_eluent[j]))
                if self.add_dipole==True:
                    database_dipole=np.vstack((database_dipole,dipole_moment[dipole_loc]))
                database_descriptor=np.vstack((database_descriptor,np.array([compound_MolWt[j],compound_TPSA[j],compound_nRotB[j],compound_HBD[j],compound_HBA[j],compound_LogP[j]]).reshape([1,6])))


        X=database_finger.copy()

        if self.add_molecular_descriptors == True:
            X = np.hstack((X, database_descriptor))

        X=np.hstack((X,database_eluent))

        if self.add_dipole==True:
            X=np.hstack((X,database_dipole))

        X=np.hstack((X,np.zeros([X.shape[0],3])))


        X = np.delete(X, [0], axis=0)
        y = np.array(y)
        return X,y

    def delete_invalid(self,database, h):
        '''
        delete invalid data which is filled with -1 when reading the dataset
        '''
        delete_row_h = np.where(h == -1)[0]
        if delete_row_h.size > 0:
            database = np.delete(database, delete_row_h, axis=0)
            h = np.delete(h, delete_row_h, axis=0)

        delete_row_data = np.where(database == -1)[0]
        if delete_row_data.size > 0:
            database = np.delete(database, delete_row_data, axis=0)
            h = np.delete(h, delete_row_data, axis=0)
        return database,h

    def plot_compound(self,target_ID=-1):
        data_range = self.data_range
        if self.download_data == True:
            Dataset_process.download_dataset(self)
        entire_info = (pd.read_excel(self.file_path, index_col=None, na_values=['NA']).fillna(-1))
        compound_info = entire_info.values[0:data_range]
        compound_ID = compound_info[:, 2]
        compound_smile = compound_info[:, 3]
        compound_name=compound_info[:,10]
        compound_list = np.unique(compound_ID)
        compound_num = compound_list.shape[0]
        # data_array = compound_list.copy()
        # np.random.seed(self.seed)
        # np.random.shuffle(data_array)

        #----------------single plotting-----------------
        # index = target_ID
        # ID_loc = np.where(compound_ID == index)[0][0]
        # smile=compound_smile[ID_loc]
        # mol= Chem.MolFromSmiles(smile)
        # smiles_pic = Draw.MolToImage(mol, size=(500, 500),dpi=300, kekulize=True)
        # plt.figure(20,figsize=(0.5,0.5),dpi=300)
        # plt.imshow(smiles_pic)
        # plt.axis('off')
        # plt.savefig(f'fig_save/compound_{index}.tiff',dpi=300)
        # plt.savefig(f'fig_save/compound_{index}.pdf', dpi=300)
        # plt.show()


        #------------Overall plotting-----------------
        if target_ID==-1:
            plt.figure(10,figsize=(7,10),dpi=300)
            num=0
            for i in range(350,384):
                index=compound_list[i]
                ID_loc = np.where(compound_ID == index)[0][0]
                smile=compound_smile[ID_loc]
                mol= Chem.MolFromSmiles(smile)
                smiles_pic = Draw.MolToImage(mol, size=(200, 100), kekulize=True)
                plt.subplot(10,7,num+1)
                #plt.title(index)
                plt.imshow(smiles_pic)
                plt.axis('off')
                num+=1
            plt.savefig(f'fig_save/new/compound_{350}~{384}.tiff',dpi=300)
            plt.savefig(f'fig_save/new/compound_{350}~{384}.pdf',dpi=300)
            plt.show()

    def split_dataset(self):
        '''
        split the dataset according to the train/validate/test num
        :return: X_train,y_train,X_validate,y_validate,X_test,y_test,data_array(shuffled compounds)
        '''
        data_range=self.data_range
        if self.download_data==True:
            Dataset_process.download_dataset(self)
        entire_info = (pd.read_excel(self.file_path, index_col=None, na_values=['NA']).fillna(-1))
        compound_info=entire_info.values[0:data_range]
        compound_ID = compound_info[:, 2]
        compound_smile = compound_info[:, 3]
        compound_Rf = compound_info[:, 9]
        eluent = np.array(compound_info[:, 4:9],dtype=np.float32)
        compound_eluent=[]
        for j in range(eluent.shape[0]):
            des=Dataset_process.get_eluent_descriptor(self,eluent[j])
            compound_eluent.append(des.tolist())
        compound_eluent=np.array(compound_eluent)


        if self.add_eluent_matrix==False:
            Eluent_PE = compound_info[:, 4].copy()
            Eluent_EA = compound_info[:, 4].copy()
            for i in range(len(compound_eluent)):
                a = int(compound_eluent[i].split('-', )[1].split('_', )[1])
                b = int(compound_eluent[i].split('-', )[0].split('_', )[1])
                Eluent_PE[i] = b
                Eluent_EA[i] = a
                compound_eluent[i] = a / (a + b)

        # Convert to mol and generate 167-dimensional vector
        compound_mol = compound_smile.copy()
        compound_finger = np.zeros([len(compound_smile), 167])
        compound_MolWt = np.zeros([len(compound_smile), 1])
        compound_TPSA = np.zeros([len(compound_smile), 1])
        compound_nRotB = np.zeros([len(compound_smile), 1])
        compound_HBD = np.zeros([len(compound_smile), 1])
        compound_HBA = np.zeros([len(compound_smile), 1])
        compound_LogP = np.zeros([len(compound_smile), 1])
        compound_ID_new = np.zeros([len(compound_smile), 1])
        compound_Rf_new = np.zeros([len(compound_smile), 1])
        compound_eluent_new = np.zeros([len(compound_smile), 6])

        use_index=0
        for i in tqdm(range(len(compound_smile))):
            compound_mol[i] = Chem.MolFromSmiles(compound_smile[i])
            try:
                Finger = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(compound_smile[i]))
            except Exception as e:
                print(f'the compound {compound_ID[i]} has no MACCkeys.')
                continue
            fingerprint = np.array([x for x in Finger])
            compound_finger[use_index] = fingerprint
            compound_MolWt[use_index] = Descriptors.ExactMolWt(compound_mol[i])
            if self.characterization_mode=='precise_TPSA':
                mol_conform=Chem.MolFromMolFile(f"3D_conform/data_{i}.mol")
                compound_TPSA[use_index] = mordred.CPSA.RPSA()(mol_conform)
            else:
                compound_TPSA[use_index] = Chem.rdMolDescriptors.CalcTPSA(compound_mol[i])
            compound_nRotB[use_index] = Descriptors.NumRotatableBonds(compound_mol[i])  # Number of rotable bonds
            compound_HBD[use_index] = Descriptors.NumHDonors(compound_mol[i])  # Number of H bond donors
            compound_HBA[use_index] = Descriptors.NumHAcceptors(compound_mol[i])  # Number of H bond acceptors
            compound_LogP[use_index] = Descriptors.MolLogP(compound_mol[i])  # LogP
            compound_ID_new[use_index]=compound_ID[i]
            compound_Rf_new[use_index]=compound_Rf[i]
            compound_eluent_new[use_index]=compound_eluent[i]
            use_index+=1


        compound_ID=compound_ID_new[0:use_index]
        compound_Rf=compound_Rf_new[0:use_index].reshape(compound_ID.shape[0],)
        compound_finger=compound_finger[0:use_index]
        compound_eluent=compound_eluent_new[0:use_index]
        compound_MolWt=compound_MolWt[0:use_index]
        compound_TPSA=compound_TPSA[0:use_index]
        compound_nRotB=compound_nRotB[0:use_index]
        compound_HBD=compound_HBD[0:use_index]
        compound_HBA=compound_HBA[0:use_index]
        compound_LogP=compound_LogP[0:use_index]

        # 读取偶极矩文件
        if self.add_dipole==True:
            dipole_info = (pd.read_excel(self.dipole_path, index_col=None, na_values=['NA']).fillna(-1)).values
            dipole_ID = dipole_info[:, 0]
            dipole_moment = dipole_info[:, 11]
        else:
            dipole_ID = None
            dipole_moment = None

        # 计算化合物的个数
        compound_list = np.unique(compound_ID)
        compound_num = compound_list.shape[0]
        # print(compound_num)
        if self.automatic_divide==True:
            self.choose_train=math.floor(0.8*compound_num)
            self.choose_validate=math.floor(0.1*compound_num)
            self.choose_test = math.floor(0.1 * compound_num)
        # print(self.choose_train,self.choose_validate,self.choose_test)
        if self.choose_train+self.choose_validate+self.choose_test>compound_num:
            raise ValueError(f'Out of compound num, which is {compound_num}')
        data_array = compound_list.copy()
        if self.shuffle_array==True:
            np.random.seed(self.seed)
            np.random.shuffle(data_array)

        X_train,y_train=Dataset_process.create_dataset(self,data_array[0:self.choose_train],self.choose_train,compound_ID, dipole_ID, compound_Rf, compound_finger,
                       compound_eluent, dipole_moment,compound_MolWt, compound_TPSA, compound_nRotB, compound_HBD, compound_HBA, compound_LogP)
        X_validate, y_validate = Dataset_process.create_dataset(self, data_array[self.choose_train:self.choose_train+self.choose_validate], self.choose_validate,
                                                          compound_ID, dipole_ID, compound_Rf, compound_finger,
                                                          compound_eluent, dipole_moment, compound_MolWt, compound_TPSA,
                                                          compound_nRotB, compound_HBD, compound_HBA, compound_LogP)
        if self.test_mode=='robot':
            X_test, y_test=Dataset_process.create_dataset(self, data_array[self.choose_train+self.choose_validate:self.choose_train+self.choose_validate+self.choose_test], self.choose_test,
                                                          compound_ID, dipole_ID, compound_Rf, compound_finger,
                                                          compound_eluent, dipole_moment, compound_MolWt, compound_TPSA,
                                                          compound_nRotB, compound_HBD, compound_HBA, compound_LogP)

        elif self.test_mode=='fix':
            X_test, y_test=Dataset_process.create_dataset(self, data_array[-self.choose_test-1:-1], self.choose_test,
                                                          compound_ID, dipole_ID, compound_Rf, compound_finger,
                                                          compound_eluent, dipole_moment, compound_MolWt, compound_TPSA,
                                                          compound_nRotB, compound_HBD, compound_HBA, compound_LogP)

        elif self.test_mode=='costum':
            X_test, y_test = Dataset_process.create_dataset(self, self.costum_array,
                                                            len(self.costum_array),
                                                            compound_ID, dipole_ID, compound_Rf, compound_finger,
                                                            compound_eluent, dipole_moment, compound_MolWt,
                                                            compound_TPSA,
                                                            compound_nRotB, compound_HBD, compound_HBA, compound_LogP)





        X_train,y_train=Dataset_process.delete_invalid(self,X_train,y_train)
        X_validate, y_validate = Dataset_process.delete_invalid(self, X_validate, y_validate)
        X_test,y_test=Dataset_process.delete_invalid(self, X_test, y_test)


        return X_train,y_train,X_validate,y_validate,X_test,y_test,data_array

    def split_dataset_all(self):
        '''
        split the dataset according to the TLC_num
        :return: X
        '''
        data_range = self.data_range
        if self.download_data == True:
            Dataset_process.download_dataset(self)
        entire_info = (pd.read_excel(self.file_path, index_col=None, na_values=['NA']).fillna(-1))
        compound_info = entire_info.values[0:data_range]
        compound_ID = compound_info[:, 2]
        compound_smile = compound_info[:, 3]
        compound_Rf = compound_info[:, 9]
        eluent = np.array(compound_info[:, 4:9], dtype=np.float32)
        compound_eluent = []
        for j in range(eluent.shape[0]):
            des = Dataset_process.get_eluent_descriptor(self, eluent[j])
            compound_eluent.append(des.tolist())
        compound_eluent = np.array(compound_eluent)
        if self.add_eluent_matrix == False:
            Eluent_PE = compound_info[:, 4].copy()
            Eluent_EA = compound_info[:, 4].copy()
            for i in range(len(compound_eluent)):
                a = int(compound_eluent[i].split('-', )[1].split('_', )[1])
                b = int(compound_eluent[i].split('-', )[0].split('_', )[1])
                Eluent_PE[i] = b
                Eluent_EA[i] = a
                compound_eluent[i] = a / (a + b)

        # Convert to mol and generate 167-dimensional vector
        compound_mol = compound_smile.copy()
        compound_finger = np.zeros([len(compound_smile), 167])
        compound_MolWt = np.zeros([len(compound_smile), 1])
        compound_TPSA = np.zeros([len(compound_smile), 1])
        compound_nRotB = np.zeros([len(compound_smile), 1])
        compound_HBD = np.zeros([len(compound_smile), 1])
        compound_HBA = np.zeros([len(compound_smile), 1])
        compound_LogP = np.zeros([len(compound_smile), 1])
        compound_ID_new = np.zeros([len(compound_smile), 1])
        compound_Rf_new = np.zeros([len(compound_smile), 1])
        compound_eluent_new = np.zeros([len(compound_smile), 6])

        use_index = 0
        for i in range(len(compound_smile)):
            compound_mol[i] = Chem.MolFromSmiles(compound_smile[i])
            try:
                Finger = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(compound_smile[i]))
            except Exception as e:
                print(f'the compound {compound_ID[i]} has no MACCkeys.')
                continue
            fingerprint = np.array([x for x in Finger])
            compound_finger[use_index] = fingerprint
            compound_MolWt[use_index] = Descriptors.ExactMolWt(compound_mol[i])
            if self.characterization_mode=='precise_TPSA':
                mol_conform=Chem.MolFromMolFile(f"3D_conform/data_{i}.mol")
                compound_TPSA[use_index] = mordred.CPSA.TPSA()(mol_conform)
            else:
                compound_TPSA[use_index] = Chem.rdMolDescriptors.CalcTPSA(compound_mol[i])
            compound_nRotB[use_index] = Descriptors.NumRotatableBonds(compound_mol[i])  # Number of rotable bonds
            compound_HBD[use_index] = Descriptors.NumHDonors(compound_mol[i])  # Number of H bond donors
            compound_HBA[use_index] = Descriptors.NumHAcceptors(compound_mol[i])  # Number of H bond acceptors
            compound_LogP[use_index] = Descriptors.MolLogP(compound_mol[i])  # LogP
            compound_ID_new[use_index]=compound_ID[i]
            compound_Rf_new[use_index]=compound_Rf[i]
            compound_eluent_new[use_index]=compound_eluent[i]
            use_index += 1

        compound_ID = compound_ID_new[0:use_index]
        compound_Rf = compound_Rf_new[0:use_index].reshape(compound_ID.shape[0], )
        compound_finger = compound_finger[0:use_index]
        compound_eluent = compound_eluent_new[0:use_index]
        compound_MolWt = compound_MolWt[0:use_index]
        compound_TPSA = compound_TPSA[0:use_index]
        compound_nRotB = compound_nRotB[0:use_index]
        compound_HBD = compound_HBD[0:use_index]
        compound_HBA = compound_HBA[0:use_index]
        compound_LogP = compound_LogP[0:use_index]
        # Read dipole moment file
        if self.add_dipole == True:
            dipole_info = (pd.read_excel(self.dipole_path, index_col=None, na_values=['NA']).fillna(-1)).values
            dipole_ID = dipole_info[:, 0]
            dipole_moment = dipole_info[:, 11]
        else:
            dipole_ID = None
            dipole_moment = None

        y = []
        ID=[]
        database_finger = np.zeros([1, 167])
        database_eluent = np.zeros([1, 6])
        database_dipole = np.zeros([1, 1])
        database_descriptor = np.zeros([1, 6])
        for i in range(compound_finger.shape[0]):
            dipole_loc = np.where(dipole_ID == compound_ID[i])[0]
            y.append([compound_Rf[i]])
            ID.append([compound_ID[i]])
            database_finger = np.vstack((database_finger, compound_finger[i]))
            database_eluent = np.vstack((database_eluent, compound_eluent[i]))
            if self.add_dipole == True:
                database_dipole = np.vstack((database_dipole, dipole_moment[dipole_loc]))
            database_descriptor = np.vstack((database_descriptor, np.array(
                [compound_MolWt[i], compound_TPSA[i], compound_nRotB[i], compound_HBD[i], compound_HBA[i],
                 compound_LogP[i]]).reshape([1, 6])))

        if self.add_MACCkeys==True:
            X = database_finger.copy()
            X = np.hstack((X, database_eluent))
        else:
            X = database_eluent.copy()
        if self.add_dipole == True:
            X = np.hstack((X, database_dipole))
        if self.add_molecular_descriptors == True:
            X = np.hstack((X, database_descriptor))

        if self.characterization_mode=='no_multi':
            X = np.delete(X, [27,42,46,103], axis=1)
        X = np.delete(X, [0], axis=0)
        y = np.array(y)
        return X,y,ID

class ANN_CC(nn.Module):
    '''
    Construct artificial neural network
    '''

    def __init__(self, in_neuron, hidden_neuron, out_neuron):
        super(ANN_CC, self).__init__()
        self.input_layer = nn.Linear(in_neuron, hidden_neuron)
        self.hidden_layer = nn.Linear(hidden_neuron, hidden_neuron)
        self.output_layer = nn.Linear(hidden_neuron, out_neuron)

    def forward(self, x):
        #x = (x - x.min(dim=0)[0]) / (x.max(dim=0)[0] - x.min(dim=0)[0] + 1e-8)
        x = self.input_layer(x)
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = F.leaky_relu(x)
        x = self.output_layer(x)
        return x

def split_dataset(data):
    if 'X_CC.npy' not in os.listdir('dataset_save'):
        all_eluent_ratio = []
        all_descriptor=np.load('dataset_save/all_descriptor_CC_8.npy')
        dataset = []
        y=[]
        for i in tqdm(range(len(data['smiles']))):
            if data['t1'][i]*data['speed'][i]>60:
                continue
            if data['t2'][i] * data['speed'][i]>120:
                continue
            smile = data['smiles'][i]
            mol = Chem.MolFromSmiles(smile)
            Finger = MACCSkeys.GenMACCSKeys(mol)
            sub_data = [x for x in Finger]
            MolWt = Descriptors.ExactMolWt(mol)
            nRotB = Descriptors.NumRotatableBonds(mol)  # Number of rotable bonds

            HBD = Descriptors.NumHDonors(mol)  # Number of H bond donors
            HBA = Descriptors.NumHAcceptors(mol)  # Number of H bond acceptors
            LogP = Descriptors.MolLogP(mol)  # LogP
            sub_data.extend([MolWt, nRotB, HBD, HBA, LogP])
            sub_data.extend(all_descriptor[i,[153,278,884,885,1273,1594,431,1768,1769,1288,1521]])
            sub_data.extend(data['eluent'][i])
            sub_data.extend([data['e'][i],data['m'][i],data['V_e'][i]])
            dataset.append(sub_data)
            y.append([data['t1'][i]*data['speed'][i],data['t2'][i]*data['speed'][i]])
            all_eluent_ratio.append(data['eluent_ratio'][i])
        X=np.array(dataset)
        y=np.array(y)
        all_eluent_ratio=np.array(all_eluent_ratio)
        np.save('dataset_save/X_CC.npy', X)
        np.save('dataset_save/Y_CC.npy', y)
        np.save('dataset_save/eluent_ratio.npy', all_eluent_ratio)
    else:
        X=np.load('dataset_save/X_CC.npy')
        y=np.load('dataset_save/Y_CC.npy')
    np.save('dataset_save/X_max_CC.npy',np.max(X,axis=0))
    np.save('dataset_save/X_min_CC.npy', np.min(X, axis=0))
    X = (X - np.min(X,axis=0)) / (np.max(X,axis=0)- np.min(X,axis=0)+ 1e-8)
    state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(state)
    np.random.shuffle(y)
    train_num = int(0.8 * X.shape[0])
    val_num = int(0.1 * X.shape[0])
    test_num = int(0.1 * X.shape[0])
    X_train= X[0:train_num]
    y_train = y[0:train_num]
    X_valid = X[train_num:train_num + val_num]
    y_valid=y[train_num:train_num + val_num]
    X_test = X[train_num + val_num:train_num + val_num + test_num]
    y_test= y[train_num + val_num:train_num + val_num + test_num]
    return X_train,y_train,X_valid,y_valid,X_test,y_test

def save_compound_index(data):
    all_smile = []
    compound_index = {}
    save_compound_index = []
    index = 0
    for i in tqdm(range(len(data['smiles']))):
        if data['smiles'][i] not in all_smile:
            all_smile.append(data['smiles'][i])
            compound_index[data['smiles'][i]] = index
            save_compound_index.append(index)
            index += 1
        else:
            save_compound_index.append(compound_index[data['smiles'][i]])
    df = pd.read_excel('dataset_1124.xlsx')
    df = df[df.t1 != -1]
    df = df[df.column_specs == 'Silica-CS 4g']
    df = df[np.isnan(df.t1) == False]
    df['compound_index'] = save_compound_index
    df.to_excel('dataset_1124_with_index.xlsx', index=False)
    print(compound_index)
    print(save_compound_index)

def split_dataset_by_compound(data):
    if 'X_CC_with_index.npy' not in os.listdir('dataset_save'):
        all_eluent_ratio = []
        all_compound_index=[]
        all_descriptor = np.load('dataset_save/all_descriptor_CC.npy')
        dataset = []
        y = []

        for i in tqdm(range(len(data['smiles']))):
            if data['t1'][i] * data['speed'][i] > 60:
                continue
            if data['t2'][i] * data['speed'][i] > 120:
                continue
            smile = data['smiles'][i]
            mol = Chem.MolFromSmiles(smile)
            Finger = MACCSkeys.GenMACCSKeys(mol)
            sub_data = [x for x in Finger]
            MolWt = Descriptors.ExactMolWt(mol)
            nRotB = Descriptors.NumRotatableBonds(mol)  # Number of rotable bonds

            HBD = Descriptors.NumHDonors(mol)  # Number of H bond donors
            HBA = Descriptors.NumHAcceptors(mol)  # Number of H bond acceptors
            LogP = Descriptors.MolLogP(mol)  # LogP
            sub_data.extend([MolWt, nRotB, HBD, HBA, LogP])
            sub_data.extend(all_descriptor[i, [153, 278, 884, 885, 1273, 1594, 431, 1768, 1769, 1288, 1521]])
            sub_data.extend(data['eluent'][i])
            sub_data.extend([data['e'][i], data['m'][i], data['V_e'][i]])
            dataset.append(sub_data)
            y.append([data['t1'][i] * data['speed'][i], data['t2'][i] * data['speed'][i]])
            all_eluent_ratio.append(data['eluent_ratio'][i])
            all_compound_index.append(data['compound_index'][i])
        X = np.array(dataset)
        y = np.array(y)
        all_eluent_ratio = np.array(all_eluent_ratio)
        all_compound_index = np.array(all_compound_index)
        np.save('dataset_save/X_CC_with_index.npy', X)
        np.save('dataset_save/Y_CC_with_index.npy', y)
        np.save('dataset_save/eluent_ratio_with_index.npy', all_eluent_ratio)
        np.save('dataset_save/compound_index.npy', all_compound_index)
    else:
        X = np.load('dataset_save/X_CC_with_index.npy')
        y = np.load('dataset_save/Y_CC_with_index.npy')
        all_eluent_ratio= np.load('dataset_save/eluent_ratio_with_index.npy')
        all_compound_index=np.load('dataset_save/compound_index.npy')
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0) + 1e-8)
    print(X.shape)
    df_X=pd.DataFrame(X)
    df_other=pd.DataFrame({'eluent_ratio':all_eluent_ratio.reshape(-1,),'compound_index':all_compound_index.reshape(-1,)})
    df_index=pd.DataFrame({'index':np.arange(0,X.shape[0],1)})
    df_y=pd.DataFrame(y)
    compound_index=np.unique(all_compound_index)
    state = np.random.get_state()
    np.random.shuffle(compound_index)

    train_num = int(0.8 * compound_index.shape[0])
    val_num = int(0.1 * compound_index.shape[0])
    test_num = int(0.1 * compound_index.shape[0])
    compound_index=compound_index.tolist()
    compound_train = compound_index[0:train_num]
    compound_valid = compound_index[train_num:train_num + val_num]
    compound_test= compound_index[train_num + val_num:train_num + val_num + test_num]
    X_train=df_X[df_other['compound_index'].isin(compound_train)].values
    X_valid = df_X[df_other['compound_index'].isin(compound_valid)].values
    X_test = df_X[df_other['compound_index'].isin(compound_test)].values
    y_train = df_y[df_other['compound_index'].isin(compound_train)].values
    y_valid = df_y[df_other['compound_index'].isin(compound_valid)].values
    y_test = df_y[df_other['compound_index'].isin(compound_test)].values
    train_index=df_index[df_other['compound_index'].isin(compound_train)].values
    valid_index=df_index[df_other['compound_index'].isin(compound_valid)].values
    test_index=df_index[df_other['compound_index'].isin(compound_test)].values
    return X_train, y_train, X_valid, y_valid, X_test, y_test,[train_index,valid_index,test_index]

class Model_ML():
    def __init__(self,config):
        super(Model_ML, self).__init__()
        self.seed=config.seed
        self.torch_seed=config.seed
        self.config=config
        self.data_range=config.data_range
        self.choose_train = config.choose_train
        self.choose_validate = config.choose_validate
        self.choose_test = config.choose_test
        self.add_dipole = config.add_dipole
        self.add_molecular_descriptors = config.add_molecular_descriptors
        self.add_eluent_matrix=config.add_eluent_matrix

        self.use_model=config.use_model
        self.LGB_max_depth=config.LGB_max_depth
        self.LGB_num_leaves=config.LGB_num_leaves
        self.LGB_learning_rate=config.LGB_learning_rate
        self.LGB_n_estimators=config.LGB_n_estimators
        self.LGB_early_stopping_rounds=config.LGB_early_stopping_rounds

        self.XGB_n_estimators=config.XGB_n_estimators
        self.XGB_max_depth = config.XGB_max_depth
        self.XGB_learning_rate = config.XGB_learning_rate

        self.RF_n_estimators=config.RF_n_estimators
        self.RF_random_state=config.RF_random_state
        self.RF_n_jobs=config.RF_n_jobs

        self.NN_hidden_neuron=config.NN_hidden_neuron
        self.NN_optimizer=config.NN_optimizer
        self.NN_lr= config.NN_lr
        self.NN_model_save_location=config.NN_model_save_location
        self.NN_max_epoch=config.NN_max_epoch
        self.NN_add_PINN=config.NN_add_PINN
        self.NN_epi=config.NN_epi
        self.device=config.device

        self.plot_row_num=config.plot_row_num
        self.plot_col_num=config.plot_col_num
        self.plot_importance_num=config.plot_importance_num



    def train(self,X_train,y_train,X_validate,y_validate):


        '''
        train model using LightGBM,Xgboost,Random forest or ANN
        '''
        print('----------Start Training!--------------')
        torch.manual_seed(self.torch_seed)
        if self.use_model=='LGB':
            model_t1 = lgb.LGBMRegressor(objective='regression', max_depth=self.LGB_max_depth,
                                      num_leaves=self.LGB_num_leaves,
                                      learning_rate=self.LGB_learning_rate, n_estimators=self.LGB_n_estimators)
            model_t2 = lgb.LGBMRegressor(objective='regression', max_depth=self.LGB_max_depth,
                                      num_leaves=self.LGB_num_leaves,
                                      learning_rate=self.LGB_learning_rate, n_estimators=self.LGB_n_estimators)
            model_t1.fit(X_train, list(y_train[:,0].reshape(y_train.shape[0], )),
                      eval_set=[(X_train, list(y_train[:,0].reshape(y_train.shape[0], ))),
                                (X_validate, list(y_validate[:,0].reshape(y_validate.shape[0], )))],
                      eval_names=('fit', 'val'), eval_metric='l2', early_stopping_rounds=self.LGB_early_stopping_rounds,
                      verbose=False)
            model_t2.fit(X_train, list(y_train[:, 1].reshape(y_train.shape[0], )),
                         eval_set=[(X_train, list(y_train[:, 1].reshape(y_train.shape[0], ))),
                                   (X_validate, list(y_validate[:, 1].reshape(y_validate.shape[0], )))],
                         eval_names=('fit', 'val'), eval_metric='l2',
                         early_stopping_rounds=self.LGB_early_stopping_rounds,
                         verbose=False)
            print('----------LGB Training Finished!--------------')
            return [model_t1,model_t2]
        elif self.use_model=='XGB':
            model_t1 = XGBRegressor(seed=self.seed,
                                 n_estimators=self.XGB_n_estimators,
                                 max_depth=self.XGB_max_depth,
                                 eval_metric='rmse',
                                 learning_rate=self.XGB_learning_rate,
                                 min_child_weight=1,
                                 subsample=1,
                                 colsample_bytree=1,
                                 colsample_bylevel=1,
                                 gamma=0)
            model_t2 = XGBRegressor(seed=self.seed,
                                    n_estimators=self.XGB_n_estimators,
                                    max_depth=self.XGB_max_depth,
                                    eval_metric='rmse',
                                    learning_rate=self.XGB_learning_rate,
                                    min_child_weight=1,
                                    subsample=1,
                                    colsample_bytree=1,
                                    colsample_bylevel=1,
                                    gamma=0)
            model_t1.fit(X_train, y_train[:,0].reshape(y_train.shape[0]))
            model_t2.fit(X_train, y_train[:, 1].reshape(y_train.shape[0]))
            print('----------XGB Training Finished!--------------')
            return [model_t1,model_t2]



    def test(self,X_test,y_test,model,save=True):
        '''
        Get test outcomes
        '''
        if self.use_model=='ANN':
            X_test = Variable(torch.from_numpy(X_test.astype(np.float32)).to(self.device), requires_grad=True)
            y_test = Variable(torch.from_numpy(y_test.astype(np.float32)).to(self.device))
            y_pred=model(X_test).cpu().data.numpy()
            y_pred_t1 = y_pred[:,0].reshape(-1,1)
            y_pred_t2 = y_pred[:,1].reshape(-1,1)
            y_test=y_test.cpu().data.numpy()

        if self.use_model=='XGB':
            y_pred_t1 = model[0].predict(X_test).reshape(-1,1)
            y_pred_t2 = model[1].predict(X_test).reshape(-1,1)
        if self.use_model=='LGB':
            y_pred_t1 = model[0].predict(X_test).reshape(-1,1)
            y_pred_t2 = model[1].predict(X_test).reshape(-1,1)
        y_test_t1 = y_test[:, 0].reshape(-1, 1)
        y_test_t2 = y_test[:, 1].reshape(-1, 1)
        if config.test_mode=='split_by_data':
            measure_t1,plot_t1=measurement(y_test_t1,y_pred_t1,f't1_{self.use_model}',c='#CD5C5C')
            measure_t2, plot_t2 = measurement(y_test_t2, y_pred_t2,f't2_{self.use_model}',c='#6495ED')
        if config.test_mode=='split_by_compound':
            measure_t1, plot_t1 = measurement(y_test_t1, y_pred_t1, f't1_{self.use_model}_compound',c='#CD5C5C')
            measure_t2, plot_t2 = measurement(y_test_t2, y_pred_t2, f't2_{self.use_model}_compound',c='#6495ED')
        df=pd.DataFrame({'true_t1':y_test_t1.reshape(-1,),
                         'true_t2':y_test_t2.reshape(-1,),
                         'pred_t1':y_pred_t1.reshape(-1,),
                         'pred_t2':y_pred_t2.reshape(-1,)})
        if save==True:
            if config.test_mode == 'split_by_data':
                df.to_csv(f'result_save/{self.use_model}.csv')
                with open(f"result_save/{self.use_model}.log", "w") as f:
                    f.write(f'MSE_t1: {measure_t1[0]}, RMSE_t1:{measure_t1[1]},  MAE_t1:{measure_t1[2]},   R_2_t1:{measure_t1[3]}\n')
                    f.write(f'MSE_t2: {measure_t2[0]}, RMSE_t2:{measure_t2[1]},  MAE_t2:{measure_t2[2]},   R_2_t2:{measure_t2[3]}')
            if config.test_mode == 'split_by_compound':
                df.to_csv(f'result_save/{self.use_model}_compound.csv')
                with open(f"result_save/{self.use_model}_compound.log", "w") as f:
                    f.write(f'MSE_t1: {measure_t1[0]}, RMSE_t1:{measure_t1[1]},  MAE_t1:{measure_t1[2]},   R_2_t1:{measure_t1[3]}\n')
                    f.write(f'MSE_t2: {measure_t2[0]}, RMSE_t2:{measure_t2[1]},  MAE_t2:{measure_t2[2]},   R_2_t2:{measure_t2[3]}')
        print(measure_t1)
        print(measure_t2)
        plt.show()


def calculate_similarity(Data_CC):
    dataset_Finger=[]
    for i in tqdm(range(len(Data_CC['smiles']))):
        if Data_CC['t1'][i] * Data_CC['speed'][i] > 60:
            continue
        if Data_CC['t2'][i] * Data_CC['speed'][i] > 120:
            continue
        smile = Data_CC['smiles'][i]
        mol = Chem.MolFromSmiles(smile)
        Figerprint = np.array([x for x in AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)])
        dataset_Finger.append(Figerprint)
    similarity=np.zeros([len(dataset_Finger),len(dataset_Finger)])
    for i in tqdm(range(len(dataset_Finger))):
        for j in range(len(dataset_Finger)):
            similarity[i,j]=calcu_tanimoto(dataset_Finger[i],dataset_Finger[j])
    np.save(f'dataset_save/similarity_all.npy',similarity)




config=parse_args()
config.test_mode='split_by_data'


### Switch between different datasets ###
# Data_CC=read_data_CC()
#save_compound_index(Data_CC)
Data_CC=read_data_CC_compound()
# Data_CC_8=read_data_CC_8()
# Data_CC_25=read_data_CC_25()
# Data_CC_40=read_data_CC_40()
# Data_CC_C18=read_data_CC_C18()
# Data_CC_reversed=read_data_CC_reversed()
#calculate_similarity(Data_CC)
#calcu_mord_CC()
if config.test_mode=='split_by_data':
    X_train_CC,y_train_CC,X_valid_CC,y_valid_CC,X_test_CC,y_test_CC=split_dataset(Data_CC)
if config.test_mode=='split_by_compound':
    X_train_CC,y_train_CC,X_valid_CC,y_valid_CC,X_test_CC,y_test_CC,split_index=split_dataset_by_compound(Data_CC)

Net_CC = ANN_CC(192, 256, 2).to(config.device)
model_name = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))
dir_name = 'model_save/'+model_name
optimizer_CC = torch.optim.Adam([{'params': Net_CC.parameters(), 'lr': config.NN_lr}])


### Switch between different modes ###
mode='predict_mix'
n_epochs=10000

if mode=='Train_NN':
    X_train_CC = Variable(torch.from_numpy(X_train_CC.astype(np.float32)).to(config.device), requires_grad=True)
    y_train_CC = Variable(torch.from_numpy(y_train_CC.astype(np.float32)).to(config.device))
    X_valid_CC = Variable(torch.from_numpy(X_valid_CC.astype(np.float32)).to(config.device), requires_grad=True)
    y_valid_CC = Variable(torch.from_numpy(y_valid_CC.astype(np.float32)).to(config.device))
    X_test_CC = Variable(torch.from_numpy(X_test_CC.astype(np.float32)).to(config.device), requires_grad=True)
    y_test_CC = Variable(torch.from_numpy(y_test_CC.astype(np.float32)).to(config.device))
    try:
        os.makedirs(dir_name)
    except OSError:
        pass

    with open(dir_name + '/' + 'data.txt', 'a+') as f:  # 设置文件对象
        for epoch in tqdm(range(n_epochs)):
            optimizer_CC.zero_grad()
            prediction_CC_1 = Net_CC(X_train_CC)[:,0].reshape(-1,1)
            prediction_CC_2 = Net_CC(X_train_CC)[:,1].reshape(-1,1)
            MSELoss_CC = torch.nn.MSELoss()
            loss_CC_1 = MSELoss_CC(y_train_CC[:, 0].reshape(-1, 1), prediction_CC_1)
            loss_CC_2 = MSELoss_CC(y_train_CC[:, 1].reshape(-1, 1), prediction_CC_2)
            loss =loss_CC_1 + loss_CC_2
            loss.backward()
            optimizer_CC.step()
            if (epoch + 1) % 100 == 0:
                pred_valid_CC = Net_CC(X_valid_CC)
                pred_test_CC = Net_CC(X_test_CC)
                valid_loss_CC = MSELoss_CC(y_valid_CC, pred_valid_CC)
                test_loss_CC = MSELoss_CC(y_test_CC, pred_test_CC)
                print(
                    f"iter_num: {epoch + 1}     loss_CC_1:{loss_CC_1.item()}   loss_CC_2:{loss_CC_2.item()}  \n"
                    f" valid_CC:{valid_loss_CC.item()}  test_CC:{test_loss_CC.item()}")
                f.write(
                    f"iter_num: {epoch + 1}      loss_CC_1:{loss_CC_1.item()}   loss_CC_2:{loss_CC_2.item()}"
                    f"valid_CC:{valid_loss_CC.item()}  \n   ")
                torch.save(Net_CC.state_dict(), dir_name + '/' + f'model_{epoch + 1}.pkl')
                torch.save(optimizer_CC.state_dict(), dir_name + '/' + f'optimizer_{epoch + 1}.pkl')
            if (epoch + 1) % 2500 == 0:
                for p in optimizer_CC.param_groups:
                    p['lr'] *= 0.9
                print('adjust lr:', optimizer_CC.state_dict()['param_groups'][0]['lr'])

if mode=='Test_NN':
    X_train_CC = Variable(torch.from_numpy(X_train_CC.astype(np.float32)).to(config.device), requires_grad=True)
    y_train_CC = Variable(torch.from_numpy(y_train_CC.astype(np.float32)).to(config.device))
    X_valid_CC = Variable(torch.from_numpy(X_valid_CC.astype(np.float32)).to(config.device), requires_grad=True)
    y_valid_CC = Variable(torch.from_numpy(y_valid_CC.astype(np.float32)).to(config.device))
    X_test_CC = Variable(torch.from_numpy(X_test_CC.astype(np.float32)).to(config.device), requires_grad=True)
    y_test_CC = Variable(torch.from_numpy(y_test_CC.astype(np.float32)).to(config.device))
    if config.test_mode == 'split_by_data':
        #Net_CC.load_state_dict(torch.load('model_save/ANN/model_700.pkl'))
        Net_CC.load_state_dict(torch.load('model_save/ANN_1124/model_600.pkl'))
    if config.test_mode == 'split_by_compound':
        Net_CC.load_state_dict(torch.load('model_save/ANN_compound/model_2000.pkl'))
    pred_test_CC=Net_CC(X_test_CC)
    def calaculate_metric(y_test, y_pred):
        MSE = np.sum(np.abs(y_test - y_pred) ** 2) / y_test.shape[0]
        RMSE = np.sqrt(MSE)
        MAE = np.sum(np.abs(y_test - y_pred)) / y_test.shape[0]
        R_square = 1 - (((y_test - y_pred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum())
        print(MSE, RMSE, MAE, R_square)
        return MSE, RMSE, MAE, R_square


    y_pred_t1=pred_test_CC[:, 0].cpu().data.numpy().reshape(-1, )
    y_test_t1=y_test_CC[:, 0].cpu().data.numpy().reshape(-1, )
    y_pred_t2=pred_test_CC[:, 1].cpu().data.numpy().reshape(-1, )
    y_test_t2=y_test_CC[:, 1].cpu().data.numpy().reshape(-1, )
    if config.test_mode == 'split_by_data':
        measure_t1, plot_t1 = measurement(y_test_t1, y_pred_t1, f't1_ANN',c='#CD5C5C')
        measure_t2, plot_t2 = measurement(y_test_t2, y_pred_t2, f't2_ANN',c='#6495ED')
        df = pd.DataFrame({'true_t1': y_test_CC[:, 0].cpu().data.numpy().reshape(-1, ),
                           'true_t2': y_test_CC[:, 1].cpu().data.numpy().reshape(-1, ),
                           'pred_t1': pred_test_CC[:, 0].cpu().data.numpy().reshape(-1, ),
                           'pred_t2': y_test_CC[:, 1].cpu().data.numpy().reshape(-1, )})
        df.to_csv(f'result_save/NN.csv')
        with open(f"result_save/NN.log", "w") as f:
            f.write(
                f'MSE_t1: {measure_t1[0]}, RMSE_t1:{measure_t1[1]},  MAE_t1:{measure_t1[2]},   R_2_t1:{measure_t1[3]}\n')
            f.write(
                f'MSE_t2: {measure_t2[0]}, RMSE_t2:{measure_t2[1]},  MAE_t2:{measure_t2[2]},   R_2_t2:{measure_t2[3]}')
    if config.test_mode == 'split_by_compound':
        measure_t1, plot_t1 = measurement(y_test_t1, y_pred_t1, f't1_ANN_compound',c='#CD5C5C')
        measure_t2, plot_t2 = measurement(y_test_t2, y_pred_t2, f't2_ANN_compound',c='#6495ED')
        df = pd.DataFrame({'true_t1': y_test_CC[:, 0].cpu().data.numpy().reshape(-1, ),
                           'true_t2': y_test_CC[:, 1].cpu().data.numpy().reshape(-1, ),
                           'pred_t1': pred_test_CC[:, 0].cpu().data.numpy().reshape(-1, ),
                           'pred_t2': pred_test_CC[:, 1].cpu().data.numpy().reshape(-1, )})
        df.to_csv(f'result_save/NN_compound.csv')
        with open(f"result_save/NN_compound.log", "w") as f:
            f.write(
                f'MSE_t1: {measure_t1[0]}, RMSE_t1:{measure_t1[1]},  MAE_t1:{measure_t1[2]},   R_2_t1:{measure_t1[3]}\n')
            f.write(
                f'MSE_t2: {measure_t2[0]}, RMSE_t2:{measure_t2[1]},  MAE_t2:{measure_t2[2]},   R_2_t2:{measure_t2[3]}')

if mode=='Train_XGB':
    Model = Model_ML(config)
    Model.use_model='XGB'
    model = Model.train(X_train_CC, y_train_CC, X_valid_CC, y_valid_CC)
    Model.test(X_test_CC, y_test_CC, model)
if mode == 'Train_LGB':
    Model = Model_ML(config)
    Model.use_model = 'LGB'
    model = Model.train(X_train_CC, y_train_CC, X_valid_CC, y_valid_CC)
    Model.test(X_test_CC, y_test_CC, model)
if mode=='QGeoGNN': ## Invoke different modules to train corresponding models
    QGeoGNN(Data_CC, MODEL='Train')
    # QGeoGNN_transfer_8g(Data_CC_8, MODEL='Test')
    # QGeoGNN_transfer_25g(Data_CC_25, MODEL='Train')
    # QGeoGNN_transfer_40g(Data_CC_40, MODEL='Train')
    # QGeoGNN_transfer_C18(Data_CC_C18, MODEL='Test')
    # QGeoGNN_transfer_reversed(Data_CC_reversed, MODEL='Test')
    #QGeoGNN_different_data_num(Data_CC, MODEL='Train')
    #QGeoGNN_different_noise_level( Data_CC, MODEL='Test')
    #QGeoGNN_cycle(Data_CC, MODEL='Train')-
if mode=='Analysis_similarity':
    def save_sim():
        QGeoGNN_cycle(Data_CC, MODEL='Test')
        for i in range(20):
            if i==0:
                df=pd.read_csv(f'result_save/cross_iter/GeoGNN_compound_{i}.csv')
            else:
                df=pd.concat([df,pd.read_csv(f'result_save/cross_iter/GeoGNN_compound_{i}.csv')])
        similarity=np.load(f'dataset_save/similarity_all.npy')
        # all_sim=[]
        # for i in range(similarity.shape[0]):
        #     all_sim.append(find_sim(heapq.nlargest(100, similarity[i])))
        # print(sorted(all_sim)[-20:])
        # plt.hist(all_sim)
        # plt.show()

        for i in range(20):
            index_save=np.load(f'result_save/cross_iter/index_{i}.npy',allow_pickle=True).item()
            test_index=index_save['test_index']
            train_index=index_save['train_index']
            train_index=train_index.reshape(-1,)
            test_index = test_index.reshape(-1, )
            similar_train = similarity[:, train_index.tolist()]



            big_sim = []
            for j in test_index:
                sim_20 = heapq.nlargest(100, similar_train[j])
                big_sim.append(find_sim(sim_20))
            big_sim = np.array(big_sim)

            if i==0:
                df_sim=df[df['test_index'].isin(test_index)]
                df_sim['similarity']=big_sim
            else:
                df_save = df[df['test_index'].isin(test_index)]
                df_save['similarity'] = big_sim
                df_sim=pd.concat([df_sim,df_save])
            df_sim.to_csv('result_save/cross_iter/result_sim.csv',index=False)
            print(df_sim)
    #save_sim()
    result=pd.read_csv('result_save/cross_iter/result_sim.csv')
    result_65=result[result['similarity']>0.65]
    result_55 = result[result['similarity'] > 0.55]
    result_45 = result[result['similarity'] > 0.45]
    result_35 = result[result['similarity'] > 0.35]
    result_25 = result[result['similarity'] > 0.25]
    result_0 = result[result['similarity'] > 0.15]
    measurement(result_65['true_t1'], result_65['pred_t1'], f't1_0.6')
    measurement(result_65['true_t2'], result_65['pred_t2'], f't2_0.6')
    print('--------------------')
    measurement(result_55['true_t1'], result_55['pred_t1'], f't1_0.6')
    measurement(result_55['true_t2'], result_55['pred_t2'], f't2_0.6')
    print('--------------------')
    measurement(result_45['true_t1'], result_45['pred_t1'], f't1_0.6')
    measurement(result_45['true_t2'], result_45['pred_t2'], f't2_0.6')
    print('--------------------')
    measurement(result_35['true_t1'], result_35['pred_t1'], f't1_0.6')
    measurement(result_35['true_t2'], result_35['pred_t2'], f't2_0.6')
    print('--------------------')
    measurement(result_25['true_t1'], result_25['pred_t1'], f't1_0.6')
    measurement(result_25['true_t2'], result_25['pred_t2'], f't2_0.6')
    print('--------------------')
    measurement(result_0['true_t1'], result_0['pred_t1'], f't1_0.6')
    measurement(result_0['true_t2'], result_0['pred_t2'], f't2_0.6')
    # measure_t1, plot_t1 = measurement(result_60['true_t1'], result_60['pred_t1'], f't1_0.6')
    # measure_t2, plot_t2 = measurement(result_60['true_t2'], result_60['pred_t2'], f't2_0.6')
    #plt.show()

if mode=='predict_mix':
    all_eluent=['0/1','1/1','2/1','5/1','10/1','20/1','50/1','1/0']
    all_prob=[]
    result_data = [] 
    
    # Can input any number of SMILES
    compound_smiles = [
        'CC(=O)C1=CC=CC=C1',
        'CCC1=CC=CC=C1'
    ]
    
    # Automatically generate color configuration (supports up to 10 compounds)
    colors = plt.cm.tab10.colors
    compound_colors = {
        i: colors[i % 10] for i in range(len(compound_smiles))
    }


    for eluent in tqdm(all_eluent):
        # Dynamically get prediction results
        pred_results = predict_separate(compound_smiles, eluent, 'PE', 100, 300)
        
        # Calculate the separation probability of the current eluent
        current_prob = calculate_separation_probability(*pred_results)
        all_prob.append(current_prob)
        
        # Build result entry
        result_entry = {
            'Eluent_Ratio': eluent,
            'Separation_Probability': current_prob
        }
        
        # Dynamically process compound data
        for idx in range(len(compound_smiles)):
            # v1 = pred_results[idx*2]
            # v2 = pred_results[idx*2+1]
            v1 = [x/10 for x in pred_results[idx*2]]
            v2 = [x/10 for x in pred_results[idx*2+1]]
            result_entry.update({
                f'Compound{idx+1}_V1_10%': v1[0],
                f'Compound{idx+1}_V1_50%': v1[1],
                f'Compound{idx+1}_V1_90%': v1[2],
                f'Compound{idx+1}_V2_10%': v2[0],
                f'Compound{idx+1}_V2_50%': v2[1],
                f'Compound{idx+1}_V2_90%': v2[2]
            })
        
        result_data.append(result_entry)

    df = pd.DataFrame(result_data)
    df.to_csv('separation_probability_results.csv', index=False)

    # ==== Dynamic Visualization ====
    plt.style.use('ggplot')
    plt.figure(figsize=(14, 6 + 2*len(compound_smiles)), dpi=300)
    
    # Redefine axes (increase top and bottom margins)
    y_interval_height = 4.0  # Increase total height
    y_padding = 0.8  # New parameter for top and bottom margins
    y_positions = {ratio: idx*y_interval_height for idx, ratio in enumerate(df['Eluent_Ratio'])}
    
    # Layout parameters
    h_spacing = 0.5
    compound_height = 0.3  # Height of single compound interval
    
    # Generate color configuration (adapted to ggplot theme)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    compound_colors = {i: colors[i % len(colors)] for i in range(len(compound_smiles))}
    
    for idx, row in df.iterrows():
        y_base = y_positions[row['Eluent_Ratio']]
        
        # Calculate compound positions (preserve margins)
        compound_y_positions = np.linspace(
            start=y_base + y_padding,
            stop=y_base + y_interval_height - y_padding,
            num=len(compound_smiles),
            endpoint=True
        )
        
        # Adjust background region height (preserve margins)
        plt.fill_betweenx(
            y=[y_base + y_padding/2, y_base + y_interval_height - y_padding/2],
            x1=0,
            x2=max(df[[f'Compound{i+1}_V2_90%' for i in range(len(compound_smiles))]].max()),
            color='lightgray',
            alpha=0.15
        )
        
        # Draw intervals for each compound
        for compound_idx in range(len(compound_smiles)):
            y_center = compound_y_positions[compound_idx]
            y_range = [y_center + compound_height, y_center - compound_height]
            
            # V1 interval
            plt.fill_betweenx(
                y=y_range,
                x1=row[f'Compound{compound_idx+1}_V1_10%'] - h_spacing,
                x2=row[f'Compound{compound_idx+1}_V1_90%'] - h_spacing,
                color=compound_colors[compound_idx],
                alpha=0.4,
                edgecolor='k',
                linewidth=0.8
            )
            # V1 median line
            plt.plot(
                [row[f'Compound{compound_idx+1}_V1_50%'] - h_spacing]*2,
                y_range,
                color=compound_colors[compound_idx],
                linestyle='--',
                lw=1.5
            )
            
            # V2 interval
            plt.fill_betweenx(
                y=y_range,
                x1=row[f'Compound{compound_idx+1}_V2_10%'] + h_spacing,
                x2=row[f'Compound{compound_idx+1}_V2_90%'] + h_spacing,
                color=compound_colors[compound_idx],
                alpha=0.6,
                edgecolor='k',
                linewidth=0.8
            )
            # V2 median line
            plt.plot(
                [row[f'Compound{compound_idx+1}_V2_50%'] + h_spacing]*2,
                y_range,
                color=compound_colors[compound_idx],
                linestyle='--',
                lw=1.5
            )
    

    # Configure axes
    plt.yticks(
        ticks=[pos + y_interval_height/2 for pos in y_positions.values()],  # Center labels
        labels=[f'{ratio}' for ratio in y_positions.keys()],
        fontsize=10
    )
    plt.gca().invert_yaxis()
    
    # Add horizontal reference lines
    for y in y_positions.values():
        plt.axhline(y=y+y_interval_height, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
    
    # Configure axes
    plt.xlabel('Retention Volume (V/mL)')
    plt.ylabel('Eluent Ratio (PE/EA)')
    plt.title(f'Retention Volume Distribution by Eluent Ratio ({len(compound_smiles)} Compounds)')

    # Dynamic legend
    legend_elements = []
    for i in range(len(compound_smiles)):
        # Add V1 interval legend
        legend_elements.append(
            plt.Rectangle((0,0),1,1, 
                         fc=compound_colors[i], 
                         alpha=0.4, 
                         label=f'Compound{i+1} $V_1$')
        )
        # Add V1 median line legend
        legend_elements.append(
            plt.Line2D([0],[0], 
                      color=compound_colors[i], 
                      lw=1.5, 
                      linestyle='--',
                      label=f'Compound{i+1} $V_{{50\%}}$')
        )
        # Add V2 interval legend
        legend_elements.append(
            plt.Rectangle((0,0),1,1, 
                         fc=compound_colors[i], 
                         alpha=0.6, 
                         label=f'Compound{i+1} $V_2$')
        )

    plt.legend(
        handles=legend_elements,
        bbox_to_anchor=(0.85, 1),
        loc='upper left',
        ncol=1,
        fontsize=9,
        framealpha=0.9
    )

    # Add Sp label (adjust vertical alignment)
    max_x = plt.gca().get_xlim()[1]
    for ratio, base_y in y_positions.items():
        sp = df[df['Eluent_Ratio'] == ratio]['Separation_Probability'].values[0]
        # Calculate the vertical center position of the label (aligned with the eluent ratio label)
        label_y = base_y + y_interval_height/2  # Use interval height to calculate midpoint
        
        plt.text(
            x=max_x * 1.02,
            y=label_y,  # Use adjusted center position
            s=f'$S_p$: {sp:.2f}',
            va='center',
            ha='left',
            fontsize=10,
            color='black',
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2)
        )

    plt.grid(True, alpha=0.3, linestyle='--')

    # Save plot
    plt.tight_layout()
    plt.savefig('retention_profile.png', bbox_inches='tight')
    plt.show()
    # ==== End of Visualization ====
    
    all_prob=np.array(all_prob)
    print(all_prob)
    print(f'the best eluent is {all_eluent[np.argmax(all_prob)]}, the best Sp is {all_prob[np.argmax(all_prob)]}')

    