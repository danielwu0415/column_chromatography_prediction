import numpy as np
import pandas as pd
import argparse
import os
import torch
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from xgboost.sklearn import XGBClassifier,XGBRegressor
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from tqdm import  tqdm
from rdkit.Chem import Descriptors
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
import random
from QGeoGNN import *
import scipy
import pubchempy as pcp
Eluent_smiles=['CCCCCC','CC(OCC)=O','C(Cl)Cl','CO','O']
def convert_CAS_to_smile(cas):
    try:
        # 搜索化合物信息
        c = pcp.get_compounds(cas, 'name')
        # 提取SMILES字符串
        smiles = c[0].isomeric_smiles
        return smiles
    except Exception as e:
        print("An error occurred:", e)
        return None
def get_descriptor(smiles,ratio):
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

def get_eluent_descriptor(eluent_array):
    eluent=eluent_array
    des = np.zeros([6,])
    for i in range(eluent.shape[0]):
        if eluent[i] != 0:
            e_descriptors = get_descriptor(Eluent_smiles[i], eluent[i])
            des+=e_descriptors
    return des

def find_sim(list_1):
    sim = 1
    for i in list_1:
        if i == 1:
            continue
        else:
            sim = i
            break
    return sim
def measurement(y_test, y_pred, method,c='blue',s=15):
    MSE = np.sum(np.abs(y_test - y_pred) ** 2) / y_test.shape[0]
    RMSE = np.sqrt(MSE)
    MAE = np.sum(np.abs(y_test - y_pred)) / y_test.shape[0]
    R_square = 1 - (((y_test - y_pred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum())
    plt.style.use('ggplot')
    fig, axes = plt.subplots(1, 1, figsize=(2, 2), dpi=300)
    plt.plot(np.linspace(0, max(np.max(y_test), np.max(y_pred)), 100),
             np.linspace(0, max(np.max(y_test), np.max(y_pred)), 100), linewidth=1, linestyle='--', color='black')

    plt.scatter(y_test, y_pred, c=c, s=s, alpha=0.4)
    # plt.xticks(x,[1,4,5,7,8,13,15,50,80,90,114,148,165],fontproperties='Arial',size=7)
    plt.xticks(fontproperties='Arial', size=7)
    plt.yticks(fontproperties='Arial', size=7)
    if method!='TLC':
        plt.ylim(-2, max(np.max(y_test), np.max(y_pred)) * 1.1)
        plt.xlim(-2, max(np.max(y_test), np.max(y_pred)) * 1.1)
    if method == 'TLC':
        plt.ylim(-0.1,  1.1)
        plt.xlim(-0.1, 1.1)
    #plt.grid(ls='--')
    #plt.xlabel('True value', fontproperties='Arial', size=7)
    #plt.ylabel("Predict value", fontproperties='Arial', size=7)
    #plt.title(f"Predict v.s. True", fontproperties='Arial', size=7)
    from matplotlib.ticker import MaxNLocator
    axes.yaxis.set_major_locator(MaxNLocator(4))
    axes.xaxis.set_major_locator(MaxNLocator(4))
    plt.savefig(f'plot_save/R_{method}.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'plot_save/R_{method}.pdf', bbox_inches='tight', dpi=300)
    print([MSE, RMSE, MAE, R_square])
    return [MSE, RMSE, MAE, R_square],fig

def obtain_3D_mol(smiles,name):
    mol = AllChem.MolFromSmiles(smiles)
    new_mol = Chem.AddHs(mol)
    res = AllChem.EmbedMultipleConfs(new_mol)
    ### MMFF generates multiple conformations
    res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
    new_mol = Chem.RemoveHs(new_mol)
    Chem.MolToMolFile(new_mol, name+'.mol')
    return new_mol
def convert_eluent(eluent):
    ratio=[]
    for e in eluent:
        PE=int(e.split('/')[0])
        EA=int(e.split('/')[1])
        ratio.append(get_eluent_descriptor(np.array([PE, EA, 0, 0, 0]) / (PE + EA)))
        # DCM = int(e.split('/')[0])
        # MeOH = int(e.split('/')[1])
        # ratio.append(get_eluent_descriptor(np.array([0,0,DCM,MeOH,0])/(DCM+MeOH)))
        # MeOH = int(e.split('/')[0])
        # H2O = int(e.split('/')[1])
        # ratio.append(get_eluent_descriptor(np.array([0,0,0,MeOH,H2O])/(MeOH+H2O)))
    return np.vstack(ratio)
def convert_e(e):
    new_e=np.zeros([e.shape[0],])
    for i in range(len(e)):
        if e[i]=='PE':
            new_e[i]=0
        elif e[i]=='EA':
            new_e[i]=1
        elif e[i]=='DCM':
            new_e[i]=2
        else:
            print(e)
    return new_e
def convert_eluent_ratio(eluent):
    ratio = []
    for e in eluent:
        PE = int(e.split('/')[0])
        EA = int(e.split('/')[1])
        ratio.append(PE / (PE + EA))
        # DCM = int(e.split('/')[0])
        # MeOH = int(e.split('/')[1])
        # ratio.append(DCM / (DCM + MeOH))
        # MeOH = int(e.split('/')[0])
        # H2O = int(e.split('/')[1])
        # ratio.append(MeOH / (MeOH + H2O))
    return np.vstack(ratio)

def read_data_CC():
    df=pd.read_excel('dataset_1124.xlsx')
    df=df[df.t1!=-1]
    df=df[df.column_specs=='Silica-CS 4g']
    df=df[np.isnan(df.t1)==False]
    t1=df['t1'].values*50/(1000*60)
    t2=df['t2'].values*50/(1000*60)
    smiles=df['smiles'].values
    rho=df['密度g/ml'].values
    V=df['V/ul'].values
    m=rho*V
    V_e=df['上样溶剂量/ul'].values
    e=df['上样溶剂'].values
    eluent=df['PE/EA'].values
    speed=df['流速 ml/min'].values
    eluent_ratio=convert_eluent_ratio(eluent)
    eluent=convert_eluent(eluent)
    e=convert_e(e)
    np.save('result_save/eluent_ratio.npy',eluent_ratio)
    data={'t1':t1,'t2':t2,'smiles':smiles,'m':m,'V_e':V_e,'e':e,'eluent':eluent,'speed':speed,'eluent_ratio':eluent_ratio}
    return data

def calcu_tanimoto(key_1,key_2):
    a=np.sum(key_1)
    b=np.sum(key_2)
    total_key=key_1+key_2
    c=np.sum(total_key==2)
    return c/(a+b-c)
def read_data_CC_compound():
    df = pd.read_excel(r'C:\Users\PC\PycharmProjects\column_learning\dataset_4g_with_index.xlsx')
    df = df[df.t1 != -1]
    df = df[df.column_specs == 'Silica-CS 4g']
    df = df[np.isnan(df.t1) == False]
    t1 = df['t1'].values * 50 / (1000 * 60)
    t2 = df['t2'].values * 50 / (1000 * 60)
    smiles = df['smiles'].values
    rho = df['密度g/ml'].values
    V = df['V/ul'].values
    m = rho * V
    V_e = df['上样溶剂量/ul'].values
    e = df['上样溶剂'].values
    eluent = df['PE/EA'].values
    speed = df['流速 ml/min'].values
    eluent_ratio = convert_eluent_ratio(eluent)
    eluent = convert_eluent(eluent)
    #column_dia=df['column_dia'].values
    #column_len = df['column_len'].values
    #column_den=df['column_den'].values
    compound_index=df['compound_index'].values
    e = convert_e(e)
    # np.save('result_save/eluent_ratio.npy',eluent_ratio)
    data = {'t1': t1, 't2': t2, 'smiles': smiles, 'm': m, 'V_e': V_e, 'e': e, 'eluent': eluent, 'speed': speed,
            'eluent_ratio': eluent_ratio,'compound_index':compound_index} 
    return data
def read_data_TLC():
    df=pd.read_excel('TLC_dataset.xlsx')
    ID=df['TLC_ID'].values
    COMPOUND_ID=df['COMPOUND_ID'].values
    H=df['H'].values
    EA=df['EA'].values
    DCM=df['DCM'].values
    MeOH=df['MeOH'].values
    Et2O= df['Et2O'].values
    Rf=df['Rf'].values
    smiles=df['COMPOUND_SMILES'].values
    data = {'TLC_ID': ID, 'COMPOUND_ID': COMPOUND_ID, 'H': H, 'EA': EA, 'DCM': DCM, 'MeOH': MeOH,
            'Et2O': Et2O, 'Rf': Rf,'smiles':smiles}
    return data
def read_data_HPLC():
    df = pd.read_csv('SMRT_dataset.csv',delimiter=';')
    inchi=df['inchi'].values
    RT=df['rt'].values
    data={'inchi':inchi,'RT':RT}
    return data
def mord(mol, nBits=1826, errors_as_zeros=True):
    try:
        result = calc(mol)
        desc_list = [r if not is_missing(r) else 0 for r in result]
        np_arr = np.array(desc_list)
        return np_arr
    except:
        return np.NaN if not errors_as_zeros else np.zeros((nBits,), dtype=np.float32)


def calcu_mord_CC():
    data=read_data_CC_compound()
    all_descriptor=[]
    for i in tqdm(range(len(data['smiles']))):
        smile = data['smiles'][i]
        mol = Chem.MolFromSmiles(smile)
        descriptor=mord(mol)
        all_descriptor.append(descriptor)
    all_descriptor=np.vstack(all_descriptor)
    np.save('dataset_save/dataset_4g.npy',all_descriptor)

def calcu_mord_TLC():
    data=read_data_TLC()
    all_descriptor=[]
    for i in tqdm(range(len(data['smiles']))):
        smile = data['smiles'][i]
        mol = Chem.MolFromSmiles(smile)
        descriptor=mord(mol)
        all_descriptor.append(descriptor)
    all_descriptor=np.vstack(all_descriptor)
    np.save('dataset_save/all_descriptor_TLC.npy',all_descriptor)

def calcu_mord_HPLC():
    data=read_data_HPLC()
    all_descriptor=[]
    for i in tqdm(range(len(data['inchi']))):
        inchi = data['inchi'][i]
        mol = Chem.inchi.MolFromInchi(inchi.strip(), sanitize=True)
        descriptor=mord(mol)
        all_descriptor.append(descriptor)
    all_descriptor=np.vstack(all_descriptor)
    np.save('dataset_save/all_descriptor_HPLC.npy',all_descriptor)

def get_correlation_CC():
    data = read_data_CC()
    all_desctiptor=np.load('dataset_save/all_descriptor_CC.npy')
    V_1=data['t1']*data['speed']
    V_2=data['t2']*data['speed']
    for i in tqdm(range(all_desctiptor.shape[1])):
        corr_1=scipy.stats.spearmanr(all_desctiptor[:,i],V_1.reshape(-1,))[0]
        if np.abs(corr_1)>0.2:
            print(i,corr_1)

def calcu_mord_CC_8():
    data=read_data_CC_8()
    all_descriptor=[]
    for i in tqdm(range(len(data['smiles']))):
        smile = data['smiles'][i]
        mol = Chem.MolFromSmiles(smile)
        descriptor=mord(mol)
        all_descriptor.append(descriptor)
    all_descriptor=np.vstack(all_descriptor)
    np.save('dataset_save/dataset_8g.npy',all_descriptor)

def calcu_mord_CC_25():
    data=read_data_CC_25()
    all_descriptor=[]
    for i in tqdm(range(len(data['smiles']))):
        smile = data['smiles'][i]
        mol = Chem.MolFromSmiles(smile)
        descriptor=mord(mol)
        all_descriptor.append(descriptor)
    all_descriptor=np.vstack(all_descriptor)
    np.save('dataset_save/dataset_25g.npy',all_descriptor)

def read_data_CC_8():
    df=pd.read_excel('dataset_8g_with_index.xlsx')
    df=df[df.t1!=-1]
    df=df[df.column_specs=='Silica-CS 4g+4g']
    df=df[np.isnan(df.t1)==False]
    t1=df['t1'].values*50/(1000*60)
    t2=df['t2'].values*50/(1000*60)
    smiles=df['smiles'].values
    rho=df['密度g/ml'].values
    V=df['V/ul'].values
    m=rho*V
    V_e=df['上样溶剂量/ul'].values
    e=df['上样溶剂'].values
    eluent=df['PE/EA'].values
    speed=df['流速 ml/min'].values
    eluent_ratio=convert_eluent_ratio(eluent)
    eluent=convert_eluent(eluent)
    compound_index = df['compound_index'].values
    e=convert_e(e)
    #np.save('result_save/eluent_ratio.npy',eluent_ratio)
    data={'t1':t1,'t2':t2,'smiles':smiles,'m':m,'V_e':V_e,'e':e,'eluent':eluent,'speed':speed,
          'eluent_ratio':eluent_ratio,'compound_index':compound_index}
    return data

def read_data_CC_25():
    df=pd.read_excel('dataset_25g_with_index.xlsx')
    df=df[df.t1!=-1]
    df=df[df.column_specs=='Silica-CS 25g']
    df=df[np.isnan(df.t1)==False]
    t1=df['t1'].values*50/(1000*60)
    t2=df['t2'].values*50/(1000*60)
    smiles=df['smiles'].values
    rho=df['密度g/ml'].values
    V=df['V/ul'].values
    m=rho*V
    V_e=df['上样溶剂量/ul'].values
    e=df['上样溶剂'].values
    eluent=df['PE/EA'].values
    speed=df['流速 ml/min'].values
    eluent_ratio=convert_eluent_ratio(eluent)
    eluent=convert_eluent(eluent)
    compound_index = df['compound_index'].values
    e=convert_e(e)
    #np.save('result_save/eluent_ratio.npy',eluent_ratio)
    data={'t1':t1,'t2':t2,'smiles':smiles,'m':m,'V_e':V_e,'e':e,'eluent':eluent,'speed':speed,
          'eluent_ratio':eluent_ratio,'compound_index':compound_index}
    return data
def read_data_CC_40():
    df=pd.read_excel('dataset_40g_with_index.xlsx')
    df=df[df.t1!=-1]
    df=df[df.column_specs=='Silica-CS 40g']
    df=df[np.isnan(df.t1)==False]
    t1=df['t1'].values*50/(1000*60)
    t2=df['t2'].values*50/(1000*60)
    smiles=df['smiles'].values
    rho=df['密度g/ml'].values
    V=df['V/ul'].values
    m=rho*V
    V_e=df['上样溶剂量/ul'].values
    e=df['上样溶剂'].values
    eluent=df['PE/EA'].values
    speed=df['流速 ml/min'].values
    eluent_ratio=convert_eluent_ratio(eluent)
    eluent=convert_eluent(eluent)
    compound_index = df['compound_index'].values
    e=convert_e(e)
    #np.save('result_save/eluent_ratio.npy',eluent_ratio)
    data={'t1':t1,'t2':t2,'smiles':smiles,'m':m,'V_e':V_e,'e':e,'eluent':eluent,'speed':speed,
          'eluent_ratio':eluent_ratio,'compound_index':compound_index}
    return data

def read_data_CC_DCM():
    df=pd.read_excel('dataset_DCM_with_index.xlsx')
    df=df[df.t1!=-1]
    df=df[df.column_specs=='Silica-CS 4g']
    df=df[np.isnan(df.t1)==False]
    t1=df['t1'].values*50/(1000*60)
    t2=df['t2'].values*50/(1000*60)
    smiles=df['smiles'].values
    rho=df['密度g/ml'].values
    V=df['V/ul'].values
    m=rho*V
    V_e=df['上样溶剂量/ul'].values
    e=df['上样溶剂'].values
    eluent=df['DCM/MeOH'].values
    speed=df['流速 ml/min'].values
    eluent_ratio=convert_eluent_ratio(eluent)
    eluent=convert_eluent(eluent)
    compound_index = df['compound_index'].values
    e=convert_e(e)
    #np.save('result_save/eluent_ratio.npy',eluent_ratio)
    data={'t1':t1,'t2':t2,'smiles':smiles,'m':m,'V_e':V_e,'e':e,'eluent':eluent,'speed':speed,
          'eluent_ratio':eluent_ratio,'compound_index':compound_index}
    return data

def read_data_CC_C18():
    df=pd.read_excel('dataset_C18_with_index.xlsx')
    df=df[df.t1!=-1]
    df=df[df.column_specs=='C18']
    df=df[np.isnan(df.t1)==False]
    t1=df['t1'].values*50/(1000*60)
    t2=df['t2'].values*50/(1000*60)
    smiles=df['smiles'].values
    rho=df['密度g/ml'].values
    V=df['V/ul'].values
    m=rho*V
    V_e=df['上样溶剂量/ul'].values
    e=df['上样溶剂'].values
    eluent=df['MeOH/H2O'].values
    speed=df['流速 ml/min'].values
    eluent_ratio=convert_eluent_ratio(eluent)
    eluent=convert_eluent(eluent)
    compound_index = df['compound_index'].values
    e=convert_e(e)
    #np.save('result_save/eluent_ratio.npy',eluent_ratio)
    data={'t1':t1,'t2':t2,'smiles':smiles,'m':m,'V_e':V_e,'e':e,'eluent':eluent,'speed':speed,
          'eluent_ratio':eluent_ratio,'compound_index':compound_index}
    return data

def read_data_CC_NH2():
    df=pd.read_excel('dataset_NH2_with_index.xlsx')
    df=df[df.t1!=-1]
    df=df[df.column_specs=='NH2']
    df=df[np.isnan(df.t1)==False]
    t1=df['t1'].values*50/(1000*60)
    t2=df['t2'].values*50/(1000*60)
    smiles=df['smiles'].values
    rho=df['密度g/ml'].values
    V=df['V/ul'].values
    m=rho*V
    V_e=df['上样溶剂量/ul'].values
    e=df['上样溶剂'].values
    eluent=df['MeOH/H2O'].values
    speed=df['流速 ml/min'].values
    eluent_ratio=convert_eluent_ratio(eluent)
    eluent=convert_eluent(eluent)
    compound_index = df['compound_index'].values
    e=convert_e(e)
    #np.save('result_save/eluent_ratio.npy',eluent_ratio)
    data={'t1':t1,'t2':t2,'smiles':smiles,'m':m,'V_e':V_e,'e':e,'eluent':eluent,'speed':speed,
          'eluent_ratio':eluent_ratio,'compound_index':compound_index}
    return data

def read_data_CC_CN():
    df=pd.read_excel('dataset_CN_with_index.xlsx')
    df=df[df.t1!=-1]
    df=df[df.column_specs=='CN']
    df=df[np.isnan(df.t1)==False]
    t1=df['t1'].values*50/(1000*60)
    t2=df['t2'].values*50/(1000*60)
    smiles=df['smiles'].values
    rho=df['密度g/ml'].values
    V=df['V/ul'].values
    m=rho*V
    V_e=df['上样溶剂量/ul'].values
    e=df['上样溶剂'].values
    eluent=df['MeOH/H2O'].values
    speed=df['流速 ml/min'].values
    eluent_ratio=convert_eluent_ratio(eluent)
    eluent=convert_eluent(eluent)
    compound_index = df['compound_index'].values
    e=convert_e(e)
    #np.save('result_save/eluent_ratio.npy',eluent_ratio)
    data={'t1':t1,'t2':t2,'smiles':smiles,'m':m,'V_e':V_e,'e':e,'eluent':eluent,'speed':speed,
          'eluent_ratio':eluent_ratio,'compound_index':compound_index}
    return data

def read_data_CC_reversed():
    df=pd.read_excel('./dataset_reversed_with_index.xlsx')
    df=df[df.t1!=-1]
    # df=df[df.column_specs=='CN', 'C18', 'NH2']
    df=df[np.isnan(df.t1)==False]
    t1=df['t1'].values*50/(1000*60)
    t2=df['t2'].values*50/(1000*60)
    smiles=df['smiles'].values
    rho=df['密度g/ml'].values
    V=df['V/ul'].values
    m=rho*V
    V_e=df['上样溶剂量/ul'].values
    e=df['上样溶剂'].values
    eluent=df['MeOH/H2O'].values
    speed=df['流速 ml/min'].values
    C_c = df['Carbon_content'].values
    eluent_ratio=convert_eluent_ratio(eluent)
    eluent=convert_eluent(eluent)
    compound_index = df['compound_index'].values
    e=convert_e(e)
    #np.save('result_save/eluent_ratio.npy',eluent_ratio)
    data={'t1':t1,'t2':t2,'smiles':smiles,'m':m,'V_e':V_e,'e':e,'eluent':eluent,'speed':speed,
          'eluent_ratio':eluent_ratio,'compound_index':compound_index, 'C_c':C_c}
    return data


def calcu_mord_CC_40():
    data=read_data_CC_40()
    all_descriptor=[]
    for i in tqdm(range(len(data['smiles']))):
        smile = data['smiles'][i]
        mol = Chem.MolFromSmiles(smile)
        descriptor=mord(mol)
        all_descriptor.append(descriptor)
    all_descriptor=np.vstack(all_descriptor)
    np.save('dataset_save/dataset_40g_20240527.npy',all_descriptor)

def calcu_mord_CC_DCM():
    data=read_data_CC_DCM()
    all_descriptor=[]
    for i in tqdm(range(len(data['smiles']))):
        smile = data['smiles'][i]
        mol = Chem.MolFromSmiles(smile)
        descriptor=mord(mol)
        all_descriptor.append(descriptor)
    all_descriptor=np.vstack(all_descriptor)
    np.save('dataset_save/dataset_DCM.npy',all_descriptor)
    print("data keys:", data.keys())

def calcu_mord_CC_C18():
    data=read_data_CC_C18()
    all_descriptor=[]
    for i in tqdm(range(len(data['smiles']))):
        smile = data['smiles'][i]
        mol = Chem.MolFromSmiles(smile)
        descriptor=mord(mol)
        all_descriptor.append(descriptor)
    all_descriptor=np.vstack(all_descriptor)
    np.save('dataset_save/dataset_C18.npy',all_descriptor)
    print("data keys:", data.keys())

def calcu_mord_CC_NH2():
    data=read_data_CC_NH2()
    all_descriptor=[]
    for i in tqdm(range(len(data['smiles']))):
        smile = data['smiles'][i]
        mol = Chem.MolFromSmiles(smile)
        descriptor=mord(mol)
        all_descriptor.append(descriptor)
    all_descriptor=np.vstack(all_descriptor)
    np.save('dataset_save/dataset_NH2.npy',all_descriptor)
    print("data keys:", data.keys())

def calcu_mord_CC_CN():
    data=read_data_CC_CN()
    all_descriptor=[]
    for i in tqdm(range(len(data['smiles']))):
        smile = data['smiles'][i]
        mol = Chem.MolFromSmiles(smile)
        descriptor=mord(mol)
        all_descriptor.append(descriptor)
    all_descriptor=np.vstack(all_descriptor)
    np.save('dataset_save/dataset_CN.npy',all_descriptor)
    print("data keys:", data.keys())

def calcu_mord_CC_reversed():
    data=read_data_CC_reversed()
    all_descriptor=[]
    for i in tqdm(range(len(data['smiles']))):
        smile = data['smiles'][i]
        mol = Chem.MolFromSmiles(smile)
        descriptor=mord(mol)
        all_descriptor.append(descriptor)
    all_descriptor=np.vstack(all_descriptor)
    np.save('dataset_save/dataset_reversed.npy',all_descriptor)
    print("data keys:", data.keys())


def get_compound_index():
    df = pd.read_excel('dataset_C18.xlsx')
    df = df[df.t1 != -1]
    #df = df[df.column_specs == 'Silica-CS 4g']
    df = df[np.isnan(df.t1) == False]
    compound_index=[]
    all_smile=[]
    index=-1
    Origin=df['smiles']
    for smile in Origin:
        if smile not in all_smile:
            index+=1
            all_smile.append(smile)
        compound_index.append(index)
    df['compound_index']=compound_index
    df.to_excel('dataset_C18_with_index.xlsx')

# get_compound_index()
# calcu_mord_CC_DCM() 
# calcu_mord_CC_reversed()