import glob
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from scipy.stats import hmean

class config():

    def __init__(self, file):
        split_file = file.split("$")
        self.data_type = split_file[0]
        self.class_machin = split_file[1]
        self.isTrain = split_file[2] == "train"
        self.anomaly = "anomaly" in split_file[-1]
        self.normal = "normal" in split_file[-1]
        self.section = split_file[-1][0:10]
        self.is_target = "target" in split_file[-1]
        self.is_source = "source" in split_file[-1]

def get_len_of_embedding(file_path):
    with open(file_path, 'rb') as f:
        embedding = pickle.load(f)
    return len(embedding)

def init_dict_array(classes, sections):
    _dict = dict()

    for cl in classes:
        _dict[cl] = dict()

    for cl in classes:
        for sc in sections:
            _dict[cl][sc] = []
    return _dict

def init_dict_zeros(classes, sections):
    _dict = dict()

    for cl in classes:
        _dict[cl] = dict()

    for cl in classes:
        for sc in sections:
            _dict[cl][sc] = 0
    return _dict


def init_dict_zeros2(classes):
    _dict = dict()

    for cl in classes:
        _dict[cl] = 0
    return _dict

def init_dict_knn(classes, sections):
    _dict = dict()

    for cl in classes:
        _dict[cl] = dict()

    for cl in classes:
        for sc in sections:
            _dict[cl][sc] = NearestNeighbors(n_neighbors=1,  metric='cosine')
    return _dict


def calc_harmonic_mean(auc, classes, sections):
    mean_auc = init_dict_zeros2(classes)

    for cl in classes:
        auc_arr = []
        for sc in sections:
            if auc[cl][sc] > 1e-3:
                auc_arr.append(auc[cl][sc])
        mean_auc[cl] = hmean(auc_arr)
    return mean_auc
    


def calc_mean_thres(thres, classes, sections):
    mean_thres = init_dict_zeros2(classes)

    for cl in classes:
        thres_arr = []
        for sc in sections:
            if thres[cl][sc] > 1e-3:
                thres_arr.append(thres[cl][sc])
        mean_thres[cl] = np.mean(thres_arr)
    return mean_thres


def get_mean_embeddings(classes, sections, dir1, dir2):
    list_files = glob.glob(dir1 + '**/*.pkl', recursive=True)

    embeddings_all = init_dict_array(classes, sections)

    knn_my = init_dict_knn(classes, sections)
    for i,file_path in enumerate(list_files):
        file = file_path.split("/")[-1]
        embedding1 = pickle.load(open(dir1 + file, 'rb'))
        embedding2 = pickle.load(open(dir2 + file, 'rb'))
        embedding = np.concatenate((embedding1,embedding2), axis=0)
        conf = config(file)
        if conf.isTrain and (conf.normal) and conf.is_source:
            embeddings_all[conf.class_machin][conf.section].append(embedding)
    
    for cl in classes:
        for sc in sections:
            knn_my[cl][sc].fit(embeddings_all[cl][sc])

    return knn_my


def calculate_anamaly_score_source(knn, classes, sections, dir1, dir2):

    list_files1 = glob.glob(dir1 + '**/*.pkl', recursive=True)
    auc = init_dict_zeros(classes, sections)
    pauc = init_dict_zeros(classes, sections)
    labels = init_dict_array(classes, sections)
    embeddings_all = init_dict_array(classes, sections)
    for file_path in list_files1:
        file = file_path.split("/")[-1]
        embedding1 = pickle.load(open(dir1 + file, 'rb'))
        embedding2 = pickle.load(open(dir2 + file, 'rb'))
        embedding = np.concatenate((embedding1,embedding2), axis=0)
        conf = config(file)
        if ( not conf.isTrain ) and (conf.normal or conf.anomaly) and conf.is_source:
            embeddings_all[conf.class_machin][conf.section].append(embedding)
            labels[conf.class_machin][conf.section].append(int(conf.anomaly))

    
    for cl in classes:
        for sc in sections:
            if embeddings_all[cl][sc]:
                lvl,_ = knn[cl][sc].kneighbors(embeddings_all[cl][sc])
                auc[cl][sc] = metrics.roc_auc_score(labels[cl][sc], lvl)
                pauc[cl][sc] = metrics.roc_auc_score(labels[cl][sc],  lvl, max_fpr = 0.1)
    
    return auc, pauc


def calculate_anamaly_score_target(knn, classes, sections, dir1, dir2):

    list_files1 = glob.glob(dir1 + '**/*.pkl', recursive=True)
    auc = init_dict_zeros(classes, sections)
    pauc = init_dict_zeros(classes, sections)
    labels = init_dict_array(classes, sections)
    embeddings_all = init_dict_array(classes, sections)
    for file_path in list_files1:
        file = file_path.split("/")[-1]
        embedding1 = pickle.load(open(dir1 + file, 'rb'))
        embedding2 = pickle.load(open(dir2 + file, 'rb'))
        embedding = np.concatenate((embedding1,embedding2), axis=0)
        conf = config(file)
        if ( not conf.isTrain ) and (conf.normal or conf.anomaly) and conf.is_target:
            embeddings_all[conf.class_machin][conf.section].append(embedding)
            labels[conf.class_machin][conf.section].append(int(conf.anomaly))

    
    for cl in classes:
        for sc in sections:
            if embeddings_all[cl][sc]:
                lvl,_ = knn[cl][sc].kneighbors(embeddings_all[cl][sc])
                auc[cl][sc] = metrics.roc_auc_score(labels[cl][sc], lvl)
                pauc[cl][sc] = metrics.roc_auc_score(labels[cl][sc],  lvl, max_fpr = 0.1)
    
    return auc, pauc




def calculate_anamaly_score_all(knn, classes, sections, dir1, dir2):

    list_files1 = glob.glob(dir1 + '**/*.pkl', recursive=True)
    auc = init_dict_zeros(classes, sections)
    pauc = init_dict_zeros(classes, sections)
    labels = init_dict_array(classes, sections)
    embeddings_all = init_dict_array(classes, sections)
    for file_path in list_files1:
        file = file_path.split("/")[-1]
        embedding1 = pickle.load(open(dir1 + file, 'rb'))
        embedding2 = pickle.load(open(dir2 + file, 'rb'))
        embedding = np.concatenate((embedding1,embedding2), axis=0)
        conf = config(file)
        if ( not conf.isTrain ) and (conf.normal or conf.anomaly):
            embeddings_all[conf.class_machin][conf.section].append(embedding)
            labels[conf.class_machin][conf.section].append(int(conf.anomaly))

    
    for cl in classes:
        for sc in sections:
            if embeddings_all[cl][sc]:
                lvl,_ = knn[cl][sc].kneighbors(embeddings_all[cl][sc])
                auc[cl][sc] = metrics.roc_auc_score(labels[cl][sc], lvl)
                pauc[cl][sc] = metrics.roc_auc_score(labels[cl][sc],  lvl, max_fpr = 0.1)
    
    return auc, pauc


def calculate_thres(knn, classes, sections, dir1, dir2):

    list_files1 = glob.glob(dir1 + '**/*.pkl', recursive=True)
    thresholds = init_dict_zeros(classes, sections)
    labels = init_dict_array(classes, sections)
    embeddings_all = init_dict_array(classes, sections)
    for file_path in list_files1:
        file = file_path.split("/")[-1]
        embedding1 = pickle.load(open(dir1 + file, 'rb'))
        embedding2 = pickle.load(open(dir2 + file, 'rb'))
        embedding = np.concatenate((embedding1,embedding2), axis=0)
        conf = config(file)
        if ( not conf.isTrain ) and (conf.normal or conf.anomaly):
            embeddings_all[conf.class_machin][conf.section].append(embedding)
            labels[conf.class_machin][conf.section].append(int(conf.anomaly))

    
    for cl in classes:
        for sc in sections:
            if embeddings_all[cl][sc]:
                lvl,_ = knn[cl][sc].kneighbors(embeddings_all[cl][sc])
                fpr, tpr, threshold =roc_curve(labels[cl][sc], lvl)
                idx = np.argmax(np.sqrt(tpr * (1 - fpr)))
                thresholds[cl][sc] = threshold[idx]
    
    return calc_mean_thres(thresholds, classes, sections)