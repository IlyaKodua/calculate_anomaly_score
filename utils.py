import glob
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
class config():

    def __init__(self, file):
        split_file = file.split("$")
        self.data_type = split_file[0]
        self.class_machin = split_file[1] + "_" + split_file[-1][0:10]
        self.isTrain = split_file[2] == "train"
        self.anomaly = "anomaly" in split_file[-1]
        self.normal = "normal" in split_file[-1]

def get_len_of_embedding(file_path):
    with open(file_path, 'rb') as f:
        embedding = pickle.load(f)
    return len(embedding)

def init_dict_array(classes):
    embeddings_all = dict()

    for cl in classes:
        embeddings_all[cl] = []
    return embeddings_all

def init_dict_zeros(classes):
    embeddings_all = dict()

    for cl in classes:
        embeddings_all[cl] = 0
    return embeddings_all


def init_dict_knn(classes):
    embeddings_all = dict()

    for cl in classes:
        embeddings_all[cl] = NearestNeighbors(n_neighbors=1,  metric='cosine')
    return embeddings_all

def get_mean_embeddings(classes, dir1, dir2):
    list_files = glob.glob(dir1 + '**/*.pkl', recursive=True)

    embeddings_all = init_dict_array(classes)

    knn_my = init_dict_knn(classes)
    for i,file_path in enumerate(list_files):
        file = file_path.split("/")[-1]
        embedding1 = pickle.load(open(dir1 + file, 'rb'))
        embedding2 = pickle.load(open(dir2 + file, 'rb'))
        embedding = np.concatenate((embedding1,embedding2), axis=0)
        conf = config(file)
        if conf.isTrain and (conf.normal):
            embeddings_all[conf.class_machin].append(embedding)
    
    for cl in classes:
        knn_my[cl].fit(embeddings_all[cl])

    return knn_my

def calculate_anamaly_score(knn, classes, dir1, dir2):

    list_files1 = glob.glob(dir1 + '**/*.pkl', recursive=True)
    auc = init_dict_zeros(classes)
    pauc = init_dict_zeros(classes)
    labels = init_dict_array(classes)
    embeddings_all = init_dict_array(classes)
    for file_path in list_files1:
        file = file_path.split("/")[-1]
        embedding1 = pickle.load(open(dir1 + file, 'rb'))
        embedding2 = pickle.load(open(dir2 + file, 'rb'))
        embedding = np.concatenate((embedding1,embedding2), axis=0)
        conf = config(file)
        if ( not conf.isTrain ) and (conf.normal or conf.anomaly) :
            embeddings_all[conf.class_machin].append(embedding)
            labels[conf.class_machin].append(int(conf.anomaly))

    
    for cl in classes:

        if embeddings_all[cl]:
            lvl,_ = knn[cl].kneighbors(embeddings_all[cl])
            auc[cl] = metrics.roc_auc_score(labels[cl], lvl)
            pauc[cl] = metrics.roc_auc_score(labels[cl],  lvl, max_fpr = 0.1)

    return auc, pauc


