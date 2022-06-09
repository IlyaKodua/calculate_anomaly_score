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

def get_mean_embeddings(folder, classes):
    list_files = glob.glob(folder + '**/*.pkl', recursive=True)
    len_of_embedding = get_len_of_embedding(list_files[0])
    embeddings_all = dict.fromkeys(classes,  [])
    labels_all = dict.fromkeys(classes, [])
    knn_my = dict.fromkeys(classes,  NearestNeighbors(n_neighbors=1,  metric='cosine'))
    for file_path in list_files:
        embedding = pickle.load(open(file_path, 'rb'))
        file = file_path.split("/")[-1]
        conf = config(file)
        if conf.isTrain and (conf.normal):
            embeddings_all.append(embedding)
            labels_all.append(np.where(conf.class_machin == classes))
    
    for cl in classes:
        knn_my[cl].fit(embeddings_all[cl], labels_all[cl])

    return knn_my

def calculate_anamaly_score(knn_my, classes, folder):

    list_files = glob.glob(folder + '**/*.pkl', recursive=True)
    anomaly_score = dict.fromkeys(classes,[])
    auc = dict.fromkeys(classes,0)
    pauc = dict.fromkeys(classes, 0)
    labels = dict.fromkeys(classes, [])
    embeddings_all = dict.fromkeys(classes, [])
    for file_path in list_files:
        embedding = pickle.load(open(file_path, 'rb'))
        file = file_path.split("/")[-1]
        conf = config(file)
        if ( not conf.isTrain ) and (conf.normal or conf.anomaly) :
            embeddings_all[conf.class_machin].append(embedding)
            labels[conf.class_machin].append(int(conf.anomaly))
    
    
    for cl in classes:
            lvl,_ = knn_my[cl].kneighbors(embeddings_all[cl])
            auc[cl] = metrics.roc_auc_score(labels[cl], lvl)
            pauc[cl] = metrics.roc_auc_score(labels[cl],  lvl, max_fpr = 0.1)

    return auc, pauc


