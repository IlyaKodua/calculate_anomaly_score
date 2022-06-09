import glob
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
class config():

    def __init__(self, file):
        split_file = file.split("$")
        self.data_type = split_file[0]
        self.class_machin = split_file[1]
        self.isTrain = split_file[2] == "train"
        self.anomaly = "anomaly" in split_file[-1]
        self.normal = "normal" in split_file[-1]
        self.section = split_file[-1][0:10]

def get_len_of_embedding(file_path):
    with open(file_path, 'rb') as f:
        embedding = pickle.load(f)
    return len(embedding)

def get_mean_embeddings(folder, classes, sections):
    list_files = glob.glob(folder + '**/*.pkl', recursive=True)
    len_of_embedding = get_len_of_embedding(list_files[0])
    sum_embeddings = dict.fromkeys(classes, dict.fromkeys(sections, np.zeros(len_of_embedding)))
    mean_embeddings = dict.fromkeys(classes, dict.fromkeys(sections, np.zeros(len_of_embedding)))
    cnt_embeddings = dict.fromkeys(classes, dict.fromkeys(sections, 0))
    for file_path in list_files:
        with open(file_path, 'rb') as f:
            embedding = pickle.load(f)
        file = file_path.split("/")[-1]
        conf = config(file)
        if conf.isTrain and conf.normal:
            sum_embeddings[conf.class_machin][conf.section] += embedding
            cnt_embeddings[conf.class_machin][conf.section] += 1
    
    for cl in classes:
        for sc in sections:
            mean_embeddings[cl][sc] =  sum_embeddings[cl][sc] / cnt_embeddings[cl][sc]

    return mean_embeddings

def calculate_anamaly_score(mean_embeddings, classes,sections, folder):

    list_files = glob.glob(folder + '**/*.pkl', recursive=True)
    anomaly_score = dict.fromkeys(classes, dict.fromkeys(sections,[]))
    auc = dict.fromkeys(classes, dict.fromkeys(sections,0))
    pauc = dict.fromkeys(classes, dict.fromkeys(sections,0))
    labels = dict.fromkeys(classes, dict.fromkeys(sections,[]))
    for file_path in list_files:
        with open(file_path, 'rb') as f:
            embedding = pickle.load(f)
        file = file_path.split("/")[-1]
        conf = config(file)
        if ( not conf.isTrain ) and (conf.normal or conf.anomaly) :
            lvl = np.sqrt(np.mean((mean_embeddings[conf.class_machin][conf.section] - embedding)**2))
            anomaly_score[conf.class_machin][conf.section].append(lvl)
            labels[conf.class_machin][conf.section].append(int(conf.anomaly))
        
    for cl in classes:
        for sc in sections:
            auc[cl][sc] = metrics.roc_auc_score(labels[cl][sc], anomaly_score[cl][sc])
            pauc[cl][sc] = metrics.roc_auc_score(labels[cl][sc],  anomaly_score[cl][sc], max_fpr = 0.1)



    return auc, pauc


