import numpy as np


import numpy as np
from utils import *

# a = dict({0:[],1:[]})

# a[0].append(0)
# a[1].append(1)
# a[0].append(2)

def print_auc(auc, name):

    for key, v in auc.items():
        print(key, name, v)

dir1 = "ast_embeddings_full/out/"
dir2 = "mobilenetv1_embeddings_full/out/"
classes = ["bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"]
sections = ["section_00", "section_01", "section_02", "section_03", "section_04", "section_05"]




knn= get_mean_embeddings( classes, sections, dir1, dir2)


# auc, pauc = calculate_anamaly_score_source(knn, classes,sections, dir1, dir2)

# auc, pauc = calculate_anamaly_score_target(knn, classes,sections, dir1, dir2)


# auc, pauc = calculate_anamaly_score_all(knn, classes,sections, dir1, dir2)

# print_auc(calc_harmonic_mean(pauc, classes, sections), "pauc")


thres =  calculate_thres(knn, classes,sections, dir1, dir2)

print_auc(thres, "tres")


print(0)
