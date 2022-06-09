import numpy as np


import numpy as np
from utils import *

# a = dict({0:[],1:[]})

# a[0].append(0)
# a[1].append(1)
# a[0].append(2)



dir1 = "ast_embeddings_full/out/"
dir2 = "mobilenetv1_embeddings_full/out/"
classes = ["bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"]
sections = ["section_00", "section_01", "section_02", "section_03", "section_04", "section_05"]

multy = []

for cl in classes:
    for sc in sections:
        multy.append(cl + "_" + sc)


knn= get_mean_embeddings( multy,  dir1, dir2)


auc, pauc = calculate_anamaly_score(knn, multy, dir1, dir2)
print(1)
auc_all = 0
pauc_all = 0
cnt = 0
for cl in multy:
    if auc[cl] != 0:
       auc_all += auc[cl]
       pauc_all += pauc[cl]
       cnt += 1
auc_all /= cnt    
pauc_all /= cnt    
print(auc_all)
print(pauc_all)