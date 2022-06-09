import numpy as np


import numpy as np
from utils import *

# dir = "ast_embeddings_full/"
dir = "mobilenetv1_embeddings_full/"
classes = ["bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"]
sections = ["section_00", "section_01", "section_02", "section_03", "section_04", "section_05"]

multy = []

for cl in classes:
    for sc in sections:
        multy.append(cl + "_" + sc)


centrals_embeddings = get_mean_embeddings(dir, multy)

auc, pauc = calculate_anamaly_score(centrals_embeddings, multy, dir)
print(auc)