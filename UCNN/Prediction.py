from PIL import Image as img
import os
from tqdm.notebook import tqdm
import numpy as np


# models = ["UNet-", "Unet+", "ResUnet"]
for model in models:
    tp_path = "tp/"
    fp_path = "fp/"
    fn_path = "fn/"

    Tp = []
    Fp = []
    Fn = []
    Tn = []
    Total = []

    ids = next(os.walk(tp_path + "//"))[2]

    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        tp_img = img.open(tp_path + str(id_))
        fp_img = img.open(fp_path + str(id_))
        fn_img = img.open(fn_path + str(id_))
        tp = np.sum( np.array(tp_img) == 255)
        fp = np.sum( np.array(fp_img) == 255)
        fn = np.sum( np.array(fn_img) == 255)

        temp = np.array(fp_img)
        total = temp.shape[0] * temp.shape[1]
        tn = total-tp-fp-fn

        Tp.append(tp)
        Fp.append(fp)
        Fn.append(fn)
        Tn.append(tn)
        Total.append(total)

    TP = np.array(Tp)
    FP = np.array(Fp)
    FN = np.array(Fn)
    TN = np.array(Tn)
    TOTAL = np.array(Total)
    
    tp = int(np.sum(TP)) 
    fp = int(np.sum(FP)) 
    fn = int(np.sum(FN)) 
    tn = int(np.sum(TN)) 
    total = tp + fp + fn + tn

    acc = (tp + tn) / total
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * pre * rec / (pre + rec)
    print("Accuracy:{:.2},  Precision:{:.2}, Recall{:.2}, F1:{:.2}".format(model, acc, pre, rec, f1))
    print(tp, fp, tn, fn)
    
    