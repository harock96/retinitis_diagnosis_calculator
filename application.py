from flask import Flask, render_template, request
import sys
import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from plotter import TensorboardPlotter
from dataset import AdversarialDataset
from model import *
from utils import plot_roc_curve, make_model_name_dir
import matplotlib.pyplot as plt
import math

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--hidden_unit', type=int, default=128, help='hidden unit')
parser.add_argument('--load_model_path', type=str, default='./model/Feature15/ADMTL/smote.pth', help='load model path')
parser.add_argument('--log_dir', type=str, default='./log/ADMTL_fs_test/', help='log dir')
args = vars(parser.parse_args())

# Fix random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Plotter
plotter = TensorboardPlotter(args['log_dir'])

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = AdversarialMTL(hidden_unit=args['hidden_unit'])
model.to(device)
model.load_state_dict(torch.load(args['load_model_path'], map_location=device))
print("Model loaded")

application = Flask(__name__)
application.config['JSON_AS_ASCII'] = False

@application.route('/', methods = ['GET', 'POST'])

def index(): 
  if request.method == 'GET' :
    return render_template("demo.html")
  elif request.method == 'POST':
    features = {}
    
    if request.form.get("CMV IgM") != '':
      cmv_igm = (request.form.get("CMV IgM", type=float) - 0.086)/0.15
    else:
      cmv_igm = 0
    features['CMV IgM[Serum]'] = cmv_igm 

    if request.form.get("CMV IgG") != '':
      cmv_igg = (request.form.get("CMV IgG", type = float)-40.971)/17.456
    else:
      cmv_igg =0
    features['CMV IgG[Serum]'] = cmv_igg 

    if request.form.get("HSV IgM") != '':
      hsv_igm = (request.form.get("HSV IgM", type = float)-0.248)/0.218
    else:
      hsv_igm = 0
    features['HSV IgM[Serum]'] = hsv_igm

    if request.form.get("HSV IgG") != '':
      hsv_igg = (request.form.get("HSV IgG", type = float)-48.021)/22.129
    else:
      hsv_igg = 0
    features['HSV IgG[Serum]'] = hsv_igg

    if request.form.get("VZV IgM") != '':
      vzv_igm = (request.form.get("VZV IgM", type = float) - 0.16)/0.202
    else:
      vzv_igm =0
    features['VZV IgM[Serum]'] = vzv_igm

    if request.form.get("VZV IgG") != '':
      vzv_igg = (request.form.get("VZV IgG", type = float) - 824.622)/516.633
    else:
      vzv_igg = 0
    features['VZV IgG[Serum]'] = vzv_igg 

    if request.form.get("WBC") != '':
      wbc = (request.form.get("WBC", type = float) - 8.169)/8.675
    else:
      wbc = 0
    features['WBC COUNT[Whole blood]'] = wbc 

    if request.form.get("Lym#") != '':
      lym_s = (request.form.get("Lym#", type = float) - 2.055)/0.853
    else:
      lym_s = 0
    features['Lymphocyte(#)[Whole blood]'] = lym_s 

    if request.form.get("Lym%") != '':
      lym_p = (request.form.get("Lym%", type = float)-29.67)/11.226
    else:
      lym_p = 0
    features['Lymphocyte(%)[Whole blood]'] = lym_p

    if request.form.get("Mono#") != '':
      mono_s = (request.form.get("Mono#", type = float)-0.476)/0.289
    else:
      mono_s = 0
    features['Monocyte(#)[Whole blood]'] = mono_s

    if request.form.get("Mono%") != '':
      mono_p = (request.form.get("Mono%", type = float)-6.637)/3.486
    else:
      mono_p = 0
    features['Monocyte(%)[Whole blood]'] = mono_p

    if request.form.get("Neu#") != '':
      neu_s = (request.form.get("Neu#", type = float)-5.029)/2.633
    else:
      neu_s = 0
    features['Neutrophil(#)[Whole blood]'] = neu_s

    if request.form.get("Neu%") != '':
      neu_p= (request.form.get("Neu%", type = float)-62.774)/11.61
    else:
      neu_p = 0
    features['Neutrophil(%)[Whole blood]'] = neu_p

    if request.form.get("ESR") != '':
      esr = (request.form.get("ESR", type = float)-24.776)/19.647
    else:
      esr = 0
    features['ESR[Whole blood]'] = esr 

    if request.form.get("CRP") != '':
      crp = (request.form.get("CRP", type = float)-14.958)/37.84
    else:
      crp = 0
    features['CRP[Serum]'] = crp

    features['Diagnosis'] = 0
    
    df = pd.DataFrame([features])
    df.to_csv('./data/demo_data/demo.csv', index = False)

    # Create dataset
    demo_data = AdversarialDataset(mode='demo')
    # Create dataloader
    demo_loader = DataLoader(demo_data, batch_size=1, shuffle=False)

    # Test
    model.eval()
    test_labels_for_ARN = np.array([])
    test_pred_for_ARN = np.array([])
    test_labels_for_CMV = np.array([])
    test_pred_for_CMV = np.array([])

    for i, (feature, label_for_ARN, label_for_CMV, label_for_adv) in enumerate(tqdm(demo_loader)):
        feature = feature.to(device)
        label_for_ARN = label_for_ARN.to(device)
        label_for_CMV = label_for_CMV.to(device)

        # Predict
        with torch.no_grad():
            output_for_ARN, output_for_CMV, _, _ = model(feature)
            output_for_ARN = output_for_ARN.squeeze()
            output_for_CMV = output_for_CMV.squeeze()
            pred_for_ARN = output_for_ARN
            pred_for_CMV = output_for_CMV
        
        # Save labels and predictions
        test_labels_for_ARN = np.append(test_labels_for_ARN, label_for_ARN.cpu().numpy())
        test_pred_for_ARN = np.append(test_pred_for_ARN, pred_for_ARN.cpu().numpy())
        test_labels_for_CMV = np.append(test_labels_for_CMV, label_for_CMV.cpu().numpy())
        test_pred_for_CMV = np.append(test_pred_for_CMV, pred_for_CMV.cpu().numpy())
    
    test_pred_for_ARN = str(round(float(test_pred_for_ARN),3))
    test_pred_for_CMV = str(round(float(test_pred_for_CMV),3))

    if float(test_pred_for_ARN) <= 0.442:
      prob_positive_ARN = 0.5/0.442 * float(test_pred_for_ARN)
    else:
      prob_positive_ARN = 0.5 + 0.5/0.558 * (float(test_pred_for_ARN) - 0.442)

    if float(test_pred_for_CMV) <= 0.378:
      prob_positive_CMV = 0.5/0.378 * float(test_pred_for_CMV)
    else:
      prob_positive_CMV = 0.5 + 0.5/0.622 * (float(test_pred_for_CMV) - 0.378)
    
    prob_negative_ARN = 1-prob_positive_ARN
    prob_negative_CMV = 1-prob_positive_CMV

    prob_ARN = prob_positive_ARN * prob_negative_CMV
    prob_CMV = prob_negative_ARN * prob_positive_CMV
    prob_NIU = prob_negative_ARN * prob_negative_CMV

    prob_total_ARN = round(prob_ARN / (prob_ARN + prob_CMV + prob_NIU),10) * 100
    prob_total_CMV = round(prob_CMV / (prob_ARN + prob_CMV + prob_NIU),10) * 100
    prob_total_NIU = round(prob_NIU / (prob_ARN + prob_CMV + prob_NIU),10) * 100

    prob_list = [prob_total_ARN, prob_total_CMV, prob_total_NIU]
    
    if max(prob_list) == prob_total_ARN :
      prob_max = "ARN"
    elif max(prob_list) == prob_total_CMV:
      prob_max = "CMV"
    else:
      prob_max = "NIU-PS"

    prob_total_ARN = round(prob_total_ARN)
    prob_total_CMV = round(prob_total_CMV)
    
    if prob_total_ARN + prob_total_CMV < 100:
      prob_total_NIU = 100 - prob_total_ARN - prob_total_CMV
    elif prob_total_ARN + prob_total_CMV == 100:
      prob_total_NIU = 0
    elif prob_total_ARN + prob_total_CMV == 101:
      tmp = min([prob_total_ARN, prob_total_CMV]) 
      if prob_total_ARN == tmp :
        prob_total_ARN -=1
      elif prob_total_CMV == tmp :
        prob_total_CMV -= 1
      prob_total_NIU = 0
    elif prob_total_ARN + prob_total_CMV == 102:
      prob_total_ARN -= 1
      prob_total_CMV -= 1
      prob_total_NIU = 0

  return render_template('demo.html', 'index.html', ARN = test_pred_for_ARN, CMV = test_pred_for_CMV,
   prob_total_ARN = prob_total_ARN, prob_total_CMV = prob_total_CMV, prob_total_NIU = prob_total_NIU, prob_max = prob_max)
  
if __name__ == "__main__":
  application.run(host = "0.0.0.0", port = "5000", debug=True)