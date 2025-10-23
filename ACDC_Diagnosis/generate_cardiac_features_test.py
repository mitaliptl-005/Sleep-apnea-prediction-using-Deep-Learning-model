"""
Updated code for extraction of cardiac features from ACDC testing dataset
with frame-based filenames and ED/ES info read from training patient folders
"""
import os
import re
import numpy as np
import pandas as pd
import nibabel as nib
import sys
import configparser

# Custom
sys.path.append("../")
from utils_heart import *

HEADER = ["Name", "ED[vol(LV)]", "ES[vol(LV)]", "ED[vol(RV)]", "ES[vol(RV)]",
          "ED[mass(MYO)]", "ES[vol(MYO)]", "EF(LV)", "EF(RV)", "ED[vol(LV)/vol(RV)]", "ES[vol(LV)/vol(RV)]",
          "ED[mass(MYO)/vol(LV)]", "ES[vol(MYO)/vol(LV)]",
          "ES[max(mean(MWT|SA)|LA)]", "ES[stdev(mean(MWT|SA)|LA)]", "ES[mean(stdev(MWT|SA)|LA)]", "ES[stdev(stdev(MWT|SA)|LA)]",
          "ED[max(mean(MWT|SA)|LA)]", "ED[stdev(mean(MWT|SA)|LA)]", "ED[mean(stdev(MWT|SA)|LA)]", "ED[stdev(stdev(MWT|SA)|LA)]",
          "GROUP"]

def read_info_file(info_file, patient_no):
    """Reads the Info.cfg file and returns ED/ES frame for the given patient number"""
    config = configparser.ConfigParser()
    config.read(info_file)
    section_name = f'patient{patient_no:03d}'
    if section_name not in config:
        return None, None
    ed_frame = int(config[section_name]['ED'])
    es_frame = int(config[section_name]['ES'])
    return ed_frame, es_frame

def calculate_metrics_from_pred(test_data_path, training_data_path, pred_name='prediction'):
    if not os.path.exists(test_data_path):
        print(f"Error: Testing data path '{test_data_path}' does not exist.")
        return

    res = []

    # Iterate over files in testing data folder
    for patient_file in sorted(os.listdir(test_data_path)):
        m = re.match(r'patient(\d{3})_frame\d{2}\.nii', patient_file)
        if not m:
            continue
        patient_No = int(m.group(1))

        # Training Info.cfg path
        training_info_file = os.path.join(training_data_path, f'patient{patient_No:03d}', 'Info.cfg')
        if not os.path.exists(training_info_file):
            print(f"Warning: Info.cfg missing for patient {patient_No:03d}, skipping.")
            continue

        # Read ED/ES frames for this patient
        ed_frame, es_frame = read_info_file(training_info_file, patient_No)
        if ed_frame is None or es_frame is None:
            print(f"Warning: ED/ES info missing in Info.cfg for patient {patient_No:03d}, skipping.")
            continue

        # Construct testing frame file paths
        ed_file = os.path.join(test_data_path, f'patient{patient_No:03d}_frame{ed_frame:02d}.nii')
        es_file = os.path.join(test_data_path, f'patient{patient_No:03d}_frame{es_frame:02d}.nii')

        if not os.path.exists(ed_file) or not os.path.exists(es_file):
            print(f"Warning: ED or ES frame file missing for patient {patient_No:03d}, skipping.")
            continue

        # Load NIfTI data
        ed_data = nib.load(ed_file)
        es_data = nib.load(es_file)

        ed_lv, ed_rv, ed_myo = heart_metrics(ed_data.get_fdata(), ed_data.header.get_zooms())
        es_lv, es_rv, es_myo = heart_metrics(es_data.get_fdata(), es_data.header.get_zooms())

        ef_lv = ejection_fraction(ed_lv, es_lv)
        ef_rv = ejection_fraction(ed_rv, es_rv)

        # Myocardial thickness
        es_props = myocardial_thickness(es_file)
        ed_props = myocardial_thickness(ed_file)

        heart_param = {
            'EDV_LV': ed_lv, 'EDV_RV': ed_rv, 'ESV_LV': es_lv, 'ESV_RV': es_rv,
            'ED_MYO': ed_myo, 'ES_MYO': es_myo, 'EF_LV': ef_lv, 'EF_RV': ef_rv,
            'ES_MYO_MAX_AVG_T': np.amax(es_props[0]), 'ES_MYO_STD_AVG_T': np.std(es_props[0]),
            'ES_MYO_AVG_STD_T': np.mean(es_props[1]), 'ES_MYO_STD_STD_T': np.std(es_props[1]),
            'ED_MYO_MAX_AVG_T': np.amax(ed_props[0]), 'ED_MYO_STD_AVG_T': np.std(ed_props[0]),
            'ED_MYO_AVG_STD_T': np.mean(ed_props[1]), 'ED_MYO_STD_STD_T': np.std(ed_props[1]),
        }

        row = [
            f'patient{patient_No:03d}',
            heart_param['EDV_LV'], heart_param['ESV_LV'],
            heart_param['EDV_RV'], heart_param['ESV_RV'],
            heart_param['ED_MYO'], heart_param['ES_MYO'],
            heart_param['EF_LV'], heart_param['EF_RV'],
            ed_lv/ed_rv if ed_rv != 0 else 0,
            es_lv/es_rv if es_rv != 0 else 0,
            ed_myo/ed_lv if ed_lv != 0 else 0,
            es_myo/es_lv if es_lv != 0 else 0,
            heart_param['ES_MYO_MAX_AVG_T'], heart_param['ES_MYO_STD_AVG_T'],
            heart_param['ES_MYO_AVG_STD_T'], heart_param['ES_MYO_STD_STD_T'],
            heart_param['ED_MYO_MAX_AVG_T'], heart_param['ED_MYO_STD_AVG_T'],
            heart_param['ED_MYO_AVG_STD_T'], heart_param['ED_MYO_STD_STD_T'],
            ''  # Placeholder for GROUP
        ]
        res.append(row)

    # Save CSV directly in training_data folder
    output_dir = os.path.join('.', 'training_data')
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"Cardiac_parameters_{pred_name}.csv")
    df = pd.DataFrame(res, columns=HEADER)
    df.to_csv(csv_path, index=False)

    print(f"Saved cardiac parameters CSV to '{csv_path}'")


if __name__ == '__main__':
    test_prediction_path = r'D:/MSDS/Data Minning/Project/Code/ACDC_dataset/testing'
    training_data_path = r'D:/MSDS/Data Minning/Project/Code/ACDC_dataset/training'
    calculate_metrics_from_pred(test_prediction_path, training_data_path, pred_name='prediction')
