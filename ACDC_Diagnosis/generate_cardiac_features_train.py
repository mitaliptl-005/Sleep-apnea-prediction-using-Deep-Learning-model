"""
    This code is for extraction of cardiac features from ACDC database segmentations
"""
import os, re
import numpy as np
import pandas as pd
import nibabel as nib
# Custom
from utils_heart import * 

HEADER = ["Name", "ED[vol(LV)]", "ES[vol(LV)]", "ED[vol(RV)]", "ES[vol(RV)]",
          "ED[mass(MYO)]", "ES[vol(MYO)]", "EF(LV)", "EF(RV)", "ED[vol(LV)/vol(RV)]", "ES[vol(LV)/vol(RV)]", 
          "ED[mass(MYO)/vol(LV)]", "ES[vol(MYO)/vol(LV)]",
          "ES[max(mean(MWT|SA)|LA)]", "ES[stdev(mean(MWT|SA)|LA)]", "ES[mean(stdev(MWT|SA)|LA)]", 
          "ES[stdev(stdev(MWT|SA)|LA)]", 
          "ED[max(mean(MWT|SA)|LA)]", "ED[stdev(mean(MWT|SA)|LA)]", "ED[mean(stdev(MWT|SA)|LA)]", 
          "ED[stdev(stdev(MWT|SA)|LA)]", "GROUP"]

def calculate_metrics_for_training(data_path_list, name='train'):
    res = []
    for data_path in data_path_list:
        if not os.path.exists(data_path):
            print(f"[WARNING] Path does not exist: {data_path}")
            continue

        patient_folder_list = os.listdir(data_path)
        for patient in patient_folder_list:
            print(f"Processing {patient}...")
            patient_folder_path = os.path.join(data_path, patient)
            config_file_path = os.path.join(patient_folder_path, 'Info.cfg')

            if not os.path.exists(config_file_path):
                print(f"[WARNING] Info.cfg not found for {patient}")
                continue

            patient_data = {}
            with open(config_file_path) as f_in:
                for line in f_in:
                    l = line.rstrip().split(": ")
                    patient_data[l[0]] = l[1]

            # Read patient number
            m = re.match(r"patient(\d{3})", patient)
            patient_No = int(m.group(1))
            # Diastole and Systole frame numbers
            ED_frame_No = int(patient_data['ED'])
            ES_frame_No = int(patient_data['ES'])
            ed_img = f"patient{patient_No:03d}_frame{ED_frame_No:02d}_gt.nii.gz"
            es_img = f"patient{patient_No:03d}_frame{ES_frame_No:02d}_gt.nii.gz"

            pid = f'patient{patient_No:03d}'
            # Load NIfTI images
            ed_data = nib.load(os.path.join(data_path, pid, ed_img))
            es_data = nib.load(os.path.join(data_path, pid, es_img))

            # Heart metrics
            ed_lv, ed_rv, ed_myo = heart_metrics(ed_data.get_fdata(), ed_data.header.get_zooms())
            es_lv, es_rv, es_myo = heart_metrics(es_data.get_fdata(), es_data.header.get_zooms())
            ef_lv = ejection_fraction(ed_lv, es_lv)
            ef_rv = ejection_fraction(ed_rv, es_rv)

            # Myocardial thickness
            myo_properties_es = myocardial_thickness(os.path.join(data_path, pid, es_img))
            es_myo_thickness_max_avg = np.amax(myo_properties_es[0])
            es_myo_thickness_std_avg = np.std(myo_properties_es[0])
            es_myo_thickness_mean_std = np.mean(myo_properties_es[1])
            es_myo_thickness_std_std = np.std(myo_properties_es[1])

            myo_properties_ed = myocardial_thickness(os.path.join(data_path, pid, ed_img))
            ed_myo_thickness_max_avg = np.amax(myo_properties_ed[0])
            ed_myo_thickness_std_avg = np.std(myo_properties_ed[0])
            ed_myo_thickness_mean_std = np.mean(myo_properties_ed[1])
            ed_myo_thickness_std_std = np.std(myo_properties_ed[1])

            heart_param = {
                'EDV_LV': ed_lv, 'EDV_RV': ed_rv, 'ESV_LV': es_lv, 'ESV_RV': es_rv,
                'ED_MYO': ed_myo, 'ES_MYO': es_myo, 'EF_LV': ef_lv, 'EF_RV': ef_rv,
                'ES_MYO_MAX_AVG_T': es_myo_thickness_max_avg, 'ES_MYO_STD_AVG_T': es_myo_thickness_std_avg,
                'ES_MYO_AVG_STD_T': es_myo_thickness_mean_std, 'ES_MYO_STD_STD_T': es_myo_thickness_std_std,
                'ED_MYO_MAX_AVG_T': ed_myo_thickness_max_avg, 'ED_MYO_STD_AVG_T': ed_myo_thickness_std_avg,
                'ED_MYO_AVG_STD_T': ed_myo_thickness_mean_std, 'ED_MYO_STD_STD_T': ed_myo_thickness_std_std,
            }

            r = [
                pid,
                heart_param['EDV_LV'], heart_param['ESV_LV'],
                heart_param['EDV_RV'], heart_param['ESV_RV'],
                heart_param['ED_MYO'], heart_param['ES_MYO'],
                heart_param['EF_LV'], heart_param['EF_RV'],
                ed_lv / ed_rv, es_lv / es_rv,
                ed_myo / ed_lv, es_myo / es_lv,
                heart_param['ES_MYO_MAX_AVG_T'], heart_param['ES_MYO_STD_AVG_T'],
                heart_param['ES_MYO_AVG_STD_T'], heart_param['ES_MYO_STD_STD_T'],
                heart_param['ED_MYO_MAX_AVG_T'], heart_param['ED_MYO_STD_AVG_T'],
                heart_param['ED_MYO_AVG_STD_T'], heart_param['ED_MYO_STD_STD_T'],
                patient_data['Group']
            ]
            res.append(r)

    # Save CSV
    df = pd.DataFrame(res, columns=HEADER)
    output_dir = os.path.join('.', 'training_data')
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, f"Cardiac_parameters_{name}.csv"), index=False)
    print(f"[INFO] Saved CSV for {name} at {output_dir}")

if __name__ == '__main__':
    base_path = r"D:\MSDS\Data Minning\Project\Code\processed_acdc_dataset\dataset"

    train_path = [os.path.join(base_path, 'train_set')]
    validation_path = [os.path.join(base_path, 'validation_set')]
    full_train = [os.path.join(base_path, 'train_set'),
                  os.path.join(base_path, 'validation_set'),
                  os.path.join(base_path, 'test_set')]
    test_path = [os.path.join(base_path, 'test_set')]  # <-- corrected test folder

    # Generate CSVs
    calculate_metrics_for_training(train_path, name='train')
    calculate_metrics_for_training(validation_path, name='validation')
    calculate_metrics_for_training(full_train, name='training')
    calculate_metrics_for_training(test_path, name='test')