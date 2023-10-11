#########################################################################
# If you use the chaos model, please cite the following paper:
# Khante, P., Thomaz, E., & de Barbaro, K. Detecting Auditory Household Chaos 
# in Child-worn Real-world Audio. (Revise & resubmit). Frontiers in Digital Health: 
# Special Issue on Artificial Intelligence for Child Health and Wellbeing (2023).
#########################################################################

# This file loads the saved model and runs it on the segmented daylong audio recordings of participants
# Outputs a file where every 5s segment has a predicted chaos level (0-3)
# Chaos level 0: Silence, Chaos level 1: Low, Chaos level 2: Medium, Chaos level 3: High

##################################################################################
# Some of this code has been inspired/taken from Eduardo Fonseca 2018, v1.0
# https://github.com/edufonseca/icassp19
##################################################################################

import os
import librosa
import sys
import tensorflow as tf
import pandas as pd
import yaml
import pickle
import shutil
import numpy as np
from scipy.stats import gmean
from tqdm import trange

from utils import load_audio_file, modify_file_variable_length, get_mel_spectrogram, save_tensor, save_shape, get_shape
from data import PatchGeneratorPerFile

# Load the pre-trained chaos model 
chaos_model = tf.keras.models.load_model('Final_Chaos_Model.h5')
print("Chaos model loaded successfully")
print(chaos_model.summary())

# Load the scaler
with open('scaler.pkl', 'rb') as fp:
	dgp = pickle.load(fp)

# Set the parameters
n_classes = 4
params_path = {'dataset_path': 'Audio_segments/',
				'dataset_file': 'Audio_segments.csv',
				'features_path': 'features/'}
params_extract = {'audio_len_s': 5, 
				'eps': 2.220446049250313e-16,  
				'fmax': 22050,
				'fmin': 0,
				'fs': 44100, 
				'hop_length_samples': 882,
				'load_mode': 'varup',
				'log': True,
				'n_fft': 2048,
				'n_mels': 96,
				'normalize_audio': True,
				'patch_hop': 50,
				'patch_len': 100,
				'spectrogram_type': 'power',
				'win_length_samples': 1764}
suffix_in = "_mel"

params_extract['audio_len_samples'] = int(params_extract.get('fs') * params_extract.get('audio_len_s'))

# Audio segments file
test_csv = pd.read_csv(params_path.get('dataset_file'))
filelist_audio = test_csv.fname.values.tolist()

# Create features folder. Replaces previously created folder
if os.path.isdir(params_path.get('features_path')):
    shutil.rmtree(params_path.get('features_path'))
os.makedirs(params_path.get('features_path'))

# list with unique n_classes labels
list_labels = sorted(list(set({'0', '1', '2', '3'})))

# create dicts such that key: value is as follows
# label: int
# int: label
label_to_int = {k: v for v, k in enumerate(list_labels)}
int_to_label = {v: k for k, v in label_to_int.items()}

n_extracted_te = 0; n_failed_te = 0

nb_files_te = len(filelist_audio)
for idx, f_name in enumerate(filelist_audio):
	f_path = os.path.join(params_path.get('dataset_path'), f_name)
	print(f_path)
	if os.path.isfile(f_path) and f_name.endswith('.wav'):
		# load audio segment and modify lengths shorter than 5 seconds, if needed
		y = load_audio_file(f_path, input_fixed_length=params_extract['audio_len_samples'], params_extract=params_extract)
		y = modify_file_variable_length(data=y,
										input_fixed_length=params_extract['audio_len_samples'],
										params_extract=params_extract)

		# compute log-scaled mel spec. row x col = time x freq
		# this is done only for the length specified by loading mode (fix, varup, varfull)
		mel_spectrogram = get_mel_spectrogram(audio=y, params_extract=params_extract)

		# save the T_F rep to a binary file (only the considered length)
		save_tensor(var=mel_spectrogram,
					out_path=os.path.join(params_path.get('features_path'),
					f_name.replace('.wav', '.data')), suffix='_mel')

		if os.path.isfile(os.path.join(params_path.get('features_path'),
						f_name.replace('.wav', '_mel.data'))):
			n_extracted_te += 1
			print('%-22s: [%d/%d] of %s' % ('Extracted test features', (idx + 1), nb_files_te, f_path))
		else:
			n_failed_te += 1
			print('%-22s: [%d/%d] of %s' % ('FAILING to extract test features', (idx + 1), nb_files_te, f_path))	
	else:
		print('%-22s: [%d/%d] of %s' % ('this test audio is in the csv but not in the folder', (idx + 1), nb_files_te, f_path))
		
print('n_extracted_te: {0} / {1}'.format(n_extracted_te, nb_files_te))
print('n_failed_te: {0} / {1}\n'.format(n_failed_te, nb_files_te))

print('\nCompute predictions on test set:==================================================\n')

list_preds = []

te_files = [f for f in os.listdir(params_path.get('features_path')) if f.endswith(suffix_in + '.data')]
# to store predictions
te_preds = np.empty((len(te_files), n_classes))

# grab every T_F rep file (computed on the file level)
# split it in T_F patches and store it in tensor, sorted by file
te_gen_patch = PatchGeneratorPerFile(feature_dir=params_path.get('features_path'),
                                     file_list=te_files,
                                     params_extract=params_extract,
                                     suffix_in='_mel',
                                     floatx=np.float32,
                                     scaler=dgp.scaler)   

for i in trange(len(te_files), miniters=int(len(te_files) / 100), ascii=True, desc="Predicting..."):
    # return all patches for a sound file
    patches_file = te_gen_patch.get_patches_file()
    print(patches_file.shape)

    # predicting now on the T_F patch level (not on the wav segment-level)
    preds_patch_list = chaos_model.predict(patches_file).tolist()
    preds_patch = np.array(preds_patch_list)

    # aggregate softmax values across patches in order to produce predictions on the file/clip level
    # take the geometric mean 
    preds_file = gmean(preds_patch, axis=0)
    
    te_preds[i, :] = preds_file

list_labels = np.array(list_labels)
pred_label_files_int = np.argmax(te_preds, axis=1)
pred_labels = [int_to_label[x] for x in pred_label_files_int]

# create dataframe with predictions
# columns: fname & label
# this is based on the features file, instead on the wav file (extraction errors could occur)
te_files_wav = [f.replace(suffix_in + '.data', '.wav') for f in os.listdir(params_path.get('features_path'))
                if f.endswith(suffix_in + '.data')]
pred = pd.DataFrame(te_files_wav, columns=["fname"])
pred['label'] = pred_labels

# Save chaos predictions to a file
pred.to_csv("Predictions.csv", sep=",", index=False)
