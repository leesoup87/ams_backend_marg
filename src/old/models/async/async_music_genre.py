from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
from googleapiclient import errors
import googleapiclient

import base64
import json 
import io
from sklearn.preprocessing import StandardScaler
import librosa
import numpy as np

import utils

def melspec(x, fs=22050, n_fft=1024, hop_size=512, n_mels=128):
	S=librosa.feature.melspectrogram(y=x, sr=fs, n_mels=n_mels,  n_fft=n_fft, hop_length=hop_size)
	log_S = np.log10(S + np.finfo(float).eps)  # log scale
	log_S_T = np.transpose(log_S).astype(np.float32) # return log_S_T

	X_test=log_S_T.reshape((1,log_S_T.shape[0],log_S_T.shape[1],1))

	return X_test

def postprocessing(response):

	CLASSES = ['Traditional', 'Old-time', 'Pop', 'Rock', 'Electronic',
	'R&B', 'World', 'Latin', 'Metal', 'Alternative',
	'Hip-Hop', 'New-Age', 'Country', 'Jazz', 'Folk', 
	'Classical', 'Punk', 'Reggae', 'Blues', 'Dance']

	SUB_CLASSES = ['Ballad', 'Trot', 'Funk']

	available_idx = [0,1,2,3,4,5,7,8,10,11,12,13,14,15,16,17,19,20,21,22]
	
	old_genre_prob=response[1][0]
	genre_prob=np.take(old_genre_prob, available_idx)
	genre_prob=genre_prob/np.sum(genre_prob)

	sorted_idx=np.argsort(genre_prob)[::-1]

	### sub_class check
	lv2_genre_prob=response[2][0]
	lv2_max_idx=np.argmax(lv2_genre_prob)

	### main post_processing
	result = []
	result_acc = []

	if np.max(genre_prob) < 0.1 :
		result.append(CLASSES[np.argmax(genre_prob)])
		result_acc.append(np.max(genre_prob))

	else :
		for i in sorted_idx:
			if genre_prob[i] > 0.1:
				if CLASSES[i] == 'Pop' and lv2_max_idx == 140:
					result.append(SUB_CLASSES[0])
					result_acc.append(genre_prob[i])
				elif CLASSES[i] == 'Traditional' and lv2_max_idx == 108:
					result.append(SUB_CLASSES[1])
					result_acc.append(genre_prob[i])
				elif CLASSES[i] == 'R&B' and lv2_max_idx == 8:
					result.append(SUB_CLASSES[2])
					result_acc.append(genre_prob[i])
				else:
					result.append(CLASSES[i])
					result_acc.append(genre_prob[i])

	result_acc=[round(n,3) for n in result_acc]

	result_dict={}
	result_dict['genre']=result
	result_dict['probability']=result_acc
	result_dict={'result':[result_dict]}

	return result_dict

def predict(data,load_model,load_graph,file_format,fs=22050):
	

	data = np.asarray(data)

	data = utils.resample(data,file_format,fs)
	data = data.astype(np.float)
	data = data / np.max(np.abs(data)+np.finfo(float).eps) # waveform normalization

	data_melspec = melspec(data, fs=fs, n_fft=1024, hop_size=512, n_mels=128)

	model = load_model
	with load_graph.as_default():
		y_pred = model.predict(data_melspec)

	res = postprocessing(y_pred)
	res_json = json.dumps(res, ensure_ascii=False)

	return res_json
