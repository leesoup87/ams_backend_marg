import base64
import json 
import io
from sklearn.preprocessing import StandardScaler
import librosa
import numpy as np
import csv

arousal=[]
valence=[]
with open('./src/server/models/stream/arousal_valence.csv','rb') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		arousal.append(int(row[0]))
		valence.append(int(row[1]))
arousal=np.array(arousal)
valence=np.array(valence)

def melspec(x, fs=22050, n_fft=1024, hop_size=512, n_mels=128):

	S=librosa.feature.melspectrogram(y=x, sr=fs, n_mels=n_mels,  n_fft=n_fft, hop_length=hop_size)
	log_S = np.log10(S + np.finfo(float).eps)  # log scale
	log_S_T = np.transpose(log_S).astype(np.float32) # return log_S_T

	X_test=log_S_T.reshape((1,log_S_T.shape[0],log_S_T.shape[1],1))

	return X_test


def postprocessing(response,arousal,valence):

	mood_arousal=np.sum(response*arousal)/5
	mood_valence=np.sum(response*valence)/5

	if mood_arousal < 0:
		alpha = -1*mood_arousal/4.0
	else :
		alpha = mood_arousal/2.0

	mood_valence=mood_valence+alpha

	mood_arousal=np.round(mood_arousal,4)
	mood_valence=np.round(mood_valence,4)

	if mood_valence >= 1:
		mood_valence = 0.9999
	elif mood_valence <= -1:
		mood_valence = -0.9999

	mood_prob=[mood_arousal, mood_valence]
	mood_prob=[round(n,3) for n in mood_prob]

	result_dict={}
	result_dict['arousal']=[mood_prob[0]]
	result_dict['valence']=[mood_prob[1]]
	result_dict={'result':[result_dict]}

	return result_dict


def predict(data,load_model,load_graph,fs):

	data = np.asarray(data)
	data = np.fromstring(data, dtype=np.float32)

	data = data / np.max(np.abs(data)+np.finfo(float).eps) # waveform normalization

	if fs != 22050:
		data = librosa.resample(data,fs,22050,res_type='kaiser_best')

	data_melspec = melspec(data, fs=fs, n_fft=1024, hop_size=512, n_mels=128)

	model = load_model
	with load_graph.as_default():
		y_pred = model.predict(data_melspec)

		res = postprocessing(y_pred[0][0],arousal,valence)
		res_json = json.dumps(res, ensure_ascii=False)

	return res_json
