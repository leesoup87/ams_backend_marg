# Copyright 2018 Cochlear.ai Inc.
# All Rights Reserved
# by Jeongsoo Park (jspark@cochlear.ai)

import numpy as np
import librosa
import scipy
import io
import json

import utils

def predict(data,file_format, fs=44100, profile_type='Krumhansl', distance_type = 'KL', n_key=1):

	data = np.asarray(data)
	data = utils.resample(data,file_format,fs)
	data = data.astype(np.float)
	data = data / np.max(np.abs(data)+np.finfo(float).eps) # waveform normalization

	if fs == 44100:
		M = 8192
		hs = 2048
	elif fs == 22050:
		M = 4096
		hs = 1024
	elif fs == 11025:
		M = 2048
		hs = 512
	else:
		M = np.round( float(8192)*float(fs)/float(44100) )
		hs = np.round( float(2048)*float(fs)/float(44100) )

	# Extracting chroma features
	chroma = librosa.feature.chroma_stft(y=data, sr=fs, n_fft=M, hop_length=hs)

	# Compute chromagram average
	ch_avg = np.mean(chroma,axis=1)

	if profile_type == 'Krumhansl':
		# Krumhansl's key profiles
		Cmaj_key_prof = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
		Cmin_key_prof = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
	elif profile_type == 'Binary':
		Cmaj_key_prof = np.array([1,0,1,0,1,1,0,1,0,1,0,1])
		Cmin_key_prof = np.array([1,0,1,1,0,1,0,1,1,0,0,1])
	elif profile_type == 'Temperley':
		Cmaj_key_prof = np.array([0.184,0.001,0.155,0.003,0.191,0.109,0.005,0.214,0.001,0.078,0.004,0.055])
		Cmin_key_prof = np.array([0.192,0.005,0.149,0.179,0.002,0.144,0.002,0.201,0.038,0.012,0.053,0.022])

	Cmaj_key_prof = Cmaj_key_prof/np.sum(Cmaj_key_prof)
	Cmin_key_prof = Cmin_key_prof/np.sum(Cmin_key_prof)

	maj_prof = np.zeros((12,12))
	min_prof = np.zeros((12,12))

	for i in range(12):
		maj_prof[i,:] = np.roll(Cmaj_key_prof,i)
		min_prof[i,:] = np.roll(Cmin_key_prof,i)

	key_prof = np.concatenate((maj_prof,min_prof),axis=0)

	if distance_type == 'Corr':
		cor = np.corrcoef(ch_avg,key_prof)
		cor = cor[0,1:]
		# key_idx = np.argmax(cor)
		key_idx = cor.argsort()[(-1*n_key):][::-1]

		# Probability
		cor = (cor-np.sum(cor))/np.std(cor)*3

		cor = np.exp(cor)
		cor = cor/np.sum(cor)
		cor.sort()
		key_prob = cor[(-1*n_key):][::-1]
	elif distance_type == 'KL':
		cor = np.zeros(24)
		for j in range(24):
			cor[j] = scipy.stats.entropy(pk=ch_avg,qk=key_prof[j,:])
		# key_idx = np.argmin(cor)
		key_idx = cor.argsort()[:n_key][0::]

		# Probability
		cor = (cor-np.sum(cor))/np.std(cor)*10

		cor = np.exp(cor)
		cor = cor/(np.sum(cor)+np.finfo(float).eps)
		cor.sort()
		key_prob = cor[(-1*n_key):][::-1]


	# key_label = ['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B','c','c#','d','eb','e','f','f#','g','ab','a','bb','b']
	key_label = ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B','Cm','Dbm','Dm','Ebm','Em','Fm','Gbm','Gm','Abm','Am','Bbm','Bm']

	key = [key_label[i] for i in key_idx]
	
	result_dict = {}
	result_dict['key'] = [key[0]]
	result_dict['probability'] = [np.around(key_prob,3)[0]]
	result_dict={'result':[result_dict]}

	res_json = json.dumps(result_dict, ensure_ascii=False,sort_keys=True)

	return res_json

