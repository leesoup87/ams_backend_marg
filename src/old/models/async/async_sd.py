import base64
import json 
import librosa
import numpy as np
import tensorflow as tf

import utils
import time

def postprocessing(response):

	result_dict = {}
	temp = []

	for i in range(len(response)):
		temp.append(response[i][1])

	temp_result=[round(n,3) for n in temp]

	result_dict['speech'] = temp_result
	result_dict={'result':[result_dict]}	

	return result_dict

def predict(data,sess,i,o,file_format,fs=16000):
	
	mso_data = np.asarray(data)
	mso_data = utils.resample(mso_data,file_format,fs)
	mso_data = mso_data.astype(np.float)
	# data = data / np.max(np.abs(data)+np.finfo(float).eps) # waveform normalization

	segments = len(mso_data)/(16000/2)
	mso_data = mso_data[:(16000/2)*segments]

	conc_data = []
	for n_seg in range(segments-1) :
		conc_data.append(mso_data[n_seg*(16000/2):(n_seg+2)*(16000/2)])
		# second-wise normailzation
		conc_data[-1] = conc_data[-1] / ( np.max(np.abs(conc_data[-1])) + np.finfo(float).eps )
	conc_data = np.asarray(conc_data)
		
	MSO_output = sess.run(o, feed_dict={i:conc_data.ravel()})

	res = postprocessing(MSO_output)
	res_json = json.dumps(res, ensure_ascii=False)

	return res_json