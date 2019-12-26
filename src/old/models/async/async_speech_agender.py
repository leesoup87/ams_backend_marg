import time
import json 
import io
from sklearn.preprocessing import StandardScaler
import librosa
import numpy as np

import utils


def postprocessing(response,MSO_output):

	result_dict = []

	kid = []
	male = []
	female = []

	for number in range(len(response)):
		kid_temp = response[number][0]
		male_temp = np.max([response[number][1],response[number][3]])
		female_temp = np.max([response[number][2],response[number][4]])

		temp = [kid_temp,male_temp,female_temp]
		temp = temp/np.sum(temp)

		if MSO_output[number][1] > 0.5:
			kid.append(temp[0])
			male.append(temp[1])
			female.append(temp[2])
		else:
			kid.append(0)
			male.append(0)
			female.append(0)

	kid=[round(k,3) for k in kid]
	male=[round(m,3) for m in male]
	female=[round(f,3) for f in female]

	kid_result = {'age/gender':'child','probability':kid}
	male_result = {'age/gender':'male','probability':male}
	female_result = {'age/gender':'female','probability':female}

	result_dict = [kid_result,male_result,female_result]
	result_dict = {'result':result_dict}

	return result_dict


def predict(data,load_model,load_graph,file_format,sess,i,o,fs=16000):

	data = np.asarray(data)
	data = utils.resample(data,file_format,fs)
	data = data.astype(np.float)
	# data = data / np.max(np.abs(data)+np.finfo(float).eps) # waveform normalization

	segments = len(data)/(16000/2)
	data = data[:(16000/2)*segments]

	conc_data = []
	for n_seg in range(segments-1) :
		conc_data.append(data[n_seg*(16000/2):(n_seg+2)*(16000/2)])
		# second-wise normailzation
		conc_data[-1] = conc_data[-1] / ( np.max(np.abs(conc_data[-1])) + np.finfo(float).eps )
	conc_data = np.asarray(conc_data)

	MSO_output = sess.run(o, feed_dict={i:conc_data.ravel()})

	# Inference

	conc_data = np.reshape(conc_data,((segments-1),1,fs))
	model = load_model

	with load_graph.as_default():

		age_output = model.predict(conc_data[:])
		res = postprocessing(age_output,MSO_output)
		res_json = json.dumps(res, ensure_ascii=False)
		return res_json