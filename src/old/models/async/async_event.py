import base64
import json 
import librosa
import numpy as np
import utils

def melspec(x, fs, n_fft=1024, hop_size=512, n_mels=128):

	time_range = fs/hop_size # 43 frames ~= 1 sec

	n_frame=len(x)/(fs/2)-1
	X_test=np.zeros((n_frame,time_range,128,1))

	for i in range(n_frame):
		i_onset=i*fs/2
		i_offset=i_onset+fs
		xx=x[i_onset:i_offset]
		xx=xx/np.max(np.abs(xx)+np.finfo(float).eps)

		S=librosa.feature.melspectrogram(y=xx, sr=fs, n_mels=n_mels,  n_fft=n_fft, hop_length=hop_size)
		log_S = np.log10(S + np.finfo(float).eps)  # log scale
		log_S_T = np.transpose(log_S)[:-1]

		X_test[i,:,:,0]=log_S_T

	return X_test

def postprocessing(response, event_name , MSO_output):

	event_list = ['babycry', 'carhorn', 'cough', 'dogbark', 'glassbreak', 'siren', 'snoring']
	event_idx=event_list.index(event_name)

	result=np.array(response)
	result=result[:,event_idx]

	for number in range(len(MSO_output)):
		if MSO_output[number][0] > 0.5:
			result[number] = 0		

	rounded_result=[round(n,3) for n in result]
	rounded_result={'event':event_name,'probability':rounded_result}
	rounded_result={'result':[rounded_result]}	

	return rounded_result

def predict(data,event_name,load_model,load_graph,file_format,sess,i,o,fs=22050):

	# MSO preprocessing

	mso_data = np.asarray(data)
	mso_data = utils.resample(mso_data,file_format,16000)
	mso_data = mso_data.astype(np.float)

	# mso_data = mso_data / np.max(np.abs(mso_data)+np.finfo(float).eps) # waveform normalization	

	segments = len(mso_data)/(16000/2)
	mso_data = mso_data[:(16000/2)*segments]

	conc_data = []
	for n_seg in range(segments-1) :
		conc_data.append(mso_data[n_seg*(16000/2):(n_seg+2)*(16000/2)])
		# second-wise normailzation
		conc_data[-1] = conc_data[-1] / ( np.max(np.abs(conc_data[-1])) + np.finfo(float).eps)
	conc_data = np.asarray(conc_data)

	MSO_output = sess.run(o, feed_dict={i:conc_data.ravel()})

	# Inference

	data_infe = np.asarray(data)
	data_infe = utils.resample(data_infe,file_format,fs)
	data_infe = data_infe.astype(np.float)
	
	mel_spec = melspec(data_infe, fs=fs, n_fft=1024, hop_size=512, n_mels=128)

	model1 = load_model[0]
	model2 = load_model[1]

	with load_graph.as_default():

		if event_name == 'glassbreak':
			y_pred=model1.predict(mel_spec)

		else :
			y_pred1 = model1.predict(mel_spec)
			y_pred2 = model2.predict(mel_spec)
			y_pred=(y_pred1+y_pred2)/2.0
			
		res = postprocessing(y_pred, event_name, MSO_output)
		res_json = json.dumps(res, ensure_ascii=False)
		
	return res_json
