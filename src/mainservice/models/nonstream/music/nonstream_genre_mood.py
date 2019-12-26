import numpy as np
import librosa
import csv
import json
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
import grpc
import tensorflow as tf

arousal=[]
valence=[]
with open('./src/mainservice/models/nonstream/music/arousal_valence.csv','r') as csvfile:
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


def genre_postprocessing(genre_result1,genre_result2):

	CLASSES = ['Traditional', 'Old-time', 'Pop', 'Rock', 'Electronic',
	'R&B', 'World', 'Latin', 'Metal', 'Alternative',
	'Hip-Hop', 'New-Age', 'Country', 'Jazz', 'Folk', 
	'Classical', 'Punk', 'Reggae', 'Blues', 'Dance']

	SUB_CLASSES = ['Ballad', 'Trot', 'Funk']

	available_idx = [0,1,2,3,4,5,7,8,10,11,12,13,14,15,16,17,19,20,21,22]
	
	old_genre_prob=genre_result1
	genre_prob=np.take(old_genre_prob, available_idx)
	genre_prob=genre_prob/np.sum(genre_prob)

	sorted_idx=np.argsort(genre_prob)[::-1]

	### sub_class check
	lv2_genre_prob=genre_result2
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

	result_acc=[round(n,4) for n in result_acc]

	result_dict={}
	result_dict['genre']=result[0]
	result_dict['prob']=result_acc[0]

	return result_dict

def mood_postprocessing(response,arousal,valence):

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
	mood_prob=[round(n,4) for n in mood_prob]

	result_dict={}
	result_dict['arousal']=mood_prob[0]
	result_dict['valence']=mood_prob[1]

	return result_dict	

def predict(data,fs=22050):

	data = data / np.max(np.abs(data)+np.finfo(float).eps) # waveform normalization
	data_melspec = melspec(data, fs=fs, n_fft=1024, hop_size=512, n_mels=128)

	channel = grpc.insecure_channel('localhost:8500')
	stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
	request = predict_pb2.PredictRequest()
	request.model_spec.name = 'music'

	request.inputs['input_audio'].CopyFrom(
		tf.contrib.util.make_tensor_proto(data_melspec.astype(dtype=np.float32),shape=data_melspec.shape))

	result_future = stub.Predict(request,50)

	mood_result = np.array(result_future.outputs['global_average_pooling2d_1/Mean:0'].float_val)
	genre_result1 = np.array(result_future.outputs['global_average_pooling2d_2/Mean:0'].float_val)
	genre_result2 = np.array(result_future.outputs['global_average_pooling2d_3/Mean:0'].float_val)

	mood_tag = mood_postprocessing(mood_result,arousal,valence)
	genre_tag = genre_postprocessing(genre_result1,genre_result2)
	result_tag = [mood_tag,genre_tag]
	return result_tag
