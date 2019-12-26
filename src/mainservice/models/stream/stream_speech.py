import base64
import json 
import librosa
import numpy as np
import csv
import time


from .utils import resample

def preprocessing(x,sr):

	# cut & reshape audio
	if len(x) < sr:
		n_sec=1
	else:
		n_sec=len(x)//sr

	x_divided=np.zeros((n_sec,sr))

	for k_sec in range(n_sec):
		x_divided[k_sec]=x[k_sec*sr:(k_sec+1)*sr]

	X_input=x_divided.reshape((x_divided.shape[0],1,x_divided.shape[1]))
	return X_input

# need to be done tomorrow

def postprocessing(result):

    event_class_names = ['kids_male','kids_female','teens_male','teens_female','twenties_male','twenties_female','thirties_male',
    						'thirties_female','forties_male','forties_female','fifties_male', 'fifties_female','sixties_male',
    						'sixties_female','old_male','old_female','null']

    frames_result=[]
    for i in range(len(result)):
    	temp_result = result[i]
    	temp_merge_prob = temp_result[16]+temp_result[17]+temp_result[18]
    	new_temp_result = temp_result[0:16]
    	new_temp_result = list(new_temp_result)
    	new_temp_result.append(temp_merge_prob)
    	new_y_pred_argmax=np.argmax(np.asarray(new_temp_result))

    	detected_class = event_class_names[new_y_pred_argmax]
    	detected_class_prob = new_temp_result[new_y_pred_argmax]

    	start_time = i*0.5
    	end_time = i*0.5+1

    	temp = {"index":i,"pred":{"tag":detected_class,"prob":round(detected_class_prob,4),"time_stamp":start_time}}
    	frames_result.append(temp)

    final_result = {"result":{
    					"task":"speech",
    					"frames":frames_result}}

    return final_result

from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
import grpc
import tensorflow as tf


def predict(data,dtype,fs):
	# preprocessing, fs of event model is 22050

	data_infe = np.asarray(data)
	data_infe = np.fromstring(data_infe, dtype=dtype)
	data_infe = data_infe[0:fs]
	data_infe = data_infe.astype(np.float)

	X_input = preprocessing(data_infe,fs)

	# Inference call to serving engine
	# To do: multiple models

	channel = grpc.insecure_channel('34.80.250.148:8500')
	stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
	request = predict_pb2.PredictRequest()
	request.model_spec.name = 'agender'

	# Inference(threading)

	Ensemble_num = 3
	from threading import Thread

	final_result = {}
	def do_inference(X_input,ver_num):	
		request.model_spec.version.value = ver_num
		request.inputs['input_audio'].CopyFrom(
			tf.contrib.util.make_tensor_proto(X_input.astype(dtype=np.float32),shape=X_input.shape))

		result_future = stub.Predict(request,50)
		result = np.array(result_future.outputs['predictions'].float_val).reshape(len(X_input),19)
		final_result[ver_num] = result

	# Start all threads
	threads = []
	for ver_num in range(1,Ensemble_num+1):
		t = Thread(target=do_inference,args=(X_input,ver_num))
		threads.append(t)

	for x in threads:
		x.start()

	# Wait for all of them to finish
	for x in threads:
		x.join()

	# Ensemble weight need to be here

	result = final_result[1]

	# post-processing
	res_json = postprocessing(result)
	res_json = json.dumps(res_json,ensure_ascii=False,sort_keys=False)
	return res_json