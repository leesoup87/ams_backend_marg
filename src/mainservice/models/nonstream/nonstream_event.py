import base64
import json 
import librosa
import numpy as np
import csv
import time


from .utils import resample

def preprocessing(x,sr):

	# preprocessing (cut & reshape audio)
	if len(x) < sr:
	    n_frames=1
	else :
	    n_frames=int(len(x)/float(sr)//0.5-1)

	x_divided=np.zeros((n_frames,sr))

	for i in range(n_frames):
	    x_i=x[int(i*sr*0.5):int((i+2)*sr*0.5)]
	    x_divided[i,:len(x_i)]=x_i

	X_input=x_divided.reshape((x_divided.shape[0],1,x_divided.shape[1]))

	return X_input

# need to be done tomorrow

def postprocessing(result):

	event_switch_idx=[20,0,1,2,5,14,22,22,3,4,6,6,7,14,8,9,10,22,13,10,14,15,
	15,14,22,8,16,17,18,19,22,14,21,15,22,22,10,22,11,12,22,22,22]


	NEW_EVENT_CLASSES=[]
	with open('src/mainservice/models/nonstream/new_labels2.txt') as f:
	    reader = csv.reader(f)
	    for row in reader:
	        NEW_EVENT_CLASSES.append(row[0])

	def class_merge(original_output):
	    final_output=np.zeros(len(NEW_EVENT_CLASSES))

	    for i in range(len(original_output)):
	        final_output[event_switch_idx[i]]+=original_output[i]

	    return final_output

	# json wrapping
	frame_result=[]
	summary_result=[]

	prev_class=None
	prev_start_time=None
	prev_end_time=None
	prob_buffer=[]

	for i in range(len(result)):
	    new_y_pred=class_merge(result[i])
	    new_y_pred_argmax=np.argmax(new_y_pred)

	    current_class=NEW_EVENT_CLASSES[new_y_pred_argmax]
	    current_class_prob=round(new_y_pred[new_y_pred_argmax],4)

	    ######################## custom_threshold #############################
	    if current_class in ["Gunshot_explosion", "Laughter", "Liquid_water", "Civil_defense_siren"]:
	        if current_class_prob <0.8:
	            current_class = None
	    #######################################################################

	    if current_class == "Others":
	        current_class = None
	        # current_class_prob = None

	    start_time=i*0.5
	    end_time=i*0.5+1

	    # frame-wise processing
	    frame_temp={"tag":current_class, "probability":current_class_prob, "start_time":start_time, "end_time":end_time}
	    frame_result.append(frame_temp)

	##########################summary part##################################

	    # summary processing
	    if prev_class==None: # prev class is 'others'!!
	        if current_class==None: # others -> others
	            prob_buffer=[]
	        elif current_class!=None: # others -> event
	            prev_start_time=start_time
	            prob_buffer=[current_class_prob]
	    else: # prev class is 'event'!!
	        if current_class!=prev_class:  # save summary if they are different
	            prev_end_time=end_time-0.5
	            mean_class_prob=round(np.mean(prob_buffer),4)
	            summary_temp={"tag":prev_class, "probability":mean_class_prob, "start_time":prev_start_time, "end_time":prev_end_time}
	            summary_result.append(summary_temp)

	            if current_class==None: # event->others
	                prev_start_time=None
	                prev_end_time=None
	                prob_buffer=[]
	            else: # event->new_event
	                prev_start_time=start_time
	                prob_buffer=[current_class_prob]
	        else: # event -> event
	            prob_buffer.append(current_class_prob)
	            prev_end_time=end_time

	    prev_class=current_class

	if prob_buffer: # if last piece exists,
	    prev_end_time=end_time
	    mean_class_prob=round(np.mean(prob_buffer),4)
	    summary_temp={"tag":prev_class, "probability":mean_class_prob, "start_time":prev_start_time, "end_time":prev_end_time}
	    summary_result.append(summary_temp)
	    
	# finalize summary
	tmp_event = {"tag":None, "probability":0.0, "start_time":None, "end_time":None}
	final_summary_result = []
	k = 0
	for i_summary in summary_result:
	    if k==0: # first
	        event_length = (i_summary["end_time"] - i_summary["start_time"])
	        tmp_event["tag"]=i_summary["tag"]
	        tmp_event["probability"]=round((i_summary["end_time"] - i_summary["start_time"]) * i_summary["probability"],4)
	        tmp_event["start_time"]=i_summary["start_time"]
	        tmp_event["end_time"]=i_summary["end_time"]
	    else: #others
	        if (tmp_event["end_time"]==i_summary["start_time"]) and (tmp_event["tag"]==i_summary["tag"]): #continue
	            tmp_event["end_time"] = i_summary["end_time"]
	            tmp_event["probability"] = round(tmp_event["probability"] + (i_summary["end_time"] - i_summary["start_time"]) * i_summary["probability"],4)
	            event_length = event_length + (i_summary["end_time"] - i_summary["start_time"])

	        else: #not continue
	            tmp_event["probability"] = round(tmp_event["probability"] / event_length,4)
	            final_summary_result.append(tmp_event)

	            tmp_event = {"tag":None, "probability":0.0, "start_time":None, "end_time":None}
	            event_length = (i_summary["end_time"] - i_summary["start_time"])
	            tmp_event["tag"]=i_summary["tag"]
	            tmp_event["probability"]=round((i_summary["end_time"] - i_summary["start_time"]) * i_summary["probability"],4)
	            tmp_event["start_time"]=i_summary["start_time"]
	            tmp_event["end_time"]=i_summary["end_time"]
	    k = k+1

	if summary_result != []:
	    tmp_event["probability"] = round(tmp_event["probability"] / event_length,4)    
	    final_summary_result.append(tmp_event)


	final_result = {"status": {
					"code":200,
					"description":"OK"}
					}

	final_result["result"]={"task":"event", "frames":frame_result, "summary":final_summary_result}	
	return final_result


from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
import grpc
import tensorflow as tf


def predict(data,file_format,fs):
	# preprocessing, fs of event model is 22050

	data_infe = np.asarray(data)
	data_infe = resample(data_infe,file_format,fs)
	data_infe = data_infe.astype(np.float)

	X_input = preprocessing(data_infe,fs)

	# Inference call to serving engine
	# To do: multiple models
	channel = grpc.insecure_channel('34.80.243.56:8500')
	stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

	# Inference(threading)

	Ensemble_num = 5
	from threading import Thread

	final_result = {}
	def do_inference(X_input,ver_num):
		request = predict_pb2.PredictRequest()
		request.model_spec.name = 'event'		
		request.model_spec.version.value = ver_num
		request.inputs['input_audio'].CopyFrom(
			tf.contrib.util.make_tensor_proto(X_input.astype(dtype=np.float32),shape=X_input.shape))

		result_future = stub.Predict(request,100)
		result = np.array(result_future.outputs['predictions'].float_val).reshape(len(X_input),43)
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
	ensemble_weights=[0.13714286, 0.19428571, 0.14285714, 0.28, 0.24571429]
	result_1 = final_result[1]*ensemble_weights[0]
	result_2 = final_result[2]*ensemble_weights[1]
	result_3 = final_result[3]*ensemble_weights[2]
	result_4 = final_result[4]*ensemble_weights[3]
	result_5 = final_result[5]*ensemble_weights[4]
	result = result_1+result_2+result_3+result_4+result_5

	# post-processing
	res_json = postprocessing(result)
	res_json = json.dumps(res_json,ensure_ascii=False,sort_keys=False)
	return res_json