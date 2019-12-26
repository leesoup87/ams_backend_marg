import base64
import json 
import librosa
import numpy as np
import csv
from threading import Thread

from .utils import resample
from .music import stream_genre_mood, stream_key, stream_tempo


def predict(data,dtype,fs):

	# preprocessing, fs of event model is 22050

	data_infe = np.asarray(data)
	data_infe = np.fromstring(data_infe, dtype=dtype)
	data_infe = data_infe.astype(np.float)

	# result threading

	final_result={}
	task_type = ['mg','key','tempo']

	def do_inference(data_infe,task_type):
		if task_type == 'mg':
			final_result[task_type] = stream_genre_mood.predict(data_infe)
		elif task_type == 'key':
			final_result[task_type] = stream_key.predict(data_infe)
		elif task_type == 'tempo':
			final_result[task_type] = stream_tempo.predict(data_infe)	

	# start all threads
	threads = []
	for i in task_type:
		t = Thread(target=do_inference,args=(data_infe,i))
		threads.append(t)

	for x in threads:
		x.start()

	for x in threads:
		x.join()

	final_result = {"result":{
						"task":"music",
						"mood":final_result['mg'][0],
						"genre":final_result['mg'][1],
						"key":final_result['key'],
						"tempo":final_result['tempo']
						}}

	res_json = json.dumps(final_result,ensure_ascii=False,sort_keys=False)
	return res_json	