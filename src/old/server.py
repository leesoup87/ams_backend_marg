# Copyright 2018 Cochlear.ai Inc. All Rights Reserved.
#

"""This is Cochlear.ai Sense gRPC API module"""

import os
import grpc
from concurrent import futures
import time
import tensorflow as tf
import scipy
from oauth2client.service_account import ServiceAccountCredentials
import datetime
from kapre.time_frequency import Melspectrogram
import sys

import hashlib
import random as rand

from ..common import cochlear_sense_pb2
from ..common import cochlear_sense_pb2_grpc

import sub_utils
#import multiprocessing
from threading import Thread


# Streaming functions import
from models.stream import stream_audio_event, stream_speech_agender, stream_music_genre, stream_music_mood, stream_md, stream_sd

# Async functions import
from models.async import async_event, async_md, async_sd, async_speech_agender, async_music_key, async_music_tempo, async_music_genre, async_music_mood

from activity_clf.demodel import get_activity_clf_graph

import pymysql
from keras.models import load_model
import time
import numpy as np
from google.cloud import storage

host = '35.194.142.205'
user = 'iyjeong'
password = 'Gramgram!'
db = 'cochlear_sense_beta'
charset = 'utf8'

## define APIKeyException

class APIKeyException(Exception): 
	'''
		A custom exception class for handling apikey error
		it is used only for the 'Invalid API Key' and 'Quota Exceeded'
	'''
	def __init__(self, message): 
		self.msg = message
	def __str__(self): 
		return self.msg
		
class streamingException(Exception): 
	'''
		A custom exception class for handling streaming
		error for streaming which is longer than 5min.
	'''
	def __init__(self, message): 
		self.msg = message
	def __str__(self): 
		return self.msg
	
## create a class to define the server functions. 

class cochlear_senseServicer(cochlear_sense_pb2_grpc.cochlear_senseServicer):

	##### Async Functions (8 functions, proto file referenced)	
	def event(self,request_iterator,context):
		return self.async(request_iterator, context, async_event.predict,'event')

	def music_detector(self,request_iterator,context):
		return self.async(request_iterator, context, async_md.predict,'music detector')

	def speech_detector(self,request_iterator,context):
		return self.async(request_iterator, context, async_sd.predict,'speech detector')

	def age_gender(self,request_iterator,context):
		return self.async(request_iterator, context, async_speech_agender.predict,'age_gender')

	def music_key(self,request_iterator,context):
		return self.async(request_iterator, context, async_music_key.predict,'music_key')

	def music_tempo(self,request_iterator,context):
		return self.async(request_iterator, context, async_music_tempo.predict,'music_tempo')

	def music_genre(self,request_iterator,context):
		return self.async(request_iterator, context, async_music_genre.predict,'music_genre')		

	def music_mood(self,request_iterator,context):
		return self.async(request_iterator, context, async_music_mood.predict,'music_genre')		


	def async(self,request_iterator,context,callback,function_name):

		# filename generation
		try:
			async_data = []

			# sql connection and session login
			conn = pymysql.connect(host=host, user=user, password=password, db=db, charset=charset)
			curs = conn.cursor(pymysql.cursors.DictCursor)

			# variable defines
			query = 'async'
			task = function_name
			version = 'v1'
			apikey = None
			ext = None
			subtask = 'None'

			# File receiving
			for i in request_iterator:
				apikey = i.apikey
				ext = i.format
				async_data.append(i.data)

			# print 'Request from {}'.format(apikey)

			# Auth and quota check
			sql = "select quota_%s, call_%s, id, caller from callers where apikey=%s limit 1"%(query,query,'%s')
			curs.execute(sql,i.apikey)
			fet = curs.fetchall()

			if fet == ():
				context.set_code(grpc.StatusCode.UNAVAILABLE)
				context.set_details("Invalid API Key")
				raise APIKeyException('Invalid API Key')


			fet = fet[0]

			if fet['call_%s'%(query)]>=fet['quota_%s'%(query)]:
				context.set_code(grpc.StatusCode.UNAVAILABLE)
				context.set_details("Quota Exceeded")
				raise APIKeyException('Quota Exceeded')


			caller = fet['caller']
			caller_id = fet['id']

			# Inference Part
			if function_name == 'age_gender':
				response = callback(async_data,agender_model,graph,i.format,sess,input_session,output_session)

			elif function_name == 'event':
				subtask = i.subtask
				response = callback(async_data,subtask,event_model,graph,i.format,sess,input_session,output_session)

			elif function_name == 'music_genre':
				response = callback(async_data,genre_model,graph,i.format)

			elif function_name == 'music detector':
				response = callback(async_data,sess,input_session,output_session,i.format)

			elif function_name == 'speech detector':
				response = callback(async_data,sess,input_session,output_session,i.format)

			elif function_name == 'music_key':
				response = callback(async_data,i.format)

			elif function_name == 'music_tempo':
				response = callback(async_data,i.format)

			elif function_name == 'music_mood':
				response = callback(async_data,mood_model,graph,i.format)
			
			postprocessing_thread = Thread(target=self.async_postprocess, args=(async_data, apikey, ext, conn, curs, caller_id,query,task, subtask, version))
			postprocessing_thread.start()
			return cochlear_sense_pb2.output(pred=response)

		except APIKeyException, e: 
			print '{} : {}'.format(apikey, str(e))
			pass
		
		except Exception, e:
			print 'Internal error ({}) : {}'.format(type(e), str(e))
			pass


	def async_postprocess(self, async_data, apikey, ext, conn, curs, caller_id, query, task, subtask, version):
		filename = sub_utils.hexrandom(20)
		sub_utils.gs_upload_async(async_data,filename,ext)
		os.remove(filename+'.mp3')
		print ('upload process is done')

		# update call log
		t = datetime.datetime.now()
		sql = """INSERT INTO `new_calls` (`caller_id`, `time`, `type`, `task`,`subtask`, `filename`, `version`) values (%s, %s, %s, %s, %s, %s, %s) """
		curs.execute(sql,(caller_id, t, query, task, subtask, filename+'.mp3', version))
		conn.commit()

		# update quota log
		sql = "Update callers set call_%s = (call_%s + 1) where apikey=%s"""%(query,query,'%s')
		curs.execute(sql,(apikey))
		conn.commit()
				
		return

	##### Streaming Functions (5 functions, proto file referenced )	

	def event_stream(self,request_iterator,context):
		return self.streaming(request_iterator, context, stream_audio_event.predict,'event')

	def age_gender_stream(self,request_iterator,context):
		return self.streaming(request_iterator, context, stream_speech_agender.predict,'age_gender')

	def music_genre_stream(self,request_iterator,context):
		return self.streaming(request_iterator, context, stream_music_genre.predict,'music_genre')

	def music_mood_stream(self,request_iterator,context):
		return self.streaming(request_iterator, context, stream_music_mood.predict,'music_mood')

	def music_detector_stream(self,request_iterator,context):
		return self.streaming(request_iterator, context, stream_md.predict,'music detector')

	def speech_detector_stream(self,request_iterator,context):
		return self.streaming(request_iterator, context, stream_sd.predict,'speech detector')						

	def streaming(self,request_iterator,context,callback,function_name):

		streaming_data = []
		filename = sub_utils.hexrandom(20)

		conn = pymysql.connect(host=host, user=user, password=password, db=db, charset=charset)
		curs = conn.cursor(pymysql.cursors.DictCursor) 

		auth_check_count = 0
		query = 'streaming'
		task = function_name
		version = 'v1'
		subtask = 'None'

		sr = 0
		apikey = ""

		def on_client_close():
			# google storage upload

			if sub_utils.gs_upload_stream(streaming_data,filename,sr) :
				os.remove(filename+'.mp3')
				os.remove(filename+'.wav')

				print ('upload process is done')

				# update call log
				t = datetime.datetime.now()
				sql = """INSERT INTO `new_calls` (`caller_id`, `time`, `type`, `task`,`subtask`, `filename`, `version`) values (%s, %s, %s, %s, %s, %s, %s) """
				curs.execute(sql,(caller_id, t, query, task, subtask, filename+'.mp3', version))
				conn.commit()

				# update quota log
				sql = "Update callers set call_%s = (call_%s + %s) where apikey=%s"""%(query,query,len(streaming_data)/2,'%s')
				curs.execute(sql,(apikey))
				conn.commit()
			else : 
				print 'not save'

		context.add_callback(on_client_close)
		
		try:
			for i in request_iterator:
				sr = i.sr
				apikey = i.apikey

				if auth_check_count==0:

					auth_check_count+=1

					## Authorization and Call log update
					sql = "select quota_%s, call_%s, id, caller from callers where apikey=%s limit 1"%(query,query,'%s')
					curs.execute(sql,apikey)
					fet = curs.fetchall()

					if fet == ():
						context.set_code(grpc.StatusCode.UNAVAILABLE)
						context.set_details("Invalid API Key")
						raise APIKeyException('Invalid API Key')
						# sys.exit(1)

					fet = fet[0]
					if fet['call_%s'%(query)]>=fet['quota_%s'%(query)]:
						context.set_code(grpc.StatusCode.UNAVAILABLE)
						context.set_details("Quota Exceeded")
						raise APIKeyException('Quota Exceeded')
						# sys.exit(1)

					caller = fet['caller']
					caller_id = fet['id']

				chunk_size = 2*sr
				streaming_data.append(i.data[-chunk_size::])
				# streaming_data.append(i.data)

				# task is 'age/gender' : input = 1s, inference unit = 1s, hopsize = 0.5s
				if function_name == 'age_gender':
					if len(streaming_data)<2:
						continue
					response = callback(streaming_data[-2::],agender_model,graph,sess,input_session,output_session)

				elif function_name == 'event':
					subtask = i.subtask
					if len(streaming_data)<2:
						continue
					response = callback(streaming_data[-2::],subtask,event_model,graph,sess,input_session,output_session)

				elif function_name == 'music_genre':
					if len(streaming_data)<6:
						continue
					response = callback(streaming_data[-6::],genre_model,graph,sr)

				elif function_name == 'music detector':
					if len(streaming_data)<2:
						continue
					response = callback(streaming_data[-2::],sess,input_session,output_session)

				elif function_name == 'speech detector':
					if len(streaming_data)<2:
						continue
					response = callback(streaming_data[-2::],sess,input_session,output_session)

				elif function_name == 'music_mood':
					if len(streaming_data)<6:
						continue
					response = callback(streaming_data[-6::],mood_model,graph,sr)

				# Quota Exceeded warning
				if len(streaming_data)/2 > 300:
					context.set_code(grpc.StatusCode.UNAVAILABLE)
					context.set_details("Too long streaming session (Max 5 min)")
					raise streamingException('Too long streaming session (Max 5 min)')

				yield cochlear_sense_pb2.output(pred=response)
		
		except APIKeyException, e: 
			print '{} : {}'.format(apikey, str(e))
			pass
		
		except streamingException, e: 
			print '{} : {}'.format(apikey, str(e))
			pass

		except Exception, e:
			print 'Internal error ({}) : {}'.format(type(e), str(e))
			pass
		

# gRPC server configuration (Max thread worker setting, define Servicer)

server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))

cochlear_sense_pb2_grpc.add_cochlear_senseServicer_to_server(
	cochlear_senseServicer(), server)

# Main Function

if __name__ == "__main__":

	print('[1/7]Starting server. Listening on port 50051.')

	event_model1 = load_model('./src/server/models/h5/event_model1.h5')
	print('[2/7]event model 1 loaded')

	event_model2 = load_model('./src/server/models/h5/event_model2.h5')
	print('[3/7]event model 2 loaded')

	event_model = [event_model1,event_model2]

	genre_model = load_model('./src/server/models/h5/music_genre.h5')
	print('[4/7]music genre model loaded')

	mood_model = load_model('./src/server/models/h5/music_mood.h5')
	print('[5/7]music mood model loaded')

	# MSO model
	g, name_input, name_output = get_activity_clf_graph()
	sess = tf.Session(graph=g)
	input_session = g.get_tensor_by_name(name_input)
	output_session = g.get_tensor_by_name(name_output)
	MSO_input = np.zeros(16000)
	MSO_output = sess.run(output_session, feed_dict={input_session:MSO_input.ravel()})
	print('[6/7]MSO model loaded')

	# age/gender model load
	agender_model = load_model('./src/server/models/h5/age_model.h5', custom_objects={'Melspectrogram':Melspectrogram})
	agender_model.load_weights('./src/server/models/h5/age_weights.h5.5th_00')
	print('[7/7]speech gender model loaded')
	print('ready to go')

	global graph
	graph = tf.get_default_graph()

	server.add_insecure_port('0.0.0.0:50051')
	server.start()

try:
	while True:
		time.sleep(86400)

except KeyboardInterrupt:
	server.stop(0)
