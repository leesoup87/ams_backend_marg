# Copyright 2019 Cochlear.ai Inc. All Rights Reserved.
#

"""This is Cochlear.ai Sense gRPC API main module"""

import grpc
import sys
import time
from concurrent import futures
import os

import numpy as np

from ..pb.SenseClient import SenseClient_pb2
from ..pb.SenseClient import SenseClient_pb2_grpc

from ..mainservice.models.nonstream import nonstream_event, nonstream_music, nonstream_speech
from ..mainservice.models.stream import stream_event, stream_music, stream_speech

from .sub_utils import auth_quota,hexrandom,gs_upload_stream,gs_upload_nonstream,quota_record_nonstream,quota_record_stream

import pymysql
from threading import Thread

host = '35.194.142.205'
user = 'iyjeong'
password = 'Gramgram!'
db = 'cochlear_sense_beta'
charset = 'utf8'


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

class SenseServicer(SenseClient_pb2_grpc.SenseServicer):

	## Non-Streaming 
	def sense(self,request_iterator,context):

		try:
			nonstream_data=[]
			for i in request_iterator:
				apikey=i.apikey
				task=i.task
				ext=i.format
				nonstream_data.append(i.data)

			# Auth part

			task_type = 'async'
			fet = auth_quota(host,user,password,db,charset,task_type,apikey)
			
			if fet == ():
				context.set_code(grpc.StatusCode.UNAVAILABLE)
				context.set_details("Invalid API Key")
				raise APIKeyException('Invalid API Key')

			if fet['call_%s'%(task_type)]>=fet['quota_%s'%(task_type)]:
				context.set_code(grpc.StatusCode.UNAVAILABLE)
				context.set_details("Quota Exceeded")
				raise APIKeyException('Quota Exceeded')

			caller = fet['caller']
			caller_id = fet['id']

			update_info = [caller,caller_id,apikey,task,task_type,ext]

			# Processing
			if task == 'event':
				return self.nonstreaming(nonstream_data,ext,context,nonstream_event.predict,update_info)
			# elif task == 'music':
			# 	return self.nonstreaming(nonstream_data,ext,context,nonstream_music.predict,update_info)
			# elif task == 'speech':
			# 	return self.nonstreaming(nonstream_data,ext,context,nonstream_speech.predict,update_info)

		except APIKeyException as err:
			print('Error:',err)
			pass

		except Exception as err:
			print('Error:',err)
			pass

	def nonstreaming (self,nonstream_data,ext,context,callback,update_info):
		try:
			# Inference Part
			response = callback(nonstream_data,ext,22050)

			postprocessing_thread = Thread(target=self.nonstream_postprocess, args=(nonstream_data, update_info))
			postprocessing_thread.start()

			return SenseClient_pb2.Response(outputs=response)

		except Exception as err:
			print('Error:',err)
			pass
		
	def nonstream_postprocess(self,nonstream_data,update_info):
		filename = hexrandom(20)
		gs_upload_nonstream(nonstream_data,filename,update_info[5])
		os.remove(filename+'.mp3')
		print ('upload process is done')

		quota_record_nonstream(host,user,password,db,charset,update_info,filename)
		print ('quota record is done')
				
		return

	## Streaming

	def sense_stream(self,request_iterator,context):

		auth_check_count = 0
		check_count = 0
		streaming_data=[]

		def on_client_close():

			filename = hexrandom(20)
			# google storage upload

			if gs_upload_stream(streaming_data,filename,update_info[5]):
				os.remove(filename+'.mp3')
				os.remove(filename+'.wav')
				print ('upload process is done')

				quota_record_stream(host,user,password,db,charset,update_info,filename)
				print ('quota record is done')
			else : 
				print ('failure to save')

		context.add_callback(on_client_close)

		try:
			for i in request_iterator:

				check_count+=1
				apikey=i.apikey
				sr=i.sr
				task=i.task
				dtype=i.dtype
				task_type = 'streaming'

				# Auth part
				if auth_check_count==0:
					auth_check_count+=1

					fet = auth_quota(host,user,password,db,charset,task_type,apikey)

					if fet == ():
						context.set_code(grpc.StatusCode.UNAVAILABLE)
						context.set_details("Invalid API Key")
						raise APIKeyException('Invalid API Key')

					if fet['call_%s'%(task_type)]>=fet['quota_%s'%(task_type)]:
						context.set_code(grpc.StatusCode.UNAVAILABLE)
						context.set_details("Quota Exceeded")
						raise APIKeyException('Quota Exceeded')

					caller = fet['caller']
					caller_id = fet['id']

				chunk_size = 2*sr

				##### here to go!
				update_info = [caller,caller_id,apikey,task,task_type,sr,dtype,check_count]
				streaming_data.append(i.data[-chunk_size::])

				if len(streaming_data) > 600:
					context.set_code(grpc.StatusCode.UNAVAILABLE)
					context.set_details("Too long streaming session (Max 5 min)")
					raise streamingException('Too long streaming session (Max 5 min)')

				if len(streaming_data) > 1:
					if task == 'event':
						yield self.streaming(streaming_data,context,stream_event.predict,update_info)
					# elif task == 'music':
					# 	yield self.streaming(streaming_data,context,stream_music.predict,update_info)
					# elif task == 'speech':
					# 	yield self.streaming(streaming_data,context,stream_speech.predict,update_info)

		except APIKeyException as err:
			print('Error:',err)
			pass

		except streamingException as err:
			print('Error:',err)
			pass

		except Exception as err:
			print('Error:',err)
			pass

	def streaming(self,streaming_data,context,callback,update_info):

		try:
			if update_info[3] == 'event':
				response = callback(streaming_data[-2::],update_info[6],update_info[5],update_info[7])
			# elif update_info[3] == 'music':
			# 	response = callback(streaming_data[-6::],update_info[6],update_info[5])
			# elif update_info[3] == 'speech':
			# 	response = callback(streaming_data[-2::],update_info[6],update_info[5])

			return SenseClient_pb2.Response(outputs=response)

		except Exception as err:
			print('Error:',err)
			pass

# gRPC server configuration (Max thread worker setting, define Servicer)

server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))
SenseClient_pb2_grpc.add_SenseServicer_to_server(
	SenseServicer(), server)

if __name__ == "__main__":
	print('Starting server. Listening on port 50051.')
	server.add_insecure_port('0.0.0.0:50051')
	server.start()

try:
	while True:
		time.sleep(86400)

except KeyboardInterrupt:
	server.stop(0)