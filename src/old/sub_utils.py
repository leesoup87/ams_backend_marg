# Copyright 2018 Cochlear.ai Inc. All Rights Reserved.


import grpc
import scipy
from oauth2client.service_account import ServiceAccountCredentials
import hashlib
import random as rand
from google.cloud import storage
import numpy as np
import pymysql
from pydub import AudioSegment
import tempfile


# Random Hex Generator

def hexrandom(digits):
	hexkey = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
	hex_str = ""
	for i in range(digits):
		x = rand.randint(0, 15)
		hex_str += hexkey[x]
	return hex_str

	
# Upload file to Google storage
def gs_upload_async(data,filename,file_format):

	ext = file_format
	data = np.asarray(data)

	with tempfile.NamedTemporaryFile(suffix='.'+ext) as temp:
		temp.write(data)
		temp.flush()
		sound = AudioSegment.from_file(temp.name, format=ext)
		sound.export(filename+'.mp3', format="mp3")

	bucket_name = 'beta_cochlear_sense'
	client = storage.Client.from_service_account_json('Cochlear-ai-56c1c3c33e5b.json')
	bucket = client.get_bucket(bucket_name)

	blob = bucket.blob(filename+'.mp3')
	blob.upload_from_filename(filename+'.mp3')


def gs_upload_stream(data,filename,fs):

	# streaming
	if len(data) > 3:
		streamed = np.asarray(data)
		temp_wav = np.fromstring(streamed,dtype=np.float32)

		scipy.io.wavfile.write(filename+'.wav',fs,temp_wav)
		sound = AudioSegment.from_file(filename+'.wav', format='wav')
		sound.export(filename+'.mp3', format="mp3")

	# if len(temp_wav.shape)<2 or temp_wav.shape[1]==1:
	# 	AudioSegment(
	# 		temp_wav[:].tobytes(),
	# 		frame_rate = int(fs),
	# 		sample_width = temp_wav.dtype.itemsize,
	# 		channels=1
	# 	).export(filename+'.mp3', format="mp3")
	#elif temp_wav.shape[1]==2:

	# else:
	# 	joined_wav = np.zeros((len(data)*2,1), dtype=temp_wav.dtype)
	# 	joined_wav[::2,0] = temp_wav[:,0]
	# 	joined_wav[1::2,0] = temp_wav[:,1]
	# 	AudioSegment(
	# 		joined_wav.tobytes(),
	# 		frame_rate = int(fs),
	# 		sample_width = joined_channel.dtype.itemsize,
	# 		channels=2
	# 	).export(filename+'.mp3', format="mp3")

		bucket_name = 'beta_cochlear_sense'
		client = storage.Client.from_service_account_json('Cochlear-ai-56c1c3c33e5b.json')
		bucket = client.get_bucket(bucket_name)

		blob = bucket.blob(filename+'.mp3')
		blob.upload_from_filename(filename+'.mp3')

		# Upload or not
		return True
	else:
		return False


