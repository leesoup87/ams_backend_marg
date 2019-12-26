# Copyright 2018 Cochlear.ai Inc. All Rights Reserved.


import grpc
import scipy
import hashlib
import random as rand
from google.cloud import storage
import numpy as np
import pymysql
from pydub import AudioSegment
import tempfile
import datetime

# Random Hex Generator

def hexrandom(digits):
	hexkey = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
	hex_str = ""
	for i in range(digits):
		x = rand.randint(0, 15)
		hex_str += hexkey[x]
	return hex_str

	
# Upload file to Google storage
def gs_upload_nonstream(data,filename,file_format):

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

	blob = bucket.blob("phase2/" + filename+'.mp3')
	blob.upload_from_filename(filename+'.mp3')


def gs_upload_stream(data,filename,fs):

	# streaming
	if len(data) > 3:
		streamed = np.asarray(data)
		temp_wav = np.fromstring(streamed,dtype=np.float32)

		scipy.io.wavfile.write(filename+'.wav',fs,temp_wav)
		sound = AudioSegment.from_file(filename+'.wav', format='wav')
		sound.export(filename+'.mp3', format="mp3")

		bucket_name = 'beta_cochlear_sense'
		client = storage.Client.from_service_account_json('Cochlear-ai-56c1c3c33e5b.json')
		bucket = client.get_bucket(bucket_name)

		blob = bucket.blob("phase2/" + filename+'.mp3')
		blob.upload_from_filename(filename+'.mp3')

		# Upload or not
		return True
	else:
		return False


def auth_quota(host,user,password,db,charset,query,apikey):
	conn = pymysql.connect(host=host, user=user, password=password, db=db, charset=charset)
	curs = conn.cursor(pymysql.cursors.DictCursor)

	sql = "select quota_%s, call_%s, id, caller from callers where apikey=%s limit 1"%(query,query,'%s')
	curs.execute(sql,apikey)
	fet = curs.fetchall()
	return fet[0]

def quota_record_nonstream(host,user,password,db,charset,update_info,filename):
	conn = pymysql.connect(host=host, user=user, password=password, db=db, charset=charset)
	curs = conn.cursor(pymysql.cursors.DictCursor)
	version = 2

	t = datetime.datetime.now()
	sql = """INSERT INTO `new_calls` (`caller_id`, `time`, `type`, `task`, `filename`, `version`, `phase`) values (%s, %s, %s, %s, %s, %s, %s) """
	curs.execute(sql,(update_info[1],t,update_info[4],update_info[3],filename+'.mp3', version, "phase2"))
	conn.commit()

	# update quota log
	sql = "Update callers set call_%s = (call_%s + 1) where apikey=%s"""%(update_info[4],update_info[4],'%s')
	curs.execute(sql,(update_info[2]))
	conn.commit()
				
	return 

def quota_record_stream(host,user,password,db,charset,update_info,filename):
	conn = pymysql.connect(host=host, user=user, password=password, db=db, charset=charset)
	curs = conn.cursor(pymysql.cursors.DictCursor)
	version = 2

	t = datetime.datetime.now()
	sql = """INSERT INTO `new_calls` (`caller_id`, `time`, `type`, `task`, `filename`, `version`, `phase`) values (%s, %s, %s, %s, %s, %s, %s) """
	curs.execute(sql,(update_info[1],t,update_info[4],update_info[3],filename+'.mp3', version, "phase2"))
	conn.commit()

	# update quota log
	sql = "Update callers set call_%s = (call_%s + 1) where apikey=%s"""%(update_info[4],update_info[4],'%s')
	curs.execute(sql,(update_info[2]))
	conn.commit()
				
	return 