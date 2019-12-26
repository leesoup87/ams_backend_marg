# Copyright 2018 Cochlear.ai Inc.
# All Rights Reserved
# by Jeongsoo Park (jspark@cochlear.ai)

from __future__ import division
import numpy as np
import scipy
import librosa
from scipy import signal
import time
import utils
import json

def localmax(x):
	nr=x.shape[0]

	lx = nr
	#x = np.transpose(x)

	m = (np.concatenate((x[0:1],x[0:lx-1]))<x)&(np.concatenate((x[1:lx],x[lx-1:lx]))<x)
	m = m+0 	# convert bool to int

	return m

def predict(data,file_format,fs=22050):


	data = np.asarray(data)
	data = utils.resample(data,file_format,fs)
	data = data.astype(np.float)
	data = data / np.max(np.abs(data)+np.finfo(float).eps) # waveform normalization

	tmean=80
	tsd=0.7

	# JS Park's method
	# Frame size 441*3 bin (=30ms), Hop size 147bin (=300frames/sec)
	# Spectrogram: 44.1kHz -> cut to 8kHz
	fs_interest=8000	# Region of interest = 0 Hz ~ fs_interest Hz
	M = int(np.round(fs/float(100)*3))
	hs = int(np.round(float(147)/float(44100)*fs))

	# Mel channels
	nmel=40

	# Sample rate for spectrogram frames
	fs_frame = int(float(fs)/float(hs))

	# The minimal bpm among what we will consider 
	# is 15 (4 seconds = 1 cycle)
	acmax = int(round(4*fs_frame))

	# Number of frequency bins used to get mel-spectrogram
	# Rest of the frequency bins are ignored
	nbin = int(np.round(float(M)*float(fs_interest)/float(fs)))

	# Mel spectrogram
	D1 = np.abs(librosa.core.stft(data,n_fft=M,hop_length=hs,win_length=M))**2
	D = np.log(librosa.feature.melspectrogram(S=D1[0:nbin,:]+10E-10,n_mels=nmel))
	D = np.maximum(D, np.max(D)-80)

	# Energy difference 
	d_diff = np.mean(np.maximum(0,np.diff(D)),axis=0)
	difflen = len(d_diff)

	# Low pass filtering --> get onset envelope
	onsetenv = signal.lfilter(b=[1,-1],a=[1,-0.99],x=d_diff)

	# Define frames to use --> deprecated because the audio signal has to be shorter than or equal to 30 sec
	max_duration = 30
	max_time = 30	# sec
	maxcol = int(np.minimum(np.round(max_time*fs_frame),len(onsetenv)))
	mincol = int(np.maximum(1,maxcol-round(max_duration*fs_frame)))

	# Correlate!
	xcorr = np.correlate(onsetenv[mincol:maxcol],onsetenv[mincol:maxcol],"full")

	# Select a part of it
	rawcorr = xcorr[int((len(xcorr)-1)/2)+np.arange(acmax+1)]


	# # Correlate by fft/ifft!
	# xcorr = np.fft.ifft(  np.square(np.abs( np.fft.fft(onsetenv[mincol:maxcol]) ))  )
	# # Select a part of it
	# rawcorr = np.real(xcorr[0:acmax+1])


	# Possible bpms
	#( frames per a min )/( [0~acmax]+0.1 )
	# Excluding the first component, bpms in consideration are
	# 60*fs_frame=1800bpm, 900bpm, 600bpm, ... , 15bpm
	bpms = (60*fs_frame)/(np.arange(acmax+1)+0.1)

	# tmean = mean tempo
	# tsd = standard deviation of the tempos
	# Make a Gaussian kernel to apply it to the correlation vector
	xcorr_win = np.exp(-0.5*(np.square(np.log(bpms/tmean)/np.log(2)/tsd)))

	# Windowed correlation
	xcorr_new = rawcorr*xcorr_win

	# Max peak selection
	xpeaks = localmax(xcorr_new)
	# xpeaks[0:np.min(np.where(xcorr_new<0))]=0
	max_peak = np.max(xcorr_new[xpeaks])

	# Length
	Lcorr = len(xcorr_new)
	xcorr00 = np.concatenate(([0],xcorr_new,[0]),axis=0)	# ?? why?

	# xcorr2 : xxxxxxx
	# xcorr3 : xxxxx
	xcorr2 = xcorr_new[0:int(np.ceil(Lcorr/2))]+ \
		0.25*xcorr00[0:Lcorr:2]+ \
		0.5*xcorr00[1:Lcorr+1:2]+ \
		0.25*xcorr00[2:Lcorr+2:2]

	xcorr3 = xcorr_new[0:int(np.ceil(Lcorr/3))]+ \
		0.33*xcorr00[0:Lcorr:3]+ \
		0.33*xcorr00[1:Lcorr+1:3]+ \
		0.33*xcorr00[2:Lcorr+2:3]

	if np.max(xcorr2) > np.max(xcorr3):
		frame_tempo1 = np.argmax(xcorr2)-1	# -1 for 0 padding in xcorr00
		frame_tempo2 = frame_tempo1*2
	else:
		frame_tempo1 = np.argmax(xcorr3)-1
		frame_tempo2 = frame_tempo1*3

	# Relative probability
	p_ratio1 = xcorr_new[1+frame_tempo1]/( xcorr_new[1+frame_tempo1]+xcorr_new[1+frame_tempo2] )
	p_ratio2 = xcorr_new[1+frame_tempo2]/( xcorr_new[1+frame_tempo1]+xcorr_new[1+frame_tempo2] )
	
	# Define output
	tempos = np.array([60/(frame_tempo1/fs_frame), 60/(frame_tempo2/fs_frame)])

	result_dict = {}
	result_dict['tempo'] = np.array(( round(tempos[0],3), round(tempos[1],3) )).tolist()
	result_dict['probability'] = np.array(( round(p_ratio1,3),round(p_ratio2,3) )).tolist()
	result_dict={'result':[result_dict]}

	res_json = json.dumps(result_dict, ensure_ascii=False)

	return res_json