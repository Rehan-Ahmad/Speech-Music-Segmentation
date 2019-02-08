# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 18:45:16 2018

@author: Rehan
"""

import librosa
import numpy as np
import argparse
from librosa.display import specshow
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--audio", 
                default="..\\drive-download-20181029T102550Z-001\\wav_with_speech\\",
                help="path to input video")

args = vars(ap.parse_args())
# Load the video and store all the video frames.
opt = ap.parse_args()

for f in os.listdir(opt.audio):    
    #audiofile = args['audio']
    audiofile = f
    verbose = False
    winlen = 0.02 #window length 
    hoplen = 0.01 #hop length 
    UseMFCC = False
    Usechroma = False
    UseMFCCChroma = False
    FlagFeatureNormalization = False
    n_mfcc = 19

    y, sr = librosa.load(opt.audio + audiofile)   
    
    S = librosa.feature.melspectrogram(y=y, sr=sr , n_fft=int(sr*winlen), hop_length=int(sr*hoplen))
    
    if UseMFCC: 
        S = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc = n_mfcc)
    elif Usechroma:
        S = librosa.feature.chroma_stft(y, sr=sr, hop_length=int(sr*hoplen))

    elif UseMFCCChroma:
        S = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc = n_mfcc)
        chroma = librosa.feature.chroma_stft(y, sr=sr, hop_length=int(sr*hoplen))
        S = np.concatenate((S,chroma), axis=0)       
    else:
        pass
    
    if FlagFeatureNormalization:
        ss = StandardScaler()
        S = ss.fit_transform(S.T).T
        print("Feature Normalization Done...")
    
    if verbose:
        plt.figure(figsize=(10, 4))
        specshow(librosa.power_to_db(S,ref=np.max),y_axis='mel', fmax=8000, x_axis='time',hop_length=int(sr*hoplen))
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel spectrogram of audio: ' + audiofile.split('\\')[-1])
        plt.tight_layout()
    
    # Computing RMS energy.
    rmse = librosa.feature.rmse(y, frame_length=int(sr*winlen), hop_length=int(sr*hoplen))
    if verbose:
        plt.figure()
        plt.plot(rmse.reshape(-1,1))
        plt.title('RMSE')

    # Computing Zero Crossing Rate.    
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=int(sr*winlen), hop_length=int(sr*hoplen))
    if verbose:
        plt.figure()
        plt.plot(zcr.reshape(-1,1))
        plt.title('zero crossing rate')
    
    # Temporal segmentation.
    if UseMFCC:
        bounds = librosa.segment.agglomerative(S, 2)
    elif Usechroma:
        bounds = librosa.segment.agglomerative(S, 2)    
    else:
        bounds = librosa.segment.agglomerative(librosa.power_to_db(S), 2)
        
    bound_times = librosa.frames_to_time(bounds)
    plt.figure()
    plt.plot(y, label='audio signal')
    plt.vlines(bounds[1]*hoplen*sr, ymin=y.min()*2, ymax=y.max()*2, linestyles='--',label='segment boundry at %.2f sec' %(bounds[1]*hoplen))
    plt.legend()
    plt.title('Audio file: ' + audiofile.split('\\')[-1])
    
    if UseMFCC: 
        resultpath = os.getcwd() + '\\MFCCResults\\'
    elif Usechroma:
        resultpath = os.getcwd() + '\\chromaResults\\'
    elif UseMFCCChroma:
        resultpath = os.getcwd() + '\\MFCCChromaResults\\'
    else:
        resultpath = os.getcwd() + '\\'
        
    plt.savefig(resultpath + audiofile.split('\\')[-1][:-4] + '.png')
