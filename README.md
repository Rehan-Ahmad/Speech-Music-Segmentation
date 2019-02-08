# Speech-Music-Segmentation
This repository consists of unsupervised segmentation of audio files consist of music and speech using temporal segmentation and diarization process. 

The two algorithms were tested on number of features that include Chroma, Mel-Spectrogram, MFCC+Chroma and MFCC. 

Each directory contain the results of the segmentation process. 

'Diarize_GMM.py' implements the diarization technique based on Gaussian Mixture Model (GMM) while 'Diarize_tempoSeg.py' implements the temporal segmentation technique from librosa package.
