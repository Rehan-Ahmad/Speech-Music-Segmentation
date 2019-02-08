# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 11:48:48 2018

@author: Rehan
"""

import time
import scipy.stats.mstats as stats
import numpy as np
from gmm import *
from pyannote.core import Segment, Timeline, Annotation
#from pyannote.metrics.diarization import DiarizationErrorRate
#from pyannote.metrics.diarization import DiarizationPurity
#from pyannote.metrics.detection import DetectionErrorRate
import librosa
import xml.etree.ElementTree as ET
from copy import copy
from sklearn.preprocessing import StandardScaler
import argparse
import os
import matplotlib.pylab as plt
from sklearn.cluster import KMeans

class Diarizer(object):

    def __init__(self, data, total_frames):
        pruned_list = data       
        floatArray = np.array(pruned_list, dtype = np.float32)
        self.X = floatArray.T
        
        self.N = self.X.shape[0]
        self.D = self.X.shape[1]
        self.total_num_frames = total_frames
        
    def write_to_RTTM(self, rttm_file_name, sp_file_name,\
                      meeting_name, most_likely, num_gmms,\
                      seg_length):

        print("...Writing out RTTM file...")
        #do majority voting in chunks of 250
        duration = seg_length
        chunk = 0
        end_chunk = duration

        max_gmm_list = []

        smoothed_most_likely = np.array([], dtype=np.float32)

        while end_chunk < len(most_likely):
            chunk_arr = most_likely[list(range(chunk, end_chunk))]
            max_gmm = stats.mode(chunk_arr)[0][0]
            max_gmm_list.append(max_gmm)
            smoothed_most_likely = np.append(smoothed_most_likely, max_gmm*np.ones(seg_length)) #changed ones from 250 to seg_length
            chunk += duration
            end_chunk += duration

        end_chunk -= duration
        if end_chunk < len(most_likely):
            chunk_arr = most_likely[list(range(end_chunk, len(most_likely)))]
            max_gmm = stats.mode(chunk_arr)[0][0]
            max_gmm_list.append(max_gmm)
            smoothed_most_likely = np.append(smoothed_most_likely,\
                                             max_gmm*np.ones(len(most_likely)-end_chunk))
        most_likely = smoothed_most_likely
        
        out_file = open(rttm_file_name, 'w')

        with_non_speech = -1*np.ones(self.total_num_frames)

        if sp_file_name:
            speech_seg = np.loadtxt(sp_file_name, delimiter=' ',usecols=(0,1))
            speech_seg_i = np.round(speech_seg).astype('int32')
#            speech_seg_i = np.round(speech_seg*100).astype('int32')
            sizes = np.diff(speech_seg_i)
        
            sizes = sizes.reshape(sizes.size)
            offsets = np.cumsum(sizes)
            offsets = np.hstack((0, offsets[0:-1]))

            offsets += np.array(list(range(len(offsets))))
        
        #populate the array with speech clusters
            speech_index = 0
            counter = 0
            for pair in speech_seg_i:
                st = pair[0]
                en = pair[1]
                speech_index = offsets[counter]
                
                counter+=1
                idx = 0
                for x in range(st+1, en+1):
                    with_non_speech[x] = most_likely[speech_index+idx]
                    idx += 1
        else:
            with_non_speech = most_likely
            
        cnum = with_non_speech[0]
        cst  = 0
        cen  = 0
        for i in range(1,self.total_num_frames): 
            if with_non_speech[i] != cnum: 
                if (cnum >= 0):
                    start_secs = ((cst)*0.01)
                    dur_secs = (cen - cst + 2)*0.01
#                    out_file.write("SPEAKER " + meeting_name + " 1 " +\
#                                   str(start_secs) + " "+ str(dur_secs) +\
#                                   " <NA> <NA> " + "speaker_" + str(cnum) + " <NA>\n")
                    out_file.write("SPEAKER " + meeting_name + " 1 " +\
                                   str(start_secs) + " "+ str(dur_secs) +\
                                   " speaker_" + str(cnum) + "\n")
                cst = i
                cen = i
                cnum = with_non_speech[i]
            else:
                cen+=1
                  
        if cst < cen:
            cnum = with_non_speech[self.total_num_frames-1]
            if(cnum >= 0):
                start_secs = ((cst+1)*0.01)
                dur_secs = (cen - cst + 1)*0.01
#                out_file.write("SPEAKER " + meeting_name + " 1 " +\
#                               str(start_secs) + " "+ str(dur_secs) +\
#                               " <NA> <NA> " + "speaker_" + str(cnum) + " <NA>\n")
                out_file.write("SPEAKER " + meeting_name + " 1 " +\
                               str(start_secs) + " "+ str(dur_secs) +\
                               " speaker_" + str(cnum) + "\n")

        print("DONE writing RTTM file")

    def write_to_GMM(self, gmmfile):
        gmm_f = open(gmmfile, 'w')

        gmm_f.write("Number of clusters: " + str(len(self.gmm_list)) + "\n")
             
        #print parameters
        cluster_count = 0
        for gmm in self.gmm_list:

            gmm_f.write("Cluster " + str(cluster_count) + "\n")
            means = gmm.components.means
            covars = gmm.components.covars
            weights = gmm.components.weights

            gmm_f.write("Number of Gaussians: "+ str(gmm.M) + "\n")

            gmm_count = 0
            for g in range(0, gmm.M):
                g_means = means[gmm_count]
                g_covar_full = covars[gmm_count]
                g_covar = np.diag(g_covar_full)
                g_weight = weights[gmm_count]

                gmm_f.write("Gaussian: " + str(gmm_count) + "\n")
                gmm_f.write("Weight: " + str(g_weight) + "\n")
                
                for f in range(0, gmm.D):
                    gmm_f.write("Feature " + str(f) + " Mean " + str(g_means[f]) +\
                                " Var " + str(g_covar[f]) + "\n")
                gmm_count+=1
                
            cluster_count+=1

        print("DONE writing GMM file")
        
    def new_gmm(self, M, cvtype):
        self.M = M
        self.gmm = GMM(self.M, self.D, cvtype=cvtype)

    def new_gmm_list(self, M, K, cvtype):
        self.M = M
        self.init_num_clusters = K
        self.gmm_list = [GMM(self.M, self.D, cvtype=cvtype) for i in range(K)]

    def segment_majority_vote(self, interval_size, em_iters):
        num_clusters = len(self.gmm_list)

        # Resegment data based on likelihood scoring
        likelihoods = self.gmm_list[0].score(self.X)
        for g in self.gmm_list[1:]:
            likelihoods = np.column_stack((likelihoods, g.score(self.X)))
        if num_clusters == 1:
            most_likely = np.zeros(len(self.X))
        else:
            most_likely = likelihoods.argmax(axis=1)

        # Across 2.5 secs of observations, vote on which cluster they should be associated with
        iter_training = {}
        
        for i in range(interval_size, self.N, interval_size):

            arr = np.array(most_likely[(list(range(i-interval_size, i)))])
            max_gmm = int(stats.mode(arr)[0][0])
            iter_training.setdefault((self.gmm_list[max_gmm],max_gmm),[]).append(self.X[i-interval_size:i,:])

        arr = np.array(most_likely[(list(range((int(self.N/interval_size))*interval_size, self.N)))])
        max_gmm = int(stats.mode(arr)[0][0])
        iter_training.setdefault((self.gmm_list[max_gmm], max_gmm),[]).\
                                  append(self.X[int(self.N/interval_size) *interval_size:self.N,:])
        
        iter_bic_dict = {}
        iter_bic_list = []

        # for each gmm, append all the segments and retrain
        for gp, data_list in iter_training.items():
            g = gp[0]
            p = gp[1]
            cluster_data =  data_list[0]

            for d in data_list[1:]:
                cluster_data = np.concatenate((cluster_data, d))

            g.train(cluster_data, max_em_iters=em_iters)

            iter_bic_list.append((g,cluster_data))
            iter_bic_dict[p] = cluster_data

        return iter_bic_dict, iter_bic_list, most_likely

    def cluster(self, em_iters, KL_ntop, NUM_SEG_LOOPS_INIT, NUM_SEG_LOOPS, seg_length):
        print(" ====================== CLUSTERING ====================== ")
        main_start = time.time()

        # ----------- Uniform Initialization -----------
        # Get the events, divide them into an initial k clusters and train each GMM on a cluster
        per_cluster = int(self.N/self.init_num_clusters)
        init_training = list(zip(self.gmm_list,np.vsplit(self.X, list(range(per_cluster, self.N, per_cluster)))))

        for g, x in init_training:
            g.train(x, max_em_iters=em_iters)

        # ----------- First majority vote segmentation loop ---------
        for segment_iter in range(0,NUM_SEG_LOOPS_INIT):
            iter_bic_dict, iter_bic_list, most_likely = self.segment_majority_vote(seg_length, em_iters)

        # ----------- Main Clustering Loop using BIC ------------

        # Perform hierarchical agglomeration based on BIC scores
        best_BIC_score = 1.0
        total_events = 0
        total_loops = 0
        while (best_BIC_score > 0 and len(self.gmm_list) > 1):

            total_loops+=1
            for segment_iter in range(0,NUM_SEG_LOOPS):
                iter_bic_dict, iter_bic_list, most_likely = self.segment_majority_vote(seg_length, em_iters)
                            
            # Score all pairs of GMMs using BIC
            best_merged_gmm = None
            best_BIC_score = 0.0
            merged_tuple = None
            merged_tuple_indices = None

            # ------- KL distance to compute best pairs to merge -------
            if KL_ntop > 0:
                top_K_gmm_pairs = self.gmm_list[0].find_top_KL_pairs(KL_ntop, self.gmm_list)
                for pair in top_K_gmm_pairs:
                    score = 0.0
                    gmm1idx = pair[0]
                    gmm2idx = pair[1]
                    g1 = self.gmm_list[gmm1idx]
                    g2 = self.gmm_list[gmm2idx]

                    if gmm1idx in iter_bic_dict and gmm2idx in iter_bic_dict:
                        d1 = iter_bic_dict[gmm1idx]
                        d2 = iter_bic_dict[gmm2idx]
                        data = np.concatenate((d1,d2))
                    elif gmm1idx in iter_bic_dict:
                        data = iter_bic_dict[gmm1idx]
                    elif gmm2idx in iter_bic_dict:
                        data = iter_bic_dict[gmm2idx]
                    else:
                        continue

                    new_gmm, score = compute_distance_BIC(g1, g2, data, em_iters)
                    
                    #print "Comparing BIC %d with %d: %f" % (gmm1idx, gmm2idx, score)
                    if score > best_BIC_score: 
                        best_merged_gmm = new_gmm
                        merged_tuple = (g1, g2)
                        merged_tuple_indices = (gmm1idx, gmm2idx)
                        best_BIC_score = score

            # ------- All-to-all comparison of gmms to merge -------
            else: 
                l = len(iter_bic_list)

                for gmm1idx in range(l):
                    for gmm2idx in range(gmm1idx+1, l):
                        score = 0.0
                        g1, d1 = iter_bic_list[gmm1idx]
                        g2, d2 = iter_bic_list[gmm2idx] 

                        data = np.concatenate((d1,d2))
                        new_gmm, score = compute_distance_BIC(g1, g2, data, em_iters)

                        #print "Comparing BIC %d with %d: %f" % (gmm1idx, gmm2idx, score)
                        if score > best_BIC_score: 
                            best_merged_gmm = new_gmm
                            merged_tuple = (g1, g2)
                            merged_tuple_indices = (gmm1idx, gmm2idx)
#                            print (best_BIC_score, score)
                            best_BIC_score = score

            # Merge the winning candidate pair if its deriable to do so
            if best_BIC_score > 0.0:
                gmms_with_events = []
                for gp in iter_bic_list:
                    gmms_with_events.append(gp[0])

                #cleanup the gmm_list - remove empty gmms
                for g in self.gmm_list:
                    if g not in gmms_with_events and g != merged_tuple[0] and g!= merged_tuple[1]:
                        #remove
                        self.gmm_list.remove(g)

                self.gmm_list.remove(merged_tuple[0])
                self.gmm_list.remove(merged_tuple[1])
                self.gmm_list.append(best_merged_gmm)
            
            print(" size of each cluster:", [ g.M for g in self.gmm_list])
            
        print("=== Total clustering time: %.2f min" %((time.time()-main_start)/60))
        print("=== Final size of each cluster:", [ g.M for g in self.gmm_list])
        ################### Added later to find likelihood ####################
        lkhoods = self.gmm_list[0].score(self.X)
        for g in self.gmm_list[1:]:
            lkhoods = np.column_stack((lkhoods, g.score(self.X)))
        if len(lkhoods.shape)==2:
            ml = lkhoods.argmax(axis=1)
        else:
            ml = np.zeros(len(self.X))                
        #######################################################################
        return most_likely,ml

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--audio", 
                    default="..\\drive-download-20181029T102550Z-001\\wav_with_speech\\",
                    help="path to input video")
    
    
    args = vars(ap.parse_args())
    # Load the video and store all the video frames.
    opt = ap.parse_args()
    f = '0Ip9orvZy_crop.wav'

    for f in os.listdir(opt.audio):    
        #audiofile = args['audio']
        
        tic = time.time()
        audiofile = f
        verbose = False
        UseMFCC = True
        Usechroma = False
        UseMFCCChroma = False
        FlagFeatureNormalization = True
        spnp = None
    
        winlen = 0.02 #window length 
        hoplen = 0.01 #hop length 
        n_mfcc = 19
        M = 2  # no of GMM components
        K = 6 # no of clusters or GMM
    
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
        
        fVectsSpeech = S
        if FlagFeatureNormalization:
            ss = StandardScaler()
            fVectsSpeech = ss.fit_transform(fVectsSpeech.T).T
            print("Feature Normalization Done...")
    
        ###########################################################################
        diarizer = Diarizer(fVectsSpeech,fVectsSpeech.shape[1])
        # Create the GMM list
        num_comps = M
        num_gmms = K
        diarizer.new_gmm_list(num_comps, num_gmms, 'diag')
    
        # Cluster
        kl_ntop = 0
        num_em_iters = 100
        num_seg_iters_init = 2 #2
        num_seg_iters = 3 #3
        seg_length = 100
        most_likely,_ = diarizer.cluster(num_em_iters, kl_ntop, num_seg_iters_init, num_seg_iters, seg_length)
        # Write out RTTM and GMM parameter files
    #    diarizer.write_to_RTTM(outfile, spnp, meeting_name, most_likely, num_gmms, seg_length)
    #    metric, ref, hyp = DER(outfile, AudioDataSet,annotationlist, audioLength)
        #diarizer.write_to_GMM(gmmfile)
        seglen = 50
        most_likely_chunk = np.split(copy(most_likely), list(range(seglen,most_likely.shape[0],seglen)))
        for i in range(len(most_likely_chunk)):
            mod = int(stats.mode(most_likely_chunk[i])[0][0])
            most_likely_chunk[i][:] = mod 
    
        most_likely_final = most_likely_chunk[0]
        for i in range(1,len(most_likely_chunk)):
            most_likely_final = np.append(most_likely_final,most_likely_chunk[i])
    
        labels = np.repeat(most_likely_final, hoplen*sr)
        segmentpoints = np.where(most_likely_final[:-1] != most_likely_final[1:])[0]
    
        plt.figure()
        plt.plot(y, label='audio signal')
        for s in segmentpoints:
            plt.vlines(s*hoplen*sr, ymin=y.min()*2, ymax=y.max()*2, linestyles='--',label='segment boundry at %.2f sec' %(s*hoplen))
        plt.plot(labels)
        plt.legend()
        plt.title('Audio file: ' + audiofile.split('\\')[-1])
        
        resultpath = os.getcwd() + '\\GMM_MFCCResults\\'
        plt.savefig(resultpath + audiofile.split('\\')[-1][:-4] + '.png')
    
        print('=== Total Time Taken: %.2f min' %((time.time()-tic)/60.0))

