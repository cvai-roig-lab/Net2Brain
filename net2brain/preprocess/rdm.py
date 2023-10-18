import scipy
import numpy as np
import collections
from multiprocessing import Pool
from itertools import combinations
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class eeg_classfier:
    def __init__(self,dataA,dataB):
        self.dataA = dataA
        self.dataB = dataB
        #print(dataA.shape,dataB.shape)
        self.lenTrn = self.dataA.shape[0] - 1
        
    def leaveOneOut(self,ii):
        clf = LDA()
        dat = np.concatenate([np.delete(self.dataA, ii, axis=0), np.delete(self.dataB, ii, axis=0)],axis=0)
        lab = np.concatenate([np.ones(self.lenTrn),np.ones(self.lenTrn)*2])
        #print(dat.shape,lab.shape)
        clf.fit(dat,lab)
        return np.mean([1,2] == clf.predict([self.dataA[ii,:],self.dataB[ii,:]]))

def eeg_rdm(eeg,labels):
    # Calculate minimum number of trial instances
    counts = collections.Counter(labels)
    counts = collections.OrderedDict(sorted(counts.items()))
    ordered_labels = list(counts.keys())
    minCount = min(counts.values())
    ########
    timepoints = eeg.shape[-1]
    numLabels = len(counts)
    index_dict = {l : np.where(labels==l)[0][:minCount] for l in counts.keys()}
    ########
    rdms = np.zeros((numLabels,numLabels,timepoints))
    for tt in range(timepoints):
        for l1,l2 in combinations(range(numLabels),2):
            #print(ii,jj)
            ii = ordered_labels[l1]
            jj = ordered_labels[l2]
            classify = eeg_classfier(np.squeeze(eeg[index_dict[ii],:,tt]),np.squeeze(eeg[index_dict[jj],:,tt]))
            results = []
            with Pool(4) as p:
                results.append(p.map(classify.leaveOneOut, list(range(0,minCount))))
            rdms[l1,l2,tt] = np.mean(results)
            rdms[l2,l1,tt] = np.mean(results)
    return rdms