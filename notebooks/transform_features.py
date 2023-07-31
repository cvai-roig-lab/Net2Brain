import glob
from pathlib import Path
from scipy.stats import pearsonr
from scipy.special import betainc
import numpy as np

def pearsonr_many(x, ys):
    x_mean = x.mean()
    y_means = ys.mean(axis=1)

    xm, yms = x - x_mean, ys - y_means[:, np.newaxis]
    r = yms @ xm / np.sqrt(xm @ xm * (yms * yms).sum(axis=1))
    r = r.clip(-1, 1)

    prob = betainc(
        len(x) / 2 - 1,
        0.5,
        1 / (1 + r * r / (1 - r * r))
    )

    return r, prob

def create_feature_rdm(roi_path,save_path,method='pearson',feature_type='fmri'):
    files = glob.glob(roi_path + "/*.npy")
    Path(save_path).mkdir(parents=True, exist_ok=True) # Create save di if it doesn't exist
    for file in files:
        roi_name = Path(file).stem.replace('fmri_','').replace('_fmri','')
        print('Creating RDM for ROI:',roi_name)
        roi_data = np.load(file)
        n_stimuli = roi_data.shape[0]
        ROI_RDM = np.zeros((n_stimuli,n_stimuli))
        for ii in range(n_stimuli):
            if method=='pearson':
                ROI_RDM[ii,ii+1:] = pearsonr_many(roi_data[ii],roi_data[ii+1:])[0]
            else:
                for jj in range(ii+1,n_stimuli):
                    ROI_RDM[ii,jj] = pearsonr(roi_data[ii],roi_data[jj]).statistic
        ROI_RDM = ROI_RDM + ROI_RDM.T + np.eye(n_stimuli)
        final_RDM = np.expand_dims(ROI_RDM, axis=0)
        np.savez(save_path+'/'+feature_type+'_'+roi_name,final_RDM)

