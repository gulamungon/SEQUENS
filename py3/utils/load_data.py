# Functions for loading and processing features.
#
# 1. load_and_proc_feats               - Load features on which VAD have already been applied. Produces idx. 
# 2. load_and_proc_feats_segm          - As 1. but allows for uniformly sampled lenght between min_len and max_len
# 3. load_and_proc_feats_expand        - Load features without VAD, expand-compress, load and apply VAD.
# 4. load_feats                        - Just load features and VAD. Processing is done later (in Theano)
# 5. load_feats_comb                   - As 4. but concatenates features and VAD.
# 6. load_feats_segm_comb              - As 5. but allows for uniformly sampled segments.
# 7. load_feats_segm_vad_time_comb     - As 6  but segments are sampled based on VAD time.
# 8. load_raw_make_noisy_features_comb - Loads raw files, applies ondra's noise, load and concatenate VAD.
#
# Note:
# Ideally, the number of frames should be the same in the VAD file and in
# the feature file. However this is usually not the case. Some VAD detectors does
# not output the last FALSE region??. Also, "pytel.htk.read_lab_to_bool_vec"
# "By default (length=0), the vector entds with the last true value."
# However, it is desirable to at least make sure that the SPEECH frames are
# more or equal than the VAD frames since if not, it could be that we are
# using features on which VAD has already been applied.
# Therefore we set length=-feat_length which will make sure minimum lenght
# equals -feat_length but no truncation is done. Then we assert that
# VAD length == FEATURE lenght (not if we do concatenation we since mistmatch
# would give error there anwyway).
#


import numpy as np
import pytel.htk, os
from pytel.htk import readhtk, readhtk_segment
import re

from utils.misc import get_logger
log = get_logger()


# Loads Davids features converted to HTK, and the vad converted to text. 
def load_jhu_feat_and_vad(feats_dir, vad_dir, files, floatX='float32'):

    feats = np.vstack( [ readhtk(feats_dir + '/'+ f + '.fea')  for f in files] )
    vad   = [ np.genfromtxt(vad_dir + '/'+ f + '.vad', dtype=int) for f in files ]
    idx   = np.cumsum( np.vstack([ len(vad[i]) for i in range(0, len(vad)) ]  ) )
    idx   = np.insert(idx, 0,0)
   
    return feats, idx, vad


def fix_names_2(nm):
    nm = nm.replace("_segm", "")
    nm = nm.replace("-reverb", "")
    nm = nm.replace("-music", "")
    nm = nm.replace("-noise", "")
    nm = nm.replace("-babble", "")
    return(nm)

p_grep_times = re.compile("-\d+")
def grep_times(fname):
    m = p_grep_times.findall(fname)[-2:]
    f = re.sub(r'-\d+-\d+$', '', fname) # Removes e.g. 8-208 in 00033-reverb-8-208
    return f, -int(m[0]), -int(m[1])


def load_jhu_feat_lab_vad(feats_dir, lab_dir, vad_dir, files, max_frame=-1, feat_is_segm=False, floatX='float32'):
    
    if (max_frame == -1):
        feats = [ readhtk(feats_dir + '/'+ f + '.fea')  for f in files]
    else:
        n_avl_samp =  [ pytel.htk.readhtk_header(feats_dir + '/' + f + '.fea')[0] for f in files ]
        feats = [ readhtk_segment(feats_dir + '/'+ f + '.fea', start=0, end=np.min((max_frame, n_avl_samp[i])))
                  for i,f in enumerate(files)]

    new_files = [ fix_names_2(f) for f in files ]

    lab =[]
    for i,nf in enumerate(new_files):
        if feat_is_segm:
            f, start, end = grep_times(nf)
        else:
            f = nf
            start=0
            end = feats[i].shape[0]
        l = np.genfromtxt(lab_dir + f + ".hp")
        if (vad_dir != None):            
            v = pytel.htk.read_lab_to_bool_vec(vad_dir + '/' + f +'.vad', length=len(l), true_label='speech')
            l = l[v]
        l = l[start:end]    
        if (len(l) != feats[i].shape[0] ):
            log.warning("WARNING: The length of lab is %d but lenght of features is %d. Is VAD missing on lab? " % (len(l), feats[i].shape[0] ) )

        lab.append(l[:,np.newaxis])
        idx   = np.cumsum( np.vstack([ len(feats[i]) for i in range(0, len(feats)) ]  ) )
        idx   = np.insert(idx, 0,0)
    #print feats
    print(lab[2].shape)
    return np.vstack(feats)[np.newaxis,:,:], idx, np.vstack(lab)[np.newaxis,:,:]



def load_jhu_feat(feats_dir, files, floatX='float32', max_frame=-1):

    if (max_frame == -1):
        feats = [ readhtk(feats_dir + '/'+ f + '.fea')  for f in files]
    else:
        n_avl_samp =  [ pytel.htk.readhtk_header(feats_dir + '/' + f + '.fea')[0] for f in files ]
        feats = [ readhtk_segment(feats_dir + '/'+ f + '.fea', start=0, end=np.min((max_frame, n_avl_samp[i])))
                  for i,f in enumerate(files)]
        
    idx   = np.cumsum( np.vstack([ len(feats[i]) for i in range(0, len(feats)) ]  ) )
    idx   = np.insert(idx, 0,0)
    return np.vstack(feats)[np.newaxis,:,:], idx

def load_jhu_feat_segm(feats_dir, files, min_len, max_len, floatX='float32', rng=np.random, start_from_zero=False):
    feats = []
    for f in files:
        n_avl_samp = pytel.htk.readhtk_header(feats_dir + '/' + f + '.fea')[0]
        n_sel_samp = rng.randint(min_len, max_len) #  +1)
        if ( n_avl_samp <= n_sel_samp ):            
            feats.append( readhtk(feats_dir + '/'+ f + '.fea') )
        else:
            if start_from_zero:
                start = 0
            else:
                start = rng.randint(0, n_avl_samp - n_sel_samp +1)
            end   = start + n_sel_samp
            feats.append( readhtk_segment(feats_dir + '/'+ f + '.fea', start, end) )
            
    idx   = np.cumsum( np.vstack([ len(feats[i]) for i in range(0, len(feats)) ]  ) )
    idx   = np.insert(idx, 0,0)
    return np.vstack(feats)[np.newaxis,:,:], idx

def load_jhu_feat_segm_fixed_len(feats_dir, files, min_len, max_len, floatX='float32', start_from_zero=False, suffix='fea', rng=np.random):

    # First we need to loop through the files and check the minimum available lenght
    #n_avl_samp = []
    #for f in files:
    #n_avl_samp.append( pytel.htk.readhtk_header(feats_dir + '/' + f + '.' + suffix)[0] )
    n_avl_samp = np.zeros((len(files), 1), dtype=int)
    for i,f in enumerate(files):
        n_avl_samp[i] = pytel.htk.readhtk_header(feats_dir + '/' + f + '.' + suffix)[0] 

    min_n_avl_samp = np.min( n_avl_samp )
    max_len        = np.min([max_len, min_n_avl_samp + 1] ) # Need to add 1 because max_len because the intervall is [min_len, max_len)
    n_sel_samp     = rng.randint(min_len, max_len)    # not [min_len, max_len] 

    #print min_n_avl_samp
    #print max_len
    #print min_len
    
    #feats = np.zeros((len(files), 512), dtype=floatX)
    for i,f in enumerate(files):
        # The start_from_zero option is mainly for debugging/development 
        if start_from_zero:
            start = 0
        else:
            last_possible_start = n_avl_samp[i] - n_sel_samp
            start = rng.randint(0,  last_possible_start + 1) # This means the intervall is [0,last_possible_start + 1) = [0, last_possible_start]
        end   = start + n_sel_samp
        #feats.append( readhtk_segment(feats_dir + '/'+ f + '.' + suffix, start, end)[np.newaxis,:,:] )
        if i==0 :
            feats_tmp = readhtk_segment(feats_dir + '/'+ f + '.' + suffix, start, end)[np.newaxis,:,:]
            feats = np.zeros((len(files), n_sel_samp, feats_tmp.shape[2]), dtype=floatX)
            feats[i, :, :]= readhtk_segment(feats_dir + '/'+ f + '.' + suffix, start, end)[np.newaxis,:,:] 
        else:
            feats[i, :, :]= readhtk_segment(feats_dir + '/'+ f + '.' + suffix, start, end)[np.newaxis,:,:] 
            
    #return np.vstack(feats)
    return feats


def load_lab_segm_fixed_len(lab_dir, files, min_len, max_len, floatX='float32',
                            start_from_zero=False, suffix='fea', rng=np.random, bn_clean=False):

    #lab_full = [np.genfromtxt(lab_dir + f + ".hp") for f in files]
    lab_full = []
    for f in files:
        if (bn_clean):
            f = f.replace("-reverb", "")
            f = f.replace("-music", "")
            f = f.replace("-noise", "")
            f = f.replace("-babble", "")
        lab_full.append( np.genfromtxt(lab_dir + f + ".hp") )
            
    n_avl_samp = np.zeros((len(files), 1), dtype=int)
    for i in range(len(files)):
        n_avl_samp[i] = lab_full[i].shape
    min_n_avl_samp = np.min( n_avl_samp )
    max_len        = np.min([max_len, min_n_avl_samp + 1] ) # Need to add 1 because max_len because the intervall is [min_len, max_len)
    n_sel_samp     = rng.randint(min_len, max_len)    # not [min_len, max_len] 

    lab = np.zeros( (len(files),n_sel_samp) )
    for i in range(len(files)):
        # The start_from_zero option is mainly for debugging/development 
        if start_from_zero:
            start = 0
        else:
            last_possible_start = n_avl_samp[i] - n_sel_samp
            start = rng.randint(0,  last_possible_start + 1) # This means the intervall is [0,last_possible_start + 1)
                                                             #  = [0, last_possible_start]
        end   = start + n_sel_samp

        lab[i, :] = lab_full[i][start:end] 
            
    return lab


def load_lab_segm_fixed_len(lab_dir,files, min_len, max_len, floatX='float32',
                            start_from_zero=False, suffix='fea', rng=np.random, bn_clean=False):

    #lab_full = [np.genfromtxt(lab_dir + f + ".hp") for f in files]
    lab_full = []
    for f in files:
        if (bn_clean):
            f = f.replace("-reverb", "")
            f = f.replace("-music", "")
            f = f.replace("-noise", "")
            f = f.replace("-babble", "")
        lab_full.append( np.genfromtxt(lab_dir + f + ".hp") )
            
    n_avl_samp = np.zeros((len(files), 1), dtype=int)
    for i in range(len(files)):
        n_avl_samp[i] = lab_full[i].shape
    min_n_avl_samp = np.min( n_avl_samp )
    max_len        = np.min([max_len, min_n_avl_samp + 1] ) # Need to add 1 because max_len because the intervall is [min_len, max_len)
    n_sel_samp     = rng.randint(min_len, max_len)    # not [min_len, max_len] 

    lab = np.zeros( (len(files),n_sel_samp) )
    for i in range(len(files)):
        # The start_from_zero option is mainly for debugging/development 
        if start_from_zero:
            start = 0
        else:
            last_possible_start = n_avl_samp[i] - n_sel_samp
            start = rng.randint(0,  last_possible_start + 1) # This means the intervall is [0,last_possible_start + 1)
                                                             #  = [0, last_possible_start]
        end   = start + n_sel_samp

        lab[i, :] = lab_full[i][start:end] 
            
    return lab




def load_jhu_feat_segm_fixed_len_plus_bottle_n(feats_dir, feats_dir_bn, files, min_len, max_len, floatX='float32',
                                               start_from_zero=False, suffix='fea', rng=np.random, bn_context=30, bn_clean=False):

    # First we need to loop through the files and check the minimum available lenght
    n_avl_samp = np.zeros((len(files), 1), dtype=int)
    for i,f in enumerate(files):
        n_avl_samp[i] = pytel.htk.readhtk_header(feats_dir + '/' + f + '.' + suffix)[0] 

    min_n_avl_samp = np.min( n_avl_samp )
    max_len        = np.min([max_len, min_n_avl_samp + 1] ) # Need to add 1 because max_len because the intervall is [min_len, max_len)
    n_sel_samp     = rng.randint(min_len, max_len)          # not [min_len, max_len] 

    n_failed = 0
    for i,f in enumerate(files):
        bn_f=f
        if (bn_clean):
            bn_f = bn_f.replace("reverb/", "clean/")
            bn_f = bn_f.replace("music/", "clean/")
            bn_f = bn_f.replace("noise/", "clean/")
            bn_f = bn_f.replace("babble/", "clean/")
        try:
            # The start_from_zero option is mainly for debugging/development 
            if start_from_zero:
                start = 0
            else:
                last_possible_start = n_avl_samp[i] - n_sel_samp
                start = rng.randint(0,  last_possible_start + 1) # This means the intervall is [0,last_possible_start + 1) = [0, last_possible_start]
            end   = start + n_sel_samp
            if ((i-n_failed) ==0) :
                feats_tmp = readhtk_segment(feats_dir + '/'+ f + '.' + suffix, start, end)[np.newaxis,:,:]
                feats = np.zeros((len(files), n_sel_samp, feats_tmp.shape[2]), dtype=floatX)
                feats[i-n_failed, :, :]= readhtk_segment(feats_dir + '/'+ f + '.' + suffix, start, end)[np.newaxis,:,:]
                
                bn_feats_tmp = readhtk_segment(feats_dir_bn + '/'+ bn_f + '.' + suffix, start, end-bn_context)[np.newaxis,:,:]
                bn_feats = np.zeros((len(files), n_sel_samp-bn_context, bn_feats_tmp.shape[2]), dtype=floatX)
                bn_feats[i-n_failed, :, :]= readhtk_segment(feats_dir_bn + '/'+ bn_f + '.' + suffix, start, end-bn_context)[np.newaxis,:,:]
            else:
                feats[i-n_failed, :, :]= readhtk_segment(feats_dir + '/'+ bn_f + '.' + suffix, start, end)[np.newaxis,:,:] 
                bn_feats[i-n_failed, :, :]= readhtk_segment(feats_dir_bn + '/'+ bn_f + '.' + suffix, start, end-bn_context)[np.newaxis,:,:]
        except:
            log.warning("Failed to load file %s, %s", f, bn_f)
            n_failed +=1
    if  n_failed > 0 :           
        return [feats[0:-n_failed], bn_feats[0:-n_failed]]
    else:
        return [feats, bn_feats]

##
def load_jhu_feat_segm_fixed_len_plus_bottle_n_2(feats_dir, feats_dir_bn, files, min_len, max_len, min_len_2, max_len_2, floatX='float32',
                                                 start_from_zero=False, suffix='fea', rng=np.random, bn_context=30, bn_clean=False):

    # First we need to loop through the files and check the minimum available lenght
    n_avl_samp = np.zeros((len(files), 1), dtype=int)
    for i,f in enumerate(files):
        n_avl_samp[i] = pytel.htk.readhtk_header(feats_dir + '/' + f + '.' + suffix)[0] 

    min_n_avl_samp = np.min( n_avl_samp )
    max_len        = np.min([max_len, min_n_avl_samp + 1] ) # Need to add 1 because max_len because the intervall is [min_len, max_len)
    n_sel_samp     = rng.randint(min_len, max_len)          # not [min_len, max_len] 
    n_sel_samp_2   = rng.randint(min_len, max_len)          # not [min_len, max_len] 

    
    n_failed = 0
    for i,f in enumerate(files):
        bn_f=f
        if (bn_clean):
            bn_f = bn_f.replace("reverb/", "clean/")
            bn_f = bn_f.replace("music/", "clean/")
            bn_f = bn_f.replace("noise/", "clean/")
            bn_f = bn_f.replace("babble/", "clean/")
        try:
            # The start_from_zero option is mainly for debugging/development 
            if start_from_zero:
                start   = 0
                start_2 = 0
            else:
                last_possible_start = n_avl_samp[i] - n_sel_samp
                start = rng.randint(0,  last_possible_start + 1)     # This means the intervall is [0,last_possible_start + 1)
                                                                     #   = [0, last_possible_start]
                last_possible_start_2 = n_avl_samp[i] - n_sel_samp_2
                start_2 = rng.randint(0,  last_possible_start_2 + 1) # -"-
            end   = start + n_sel_samp
            end_2   = start_2 + n_sel_samp_2
            if ((i-n_failed) ==0) :
                feats_tmp = readhtk_segment(feats_dir + '/'+ f + '.' + suffix, start, end)[np.newaxis,:,:]
                feats = np.zeros((len(files), n_sel_samp, feats_tmp.shape[2]), dtype=floatX)
                feats[i-n_failed, :, :]= readhtk_segment(feats_dir + '/'+ f + '.' + suffix, start, end)[np.newaxis,:,:] # Why not setting it to feat_tmp?
                
                bn_feats_tmp = readhtk_segment(feats_dir_bn + '/'+ bn_f + '.' + suffix, start_2, end_2 - bn_context)[np.newaxis,:,:]
                bn_feats = np.zeros((len(files), n_sel_samp_2-bn_context, bn_feats_tmp.shape[2]), dtype=floatX)
                bn_feats[i-n_failed, :, :]= readhtk_segment(feats_dir_bn + '/'+ bn_f + '.' + suffix, start_2, end_2-bn_context)[np.newaxis,:,:]
            else:
                feats[i-n_failed, :, :]= readhtk_segment(feats_dir + '/'+ bn_f + '.' + suffix, start, end)[np.newaxis,:,:] 
                bn_feats[i-n_failed, :, :]= readhtk_segment(feats_dir_bn + '/'+ bn_f + '.' + suffix, start_2, end_2-bn_context)[np.newaxis,:,:]
        except:
            log.warning("Failed to load file %s, %s", f, bn_f)
            n_failed +=1
    if  n_failed > 0 :           
        return [feats[0:-n_failed], bn_feats[0:-n_failed]]
    else:
        return [feats, bn_feats]

# Loads
# 1 feat from "segm A"
# 2 feat from "segm B"
# 3 lab from "segm B"
def fix_names(nm):
    nm = nm.replace("-reverb", "")
    nm = nm.replace("-music", "")
    nm = nm.replace("-noise", "")
    nm = nm.replace("-babble", "")
    return(nm)

def load_jhu_feat_segm_fixed_len_plus_lab(feats_dir, lab_dir, files, min_len, max_len, min_len_B, max_len_B, floatX='float32',
                                          start_from_zero=False, suffix='fea', rng=np.random, segm_B_clean=1, segm_A=True,
                                          vad_dir=None, ab_same=False):


    # First we need to loop through the files and check the minimum available lenght
    n_avl_samp = np.zeros((len(files), 1), dtype=int)
    for i,f in enumerate(files):
        n_avl_samp[i] = pytel.htk.readhtk_header(feats_dir + '/' + f + '.' + suffix)[0] 

    min_n_avl_samp = np.min( n_avl_samp )
    max_len        = np.min([max_len, min_n_avl_samp + 1] ) # Need to add 1 because max_len because the intervall is [min_len, max_len)
    if segm_A:
        n_sel_samp     = rng.randint(min_len, max_len)      # not [min_len, max_len]
    if ab_same:
        n_sel_samp_B = n_sel_samp
    else:
        n_sel_samp_B   = rng.randint(min_len, max_len)          # not [min_len, max_len]

    if ( segm_B_clean >=1 ):
        lab = [ np.genfromtxt(lab_dir + fix_names(f) + ".hp") for f in files  ]
    else:
        lab = [ np.genfromtxt(lab_dir + f + ".hp") for f in files  ]
        
    if (vad_dir != None):    
        lab = [lab[i][pytel.htk.read_lab_to_bool_vec(vad_dir + '/' + fix_names(f) +'.vad', true_label='speech',
                                                    length=len(lab[i]))] for i,f in enumerate(files)]
            
    # Sanity check
    if (len(lab[0]) != n_avl_samp[0] ):
        log.warning("WARNING: The length of lab is %d but lenght of features is %d. Is VAD missing on lab? " % (len(lab[0]), n_avl_samp[0] ) )
    
    n_failed = 0
    for i,f in enumerate(files):
        segm_b_f=f
        if ( segm_B_clean >=2 ):
            segm_b_f = fix_names(segm_b_f)
            
        try:
            # The start_from_zero option is mainly for debugging/development 
            if start_from_zero:
                if segm_A:
                    start   = 0
                start_B = 0
            else:
                if segm_A:
                    last_possible_start = n_avl_samp[i] - n_sel_samp
                    start = rng.randint(0,  last_possible_start + 1) # This means the intervall is [0,last_possible_start + 1)
                                                                 #    = [0, last_possible_start]
                if ab_same:
                    start_B = start
                else:
                    last_possible_start_B = n_avl_samp[i] - n_sel_samp_B
                    start_B = rng.randint(0,  last_possible_start_B + 1) # - " -
            if segm_A:
                end   = start + n_sel_samp
            if ab_same:
                end_B = end
            else:                
                end_B   = start_B + n_sel_samp_B
            if ((i-n_failed) ==0) :
                if segm_A:
                    feats_tmp = readhtk_segment(feats_dir + '/'+ f + '.' + suffix, start, end)[np.newaxis,:,:]
                    feats = np.zeros((len(files), n_sel_samp, feats_tmp.shape[2]), dtype=floatX)
                    feats[i-n_failed, :, :] = readhtk_segment(feats_dir + '/'+ f + '.' + suffix, start, end)[np.newaxis,:,:]

                feats_tmp_B = readhtk_segment(feats_dir + '/'+ segm_b_f + '.' + suffix, start_B, end_B)[np.newaxis,:,:]
                feats_B = np.zeros((len(files), n_sel_samp_B, feats_tmp_B.shape[2]), dtype=floatX)
                feats_B[i-n_failed, :, :] = readhtk_segment(feats_dir + '/'+ segm_b_f + '.' + suffix, start_B, end_B)[np.newaxis,:,:]
                
                lab[i-n_failed]=lab[i][start_B:end_B]
            else:
               if segm_A:
                   feats[i-n_failed, :, :]= readhtk_segment(feats_dir + '/'+ f + '.' + suffix, start, end)[np.newaxis,:,:]
               feats_B[i-n_failed, :, :]= readhtk_segment(feats_dir + '/'+ segm_b_f + '.' + suffix, start_B, end_B)[np.newaxis,:,:]
               lab[i-n_failed]=lab[i][start_B:end_B]
        except Exception as e:
            log.warning("Failed to load file %s, %s", f)
            log.warning(e.__doc__)
            log.warning(e.message)
            n_failed +=1

    lab=np.vstack(lab[0:i-n_failed+1])
    if  n_failed > 0 :
        if segm_A:
            return [feats[0:-n_failed], feats_B[0:-n_failed], lab[0:-n_failed]]
        else:
            return [feats_B[0:-n_failed], lab[0:-n_failed]]
    else:
        if segm_A:
            return [feats, feats_B, lab]
        else:
            return [feats_B, lab]






    
# These functions will load and process feature and return the
# the input needed for the network.
#
# This one uses features on which VAD have already been applied
def load_and_proc_feats(feats_dir, files, frame_step=1, floatX='float32', max_length=None):        
    feats = []
    idx   = [0]
    for f in files:
        f_path = feats_dir + '/'+ f + '.fea'
        if  ( os.path.isfile( f_path ) ):
            tmp_feats = readhtk(feats_dir + '/'+ f + '.fea')[0:max_length:frame_step,:]
            idx.append(tmp_feats.shape[0])
            feats.append(tmp_feats)
        else:
            idx.append(0)

    idx = np.cumsum( np.vstack( idx ) )
    feats = np.concatenate(feats)

    return [ feats.astype(floatX), idx.astype(floatX) ] 


# As above but a smaller segment.
def load_and_proc_feats_segm(feats_dir, files, frame_step, min_len, max_len, floatX='float32'):
    feats_o = []
    for f in files:
        n_avl_samp = pytel.htk.readhtk_header(feats_dir + '/' + f + '.fea')[0]
        n_sel_samp = np.random.randint(min_len, max_len)
        if ( n_avl_samp <= n_sel_samp ):
            feats_o.append( readhtk(feats_dir + '/'+ f + '.fea') )
        else:
            # Randomly select the starting point
            start = np.random.randint(0, n_avl_samp - n_sel_samp)
            end   = start + n_sel_samp
            feats_o.append( readhtk_segment(feats_dir + '/'+ f + '.fea', start, end) )

    idx = np.cumsum( np.vstack([ len(feats_o[i]) for i in range(0, len(feats_o)) ]  ) )
    idx = np.insert(idx, 0,0)

    # Need these as arrays
    feats_o = np.vstack( feats_o[i] for i in range(0, len(feats_o)))

    return [ feats_o, idx.astype(floatX) ] 





# This one does the processing used in our first end-to-en system
# This inclued expanded features. And after that, apply VAD.
def load_and_proc_feats_expand(vad_dir, feats_dir, files, frame_step=1, floatX='float32', max_length=None):        
    feats_o = [ readhtk(feats_dir + '/'+ f + '.fea')[::frame_step,:]  for f in files]
    feats   = [ nn_def_anya.preprocess_nn_input_MFCC( readhtk(feats_dir + '/'+ f + '.fea')[0:max_length:frame_step,:] ) for f in files]

    vad = [pytel.htk.read_lab_to_bool_vec(vad_dir + '/' + files[i]+'.lab.gz', true_label='sp',
                                          length=-len(feats_o[i]) )[15:-15] for i in range(0, len(feats))  ]
    #vad = [pytel.htk.read_lab_to_bool_vec(vad_dir + '/' + files[i]+'.lab.gz', true_label='sp' )[15:-15]  
    #       for i in range(0, len(feats))  ]               

    for i in range(0, len(feats)):
        assert(len(vad) == feats[i].shape[0])
    
    # Now apply the VAD
    feats_o = np.vstack( feats_o[i][15:-15][vad[i]] for i in range(0, len(feats)))
    feats   = np.vstack( feats[i][vad[i]] for i in range(0, len(feats)))

    idx = np.cumsum( np.vstack([ np.sum(vad[i]) for i in range(0, len(vad)) ]  ) )
    idx = np.insert(idx, 0,0)

    return [ feats, feats_o, idx.astype(floatX) ] 

# This one only load, feats and vad without processing them.
# Processing is done later by Theano code.
def load_feats(vad_dir, feats_dir, files, frame_step=1, floatX='float32'):        
    feats_o = [ readhtk(feats_dir + '/'+ f + '.fea')[::frame_step,:]  for f in files]
    #feats   = [ nn_def_anya.preprocess_nn_input_MFCC( readhtk(feats_dir + '/'+ f + '.fea')[::frame_step,:] ) for f in files]

    vad = [pytel.htk.read_lab_to_bool_vec(vad_dir + '/' + files[i]+'.lab.gz', true_label='sp', length=-len(feats_o[i]) )  
           for i in range(0, len(feats_o))  ]               

    for i in range(0, len(feats_o)):
        assert(len(vad) == feats_o[i].shape[0])

    idx = np.cumsum( np.vstack([ len(feats_o[i]) for i in range(0, len(vad)) ]  ) )
    idx = np.insert(idx, 0,0)

    # Need these as arrays
    feats_o = np.vstack( feats_o[i] for i in range(0, len(feats_o)))
    vad     = np.hstack( vad[i] for i in range(0, len(vad)))

    return [ feats_o, vad, idx.astype(floatX) ] 



# This one only load, feats and vad without processing them.
# Processing is done later by Theano code.
def load_feats_comb(vad_dir, feats_dir, files, frame_step=1, floatX='float32'):        
    feats_o = [ readhtk(feats_dir + '/'+ f + '.fea')[::frame_step,:]  for f in files]

    vad = [pytel.htk.read_lab_to_bool_vec(vad_dir + '/' + files[i]+'.lab.gz', true_label='sp', length=-len(feats_o[i]) )   
           for i in range(0, len(feats_o))  ]               
        
    idx = np.cumsum( np.vstack([ len(feats_o[i]) for i in range(0, len(vad)) ]  ) )
    idx = np.insert(idx, 0,0)

    # Need these as arrays
    feats_o = np.vstack( feats_o[i] for i in range(0, len(feats_o)))
    vad     = np.hstack( vad[i] for i in range(0, len(vad)))

    #return [ np.hstack( ( feats_o, vad[:,np.newaxis].astype(T.config.floatX) ) ), idx.astype('float32') 
    return [ np.hstack( ( feats_o, vad[:,np.newaxis].astype(floatX) ) ), idx.astype(floatX) ]

# Need to fix this one for dealing with situation where
# vad info is shorter than the number of frames in the feature file.
def load_feats_segm_comb(vad_dir, feats_dir, files, frame_step, min_len, max_len, floatX='float32'):
    feats_o = []
    vad     = []
    for f in files:
        n_avl_samp = pytel.htk.readhtk_header(feats_dir + '/' + f + '.fea')[0]
        n_sel_samp = np.random.randint(min_len, max_len)
        if ( n_avl_samp <= n_sel_samp ):
            feats_o.append( readhtk(feats_dir + '/'+ f + '.fea') )
            #vad.append(  pytel.htk.read_lab_to_bool_vec(vad_dir + '/' + f +'.lab.gz', true_label='sp' ) )
            vad.append(  pytel.htk.read_lab_to_bool_vec(vad_dir + '/' + f +'.lab.gz', true_label='sp',
                                                        length=-len(feats_o[i]) ) )   
        else:
            # Randomly select the starting point
            start = np.random.randint(0, n_avl_samp - n_sel_samp)
            end   = start + n_sel_samp
            feats_o.append( readhtk_segment(feats_dir + '/'+ f + '.fea', start, end) )
            #vad.append(  pytel.htk.read_lab_to_bool_vec(vad_dir + '/' + f +'.lab.gz', true_label='sp' )[start:end] )
            vad_tmp = pytel.htk.read_lab_to_bool_vec(vad_dir + '/' + f +'.lab.gz', true_label='sp', length = -n_avl_samp )
            assert( len(vad_tmp) ==  n_avl_samp )                
            vad.append( vad_tmp[start:end] )
            
    idx = np.cumsum( np.vstack([ len(feats_o[i]) for i in range(0, len(vad)) ]  ) )
    idx = np.insert(idx, 0,0)

    # Need these as arrays
    feats_o = np.vstack( feats_o[i] for i in range(0, len(feats_o)))
    vad     = np.hstack( vad[i] for i in range(0, len(vad)))

    return [ np.hstack( ( feats_o, vad[:,np.newaxis].astype(floatX) ) ), idx.astype(floatX) ] 

# As the above but checks time according to VAD.
def load_feats_segm_vad_time_comb(vad_dir, feats_dir, files, frame_step, min_len, max_len, floatX='float32'):
    feats_o = []
    vad     = []
    for f in files:
        n_avl_samp_tot = pytel.htk.readhtk_header(feats_dir + '/' + f + '.fea')[0]
        vad_tmp        = pytel.htk.read_lab_to_bool_vec(vad_dir + '/' + f +'.lab.gz', true_label='sp',
                                                        length =- n_avl_samp_tot )
        assert(len(vad_tmp) == n_avl_samp_tot)
        n_avl_samp     = np.sum(vad_tmp) # The speech time
        n_sel_samp     = np.random.randint(min_len, max_len)
        if ( n_avl_samp <= n_sel_samp ):
            feats_o.append( readhtk(feats_dir + '/'+ f + '.fea') )
            vad.append( vad_tmp )   
        else:
            # Randomly select the starting point
            start = np.random.randint(0, n_avl_samp - n_sel_samp)
            end   = start + n_sel_samp

            # Need to find the start and end index 
            idx = np.cumsum( vad_tmp )
            start= int(np.where(np.cumsum(vad_tmp) == start)[0])
            end  = int(np.where(np.cumsum(vad_tmp) == end)[0])

            feats_o.append( readhtk_segment(feats_dir + '/'+ f + '.fea', start, end) )
            vad.append( vad_tmp[start:end] )   

    idx = np.cumsum( np.vstack([ len(feats_o[i]) for i in range(0, len(vad)) ]  ) )
    idx = np.insert(idx, 0,0)

    # Need these as arrays
    feats_o = np.vstack( feats_o[i] for i in range(0, len(feats_o)))
    vad     = np.hstack( vad[i] for i in range(0, len(vad)))

    return [ np.hstack( ( feats_o, vad[:,np.newaxis].astype(floatX) ) ), idx.astype(floatX) ] 



def load_raw_make_noisy_features_comb(vad_dir, raw_dir, files, n=None, floatX='float32'):
    window = np.hamming(200)
    noverlap=120
    deltawindow = accwindow = 2
    targetkind=pytel.htk.MFCC|pytel.htk._D|pytel.htk._A|pytel.htk._0|pytel.htk._C
    cmvn_lc = cmvn_rc =150
    fbank_mx= pytel.features.mel_fbank_mx(window.size, fs=8000, NUMCHANS=24, LOFREQ=120.0, HIFREQ=3800.0)
    left_ctx = right_ctx = 15
    
    # Basically from Anya's mk_tars_raw_MFCC19C0DA120_3800.py
    features =[]
    for fn in files:

        fea = np.fromfile(raw_dir+'/'+fn+'.raw',
                          dtype='int16').astype(floatX) # I think converting to float here should not be necessary.
        if (n == None):
            fea = pytel.features.add_dither(fea, 1.0)       # No point to add dither if we add other noise I think   
        else:
            fea = n.add_corruption_to_batch([fea])[0]
        """
        fea0 = np.fromfile(raw_dir+'/'+fn+'.raw',
                          dtype='int16').astype(floatX) # I think converting to float here should not be necessary.
        vad0 = pytel.htk.read_lab_to_bool_vec(vad_dir + '/' + fn +'.lab.gz', true_label='sp', length=len(fea0) ) 
        fea, vad = n.add_corruption_to_batch([fea0, vad0])
        print len(fea)
        print len(fea0)
        print sum(vad)
        print sum(vad0)
        assert(len(fea)== len(fea0))
        assert(np.all(vad==vad0))
        """
        fea = pytel.features.mfcc_htk(fea, window, noverlap, fbank_mx, NUMCEPS=19, USEPOWER=True, ZMEANSOURCE=True)
        fea = pytel.features.add_deriv(fea,(deltawindow,accwindow))
        if len(fea) < 3: raise Exception("Too few frames left: " + str(len(fea)))
        fea = pytel.features.cmvn_floating(fea, cmvn_lc, cmvn_rc, unbiased=True)
        # This is to make sure we have enough features for extracting at least                                               
        # one frame in the NN. Think this was only used in the mk_tars script??                                              
        #if end+right_ctx+1>len(fea):
        #    fea  = np.r_[fea, np.repeat(fea[[-1]], end+right_ctx+1-len(fea), axis=0)]
        #if start-left_ctx<0:
        #    fea  = np.r_[np.repeat(fea[[0]], left_ctx-start, axis=0), fea]
        #    end  = end + left_ctx - start
        #    start=left_ctx
        #fea    = fea[start-left_ctx:end+right_ctx+1]                                                                        
        vad = pytel.htk.read_lab_to_bool_vec(vad_dir + '/' + fn +'.lab.gz', true_label='sp', length=len(fea) ) 
        fea = np.hstack((fea,vad[:,np.newaxis]))
        features.append(fea)

    idx = np.cumsum( np.vstack([ len(features[i]) for i in range(0, len(features)) ]  ) )
    idx = np.insert(idx, 0,0)
        
    return [np.vstack(features).astype(floatX), idx.astype(floatX) ] 


"""
# Loads
# 1 feat from "segm A"
# 2 feat from "segm B"
# 3 lab from "segm B"
def load_jhu_feat_segm_fixed_len_plus_lab_2(feats_dir, feats_dir_lab, files, min_len, max_len, min_len_2, max_len_2, floatX='float32',
                                                 start_from_zero=False, suffix='fea', rng=np.random, lab_clean=False):

    # First we need to loop through the files and check the minimum available lenght
    n_avl_samp = np.zeros((len(files), 1), dtype=int)
    for i,f in enumerate(files):
        n_avl_samp[i] = pytel.htk.readhtk_header(feats_dir + '/' + f + '.' + suffix)[0] 

    min_n_avl_samp = np.min( n_avl_samp )
    max_len        = np.min([max_len, min_n_avl_samp + 1] ) # Need to add 1 because max_len because the intervall is [min_len, max_len)
    n_sel_samp     = rng.randint(min_len, max_len)          # not [min_len, max_len] 
    n_sel_samp_2   = rng.randint(min_len, max_len)          # not [min_len, max_len] 

    
    n_failed = 0
    for i,f in enumerate(files):
        lab_f=f
        if (bn_clean):
            lab_f = bn_f.replace("-reverb", "")
            lab_f = bn_f.replace("-music", "")
            lab_f = bn_f.replace("-noise", "")
            lab_f = bn_f.replace("-babble", "")
        try:
            # The start_from_zero option is mainly for debugging/development 
            if start_from_zero:
                start   = 0
                start_2 = 0
            else:
                last_possible_start = n_avl_samp[i] - n_sel_samp
                start = rng.randint(0,  last_possible_start + 1) # This means the intervall is [0,last_possible_start + 1)
                                                                 #    = [0, last_possible_start]
                last_possible_start_2 = n_avl_samp[i] - n_sel_samp
                start_2 = rng.randint(0,  last_possible_start_2 + 1) # - " -
            end   = start + n_sel_samp
            end_2   = start_2 + n_sel_samp_2
            if ((i-n_failed) ==0) :
                feats_tmp = readhtk_segment(feats_dir + '/'+ f + '.' + suffix, start, end)[np.newaxis,:,:]
                feats = np.zeros((len(files), n_sel_samp, feats_tmp.shape[2]), dtype=floatX)
                feats[i-n_failed, :, :] = readhtk_segment(feats_dir + '/'+ f + '.' + suffix, start, end)[np.newaxis,:,:]

                feats_tmp_2 = readhtk_segment(feats_dir + '/'+ f + '.' + suffix, start_2, end_2)[np.newaxis,:,:]
                feats_2 = np.zeros((len(files), n_sel_samp, feats_tmp.shape[2]), dtype=floatX)
                feats_2[i-n_failed, :, :] = readhtk_segment(feats_dir + '/'+ f + '.' + suffix_2, start_2, end)[np.newaxis,:,:]
                
                lab = np.zeros( (len(files),1) ) 
                lab[i-n_failed, :] = np.genfromtxt(lab_dir + lab_f + ".hp")[start_2, end_2]
            else:
                feats[i-n_failed, :, :]= readhtk_segment(feats_dir + '/'+ lab_f + '.' + suffix, start, end)[np.newaxis,:,:]
                feats[i-n_failed, :, :]= readhtk_segment(feats_dir + '/'+ lab_f + '.' + suffix, start_2, end_2)[np.newaxis,:,:]
                lab[i-n_failed, :] = np.genfromtxt(lab_dir + lab_f + ".hp")[start_2, end_2]
        except:
            log.warning("Failed to load file %s, %s", f, lab_f)
            n_failed +=1
    if  n_failed > 0 :           
        return [feats[0:-n_failed], lab[0:-n_failed]]
    else:
        return [feats, feats_2, lab]

"""
