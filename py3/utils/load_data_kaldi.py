import numpy as np
import kaldi_io


def load_kaldi_feats(files):
    
    pass




def load_kaldi_feats_segm_same_dur(rng, files, min_length, max_length, n_avl_samp, start_from_zero):
    
    min_n_avl_samp = np.min( n_avl_samp )
    max_len        = np.min( [min_n_avl_samp+1, max_length] ) # Need to add 1 because max_len because the intervall is [min_len, max_len)?????!!!!???
    n_sel_samp = rng.randint(min_length, max_len)             # not [min_len, max_len]
    start      = []
    end        = []
    for i,f in enumerate(files):
        # The start_from_zero option is mainly for debugging/development
        if start_from_zero:
            start.append(0)
        else:
            last_possible_start = n_avl_samp[i] - n_sel_samp
            start.append(rng.randint(0,  last_possible_start + 1)[0] )# This means the intervall is [0,last_possible_start + 1) = [0, last_possible_start]
            end.append(start[-1] + n_sel_samp)
    ff = [ "xxx {}[{}:{},:]".format( files[i], start[i], end[i] ) for i in range(len(files)) ]
    data = [ rr[1] for rr in kaldi_io.read_mat_scp(ff) ]
    data = np.stack( data, axis=0 )
    #print(data.shape)
    return data
        

def load_kaldi_feats_segm_same_dur_plus_lab(rng, files, min_length, max_length, n_avl_samp, lab_dir, f_ids, vad_files, start_from_zero=False):

    min_n_avl_samp = np.min( n_avl_samp )
    max_len        = np.min( [min_n_avl_samp+1, max_length] ) # Need to add 1 because max_len because the intervall is [min_len, max_len)?????!!!!???
    n_sel_samp = rng.randint(min_length, max_len)             # not [min_len, max_len]
    start      = []
    end        = []

    vad  = [kaldi_io.read_vec_flt(vf).astype(bool) for vf in vad_files]
    lab = [ np.genfromtxt(lab_dir + f + ".hp")[vad[i]] for i,f in enumerate(f_ids)  ]

    assert(len(lab[0]) == n_avl_samp[0])
    
    for i,f in enumerate(files):
        # The start_from_zero option is mainly for debugging/development
        if start_from_zero:
            start.append(0)
        else:
            last_possible_start = n_avl_samp[i] - n_sel_samp
            start.append(rng.randint(0,  last_possible_start + 1)[0] )# This means the intervall is [0,last_possible_start + 1) = [0, last_possible_start]
            end.append(start[-1] + n_sel_samp)
    ff = [ "xxx {}[{}:{},:]".format( files[i], start[i], end[i] ) for i in range(len(files)) ]
    data = [ rr[1] for rr in kaldi_io.read_mat_scp(ff) ]
    data = np.stack( data, axis=0 )

    lab = np.array([ l[start[i]:end[i]] for i,l in enumerate(lab)])
    
    return data, lab

