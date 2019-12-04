

# Code for generating mini-batches. The outout can be any
# combination of: 
# features/0-stats/1-stats/i-vectors/labels/scp-indices
#
# scp-indices means which position in the data had. This 
# can for example be used for looking up a trial weight 
# from a list or for accessing data that is already 
# loaded to the GPU.
#
# There are three different generators:
#
# gen_mbatch_spk_all_utts: 
#  Generates minibatches that each contains all the utterances
#  from a small set of speakers. The batch size is controlled
#  by setting The order of the speakers is "max_spk_per_batch"
#  and "max_utt_per_batch". The order of speakers is 
#  determined by a function, "randomize_spk_id_order",  which 
#  will be called before the training starts as well as after 
#  all speakers in the training set have been used. This function
#  can be provided by the user. For example, it can be a function
#  that simply randomizes the speakers. But we could also consider
#  to make a function that puts e.g. 10 "similar" speakers in 
#  consequtively and max_spk_per_batch=10 to get them in the same
#  batch.

# gen_mbatch_utt_groups
#  This generator gives batches according to a list of "utterance groups"
#  A function that generate the utterance groups needs to be provided.
#  This function will be called before the training starts as well as 
#  after all data have been used so that it can be re-shuffled (or re-ordered
#  according to some other rule)
#  Assumming it gives [g1, g2,...] where gi is a group of utterances, e.g.,
#  g1 = [u11, u12, u13,...]
#  The generator has three options to form mini-batches.
#  1, "diag":     Minibatches are (g1 - g1), (g2 - g2), (g3 - g3),... 
#  2, "rowwise":  Minibatches are (g1 - g1), (g1 - g2), (g1 - g3),...,(g2 - g2), (g2 - g3)..., (g3 - g3) 
#                 All possible batches are looped throuhg in order. Advantage: One use the data more per
#                 copying to the GPU (BUT THIS REMAINS TO BE IMPLEMENTED.)  
#                 Disadvantage: Consequetive batches, e.g. (g1 - g1) and (g1 - g2) are more statistically dependent.   
#  3, "random":   Minibatches are (gi - gj),... Indices "i" and "j" are generated randomly until all 
#                 possible batces have been used
#
# gen_mbatch_trials --NOT IMPLEMENTED YET--
#  Will take a list of (difficult) trials and divide into batches.



from utils.misc import get_logger
log = get_logger()


import h5py, os, time
from pympler.asizeof import asizeof
import sys
sys.path= sys.path + ['/mnt/matylda6/rohdin/pytel_venv_2.7/venv_20170106/lib/python2.7/site-packages/pympler/']
#from asizeof import asizeof
import numpy as np
from pytel.htk import readhtk, readhtk_segment
import threading



########################################################################################
### General functions for processing the scp, loading data, etc.

# Check that, ivec, stats, and features exists
def check_data_exists(ivec_dir, stats_dir, feat_dir, f_name, ivec_suffix=None, stats_suffix=None, feat_suffix=None, stats_clean=False):
    # The MISSING info is a bit misleading since it will be printed if an earlier check has failed.
    missing=False
    if (ivec_dir != None):
        if (ivec_suffix==None):
            missing = not (os.path.isfile( ivec_dir + '/' + f_name + '.i.gz' )  or os.path.isfile( ivec_dir + '/' + f_name + '.xvec' ) )
            if missing:
                log.info ("MISSING: " + ivec_dir + '/' + f_name + '.i.gz')
        else:
            missing = not os.path.isfile( ivec_dir + '/' + f_name + "." + ivec_suffix)
            if missing:
                log.info ("MISSING: " + ivec_dir + '/' + f_name + "." + ivec_suffix )
            
    if (stats_dir != None):
        st_f_name = f_name

        if stats_clean:
            st_f_name = st_f_name.replace("reverb/", "clean/")
            st_f_name = st_f_name.replace("music/", "clean/")
            st_f_name = st_f_name.replace("noise/", "clean/")
            st_f_name = st_f_name.replace("babble/", "clean/")
            
        if (stats_suffix==None):                                  
            missing = missing or not os.path.isfile( stats_dir + '/' + st_f_name + '.h5')
            if missing:
                log.info ("MISSING: " + stats_dir + '/' + st_f_name + '.h5')
        elif (stats_suffix=='fea'):                                  
            missing = missing or not os.path.isfile( stats_dir + '/' + st_f_name + '.fea')
            if missing:
                log.info ("MISSING: " + stats_dir + '/' + st_f_name + '.fea')
            try:
                tmp_h = readhtk_segment(stats_dir + '/' + st_f_name + '.fea', 0, 1)[np.newaxis,:,:]
            except Exception as e:
                log.info ("Failed readhtk_segment: " + stats_dir + '/' + st_f_name + '.fea')
                log.info(str(e))
                mising = True
        else:
            missing = missing or not os.path.isfile( stats_dir + '/' + st_f_name + "." + stats_suffix)
            if missing:
                log.info ("MISSING: " + stats_dir + '/' + st_f_name + "." + stats_suffix)
                
            
    if (feat_dir != None):
        if (feat_suffix==None):                                  
            missing = missing or not os.path.isfile( feat_dir + '/'+ f_name + '.fea' )
            if missing:
                log.info ("MISSING: " + feat_dir + '/'+ f_name + '.fea')
        else:
            missing = missing or not os.path.isfile( feat_dir + '/'+ f_name +  "." + feats_suffix)
            if missing:
                log.info ("MISSING: " + feat_dir + '/'+ f_name +  "." + feats_suffix)
                
    return missing

# Gathers speaker info from an scp.
def get_scp_info(scp, ivec_dir, stats_dir, feat_dir, ivec_suffix=None, stats_suffix=None, feat_suffix=None, stats_clean=False):
    
    print("Processing scp " + scp)
    utt2file = []
    spk_ids    = []
    utt2sideInfo = []

    f = open(scp, 'r')

    n_unk     = 0
    n_missing = 0

    # We will remove missing files. And we may need
    # to know how to map the remaining files to their
    # original position in the scp.
    utt2scpInd=[]
    scpInd    = 0
    for line in f:
 
        scp_info  = line.rstrip().split("=")    
        n_scp_col = len(scp_info)

        if ( n_scp_col == 1 ):
            
            f_name = scp_info[0]
            
            if not check_data_exists( ivec_dir, stats_dir, feat_dir, f_name, ivec_suffix, stats_suffix, feat_suffix, stats_clean ):
                utt2file.append( f_name )
                spk_ids.append("unk" + str(n_unk) )
                n_unk += 1
                utt2scpInd.append(scpInd)
            else:
                n_missing += 1


        elif ( n_scp_col == 3 ):
            f_name = scp_info[1]
            if not check_data_exists( ivec_dir, stats_dir, feat_dir, f_name, ivec_suffix, stats_suffix, feat_suffix, stats_clean ):
                spk_ids.append( scp_info[0] )
                utt2file.append( f_name )
                utt2sideInfo.append( scp_info[2] )
            # Added 2019-11-08. Should be here, shouldn't it?    
            else:
                n_missing += 1

        else:

            f_name = scp_info[1]

            if not check_data_exists( ivec_dir, stats_dir, feat_dir, f_name, ivec_suffix, stats_suffix, feat_suffix, stats_clean ):

                utt2file.append( f_name )
                scp_info  = scp_info[0].split(" ")    
                n_scp_col = len(scp_info)

                utt2scpInd.append(scpInd)
                if ( n_scp_col == 1 ): 
                    spk_ids.append( scp_info[0] )
                else:
                    spk_ids.append( scp_info[1] )

            else:
                n_missing += 1
        scpInd += 1
    if ( n_missing > 0 ):
        print("WARNING: A total of " + str(n_missing) + " entries in the scp file with missing data have been skipped")

    if (n_unk > 0):
        print("WARNING: A total of " + str(n_unk) + " files did not have a speaker ID (excluding files with missing data).")
        print("         These files have been given a unique ID each.")

    f.close()


    [ spk_name, utt2spk, spk_counts ] = np.unique( spk_ids, return_inverse=True, return_counts=True )
    print("Processed " + str(scpInd) + " scp entries")
    print("Found " + str(len(utt2spk)) +  " utterances and " + str(len(spk_name)) + " speakers (including utterances with missing speaker ID)")

    # utt2scpInd one is mainly used to find original scp index in case some files
    # have been missing and thus the number of files in e.g, utt2spk does
    # not match the original scp length. 
    scp_info = { 'spk_name' : spk_name, 'utt2spk' : utt2spk, 
                 'spk_counts' : spk_counts, 'utt2file' : utt2file, 
                 'utt2scpInd' : utt2scpInd, 'utt2sideInfo' : utt2sideInfo }
    return scp_info



# Gathers speaker info from an scp.
def get_scp_info_master(scp, ivec_dir, stats_dir, feat_dir, ivec_suffix="embd_A"):
    
    print("Processing scp " + scp)
    utt2file = []
    spk_ids    = []
    utt2sideInfo = []

    f = open(scp, 'r')

    n_unk     = 0
    n_missing = 0

    # We will remove missing files. And we may need
    # to know how to map the remaining files to their
    # original position in the scp.
    utt2scpInd=[]
    scpInd    = 0
    for line in f:
 
        scp_info  = line.rstrip().split(" ")    
        n_scp_col = len(scp_info)

        f_name = scp_info[14]
        
        if not check_data_exists( ivec_dir, stats_dir, feat_dir, f_name, ivec_suffix=ivec_suffix):
            utt2file.append( f_name )
            spk_ids.append( scp_info[1] )

        else:
            n_missing += 1
            
        scpInd += 1
    if ( n_missing > 0 ):
        print("WARNING: A total of " + str(n_missing) + " entries in the scp file with missing data have been skipped")

    if (n_unk > 0):
        print("WARNING: A total of " + str(n_unk) + " files did not have a speaker ID (excluding files with missing data).")
        print("         These files have been given a unique ID each.")

    f.close()


    [ spk_name, utt2spk, spk_counts ] = np.unique( spk_ids, return_inverse=True, return_counts=True )
    print("Processed " + str(scpInd) + " scp entries")
    print("Found " + str(len(utt2spk)) +  " utterances and " + str(len(spk_name)) + " speakers (including utterances with missing speaker ID)")
 
    scp_info = { 'spk_name' : spk_name, 'utt2spk' : utt2spk, 
                 'spk_counts' : spk_counts, 'utt2file' : utt2file, 
                 'utt2scpInd' : utt2scpInd, 'utt2sideInfo' : utt2sideInfo }
    return scp_info





# This function is copied from Keras (downloaded 20170106)
def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    '''Pads each sequence to the same length:
    the length of the longest sequence.

    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.

    Supports post-padding and pre-padding (default).

    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.

    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


# Load features. 
def load_feats(feat_dir, files, max_length, frame_step):           
    d = [readhtk(feat_dir + '/'+ f + '.fea')[::frame_step,:] for f in files]
    return  [pad_sequences(d, max_length, dtype=float)]

# Load the data we want
def load_data(scp_info, ivec_dir, stats_dir, feat_dir, stats_order, max_length, frame_step, output_labs, output_scp_ind, utts):

    utt2file    = scp_info[ 'utt2file' ]
    files       = [ utt2file[u] for u in utts ]
    utt2spk     = scp_info[ 'utt2spk' ]
    utt2scpInd  = scp_info[ 'utt2scpInd' ]

    data = []
    if (ivec_dir != None):
        data = data + [ np.vstack( np.loadtxt( ivec_dir+'/'+r+'.i.gz' )[None,:] for r in files) ]

    if (stats_dir != None):
        data = data + load_stats(stats_dir, files, stats_order)

    if (feat_dir != None): 
        data = data + load_feats(feat_dir, files, max_length, frame_step)

    output = [data]

    if (output_labs):
        lab = utt2spk[utts]
        output += [lab]

    # Append the scp indices for the selected utterances if wanted. 
    # (Can be used for e.g. looking up trial weigts or 
    # for obtaining e.g., i-vectors or 0th stats if these
    # are alreade stored on the GPU)
    if (output_scp_ind):
        batch_scp_ind = [utt2scpInd[u] for u in utts]
        output += [batch_scp_ind] 

    return output

# Returns an iterator that gives batches consisting of "n_spk_per_batch"
# randomly selected speakers with "n_utt_per_spk" segments each. 
def gen_mbatch_spk_bal(scp_info, ivec_dir, stats_dir, feat_dir, stats_order,
                       frame_step=1, max_length=30000,  
                       y_function=None, verbose=False,
                       arrange_spk_fcn = None, n_spk_per_batch=50, n_utt_per_spk=2,
                       output_labs=True, output_scp_ind=False, output_utt_id=False,
                       rng=np.random, out2put_utt2sideInfo=False ):     
    
    # We assume "scp_info" either the scp file name or the
    # info resulting from reading it with "get_scp_info(.)"
    if not isinstance(scp_info, dict): 
        scp_info = get_scp_info(scp_info, ivec_dir, stats_dir, feat_dir) 
 
    utt2file    = scp_info[ 'utt2file' ]
    utt2spk     = scp_info[ 'utt2spk' ]
    utt2scpInd  = scp_info[ 'utt2scpInd' ]
    spk_name    = scp_info[ 'spk_name' ] 
    spk_counts  = scp_info[ 'spk_counts' ]

    if out2put_utt2sideInfo:
        utt2sideInfo  = scp_info[ 'utt2sideInfo' ] 
    
    n_spk = len(spk_name) 

    # Randomize the speakers
    spk_arr =  [] 
    
    # This list has one entry per speaker which keeps a list of the speakers utterances
    spk2utt_fixed = [np.where(utt2spk ==i)[0] for i in range(n_spk)]
    
    # As above but the speakers' utterance lists are randomized and gradually poped when
    # batches are created. Whenever a list becomes empty, a new is created randomly.
    spk2utt_rand = [ spk2utt_fixed[i][ rng.permutation(len(spk2utt_fixed[i])) ] for i in range(n_spk)  ]

    while True:
        
        if len(spk_arr) < n_spk_per_batch:
            spk_arr = spk_arr + list(rng.permutation( n_spk )) 

        spk_this_batch = spk_arr[0:n_spk_per_batch]
        del spk_arr[0:n_spk_per_batch]

        utts = np.array([], dtype=int)
        for i in range( len(spk_this_batch) ):
            ii = spk_this_batch[i]
            if ( len( spk2utt_rand[ii] ) < n_utt_per_spk ):
                spk2utt_rand[ii] = np.concatenate( [spk2utt_rand[ii], spk2utt_fixed[ii][ rng.permutation(len(spk2utt_fixed[ii])) ] ] )
                                                                                                     
            utts  = np.concatenate((utts, spk2utt_rand[ii][0:n_utt_per_spk]))
            spk2utt_rand[ii] = np.delete( spk2utt_rand[ii], np.arange(0, n_utt_per_spk) )
        
        files = [utt2file[u] for u in utts ]
###
        data  = load_data(scp_info, ivec_dir, stats_dir, feat_dir, stats_order, 
                      max_length, frame_step, output_labs, output_scp_ind, utts)
        
        if ( output_utt_id ):
            data.append(utts)

        if ( out2put_utt2sideInfo ):
            sideInfo = [ utt2sideInfo[u] for u in utts ]
            data.append( sideInfo )
            
        yield data



# Returns an iterator that gives batches consisting of "n_spk_per_batch"
# randomly selected speakers with "n_utt_per_spk" segments each. 
def gen_mbatch_spk_bal_semi(scp_info, ivec_dir, stats_dir, feat_dir, stats_order,
                            frame_step=1, max_length=30000,  
                            y_function=None, verbose=False,
                            arrange_spk_fcn = None, n_spk_per_batch=50, n_utt_per_spk=2,
                            output_labs=True, output_scp_ind=False, output_utt_id=False,
                            rng=np.random, out2put_utt2sideInfo=False, n_unk_per_batch=50 ):     
    
    # We assume "scp_info" either the scp file name or the
    # info resulting from reading it with "get_scp_info(.)"
    if not isinstance(scp_info, dict): 
        scp_info = get_scp_info(scp_info, ivec_dir, stats_dir, feat_dir) 
 
    utt2file    = scp_info[ 'utt2file' ]
    utt2spk     = scp_info[ 'utt2spk' ]
    utt2scpInd  = scp_info[ 'utt2scpInd' ]
    spk_name    = scp_info[ 'spk_name' ] 
    spk_counts  = scp_info[ 'spk_counts' ]


    if out2put_utt2sideInfo:
        utt2sideInfo  = scp_info[ 'utt2sideInfo' ] 

    # Make one spk name is "unk"
    assert ( np.asarray( spk_name == "unk"  ).nonzero()[0][0] )

    unk_spk_id = np.asarray( spk_name == "unk"  ).nonzero()[0][0]

    # A list of the speaker IDs that are not "unk"
    spk_fixed = list(range( len(spk_name)))
    del spk_fixed[unk_spk_id]
    
    n_spk = len(spk_name) -1


    # For now we assert unk is the lst ID. Otherwise we need to update the returned labels. The train script assumes the last ID is unk.
    assert (unk_spk_id == n_spk)
    
    # Randomize the speakers
    spk_arr =  [] 
    
    # This list has one entry per speaker which keeps a list of the speakers utterances
    spk2utt_fixed = [np.where(utt2spk ==i )[0] for i in spk_fixed]
    log.info("Number of speakers: %d", len(spk2utt_fixed))
    n_sup_utt = sum([len(spk2utt_fixed[i]) for i in range(n_spk) ])
    log.info("Number of utterances with speaker ID: %d", n_sup_utt)

    # As above but the speakers' utterance lists are randomized and gradually poped when
    # batches are created. Whenever a list becomes empty, a new is created randomly.
    spk2utt_rand = [ spk2utt_fixed[i][ rng.permutation(len(spk2utt_fixed[i])) ] for i in range(n_spk)  ]


    unk2utt_fixed = np.where( utt2spk ==1000 )[0]
    unk2utt_rand  = np.random.permutation( unk2utt_fixed )
    log.info("Number of utterances with unknown speakers: %d", unk2utt_fixed.shape[0])


    
    while True:

        # This part is for the supervised data
        if len(spk_arr) < n_spk_per_batch:
            spk_arr = spk_arr + list(rng.permutation( n_spk )) 

        spk_this_batch = spk_arr[0:n_spk_per_batch]
        del spk_arr[0:n_spk_per_batch]

        utts = np.array([], dtype=int)
        for i in range( len(spk_this_batch) ):
            ii = spk_this_batch[i]
            if ( len( spk2utt_rand[ii] ) < n_utt_per_spk ):
                spk2utt_rand[ii] = np.concatenate( [spk2utt_rand[ii], spk2utt_fixed[ii][ rng.permutation(len(spk2utt_fixed[ii])) ] ] )
                                                                                                     
            utts  = np.concatenate((utts, spk2utt_rand[ii][0:n_utt_per_spk]))
            spk2utt_rand[ii] = np.delete( spk2utt_rand[ii], np.arange(0, n_utt_per_spk) )

        # This part is for the unsupervised data
        if ( len( unk2utt_rand ) < n_unk_per_batch ):
            unk2utt_rand = np.concatenate( [unk2utt_rand, unk2utt_fixed[ rng.permutation(len(unk2utt_fixed)) ] ] )

        utts  = np.concatenate((utts, unk2utt_rand[0:n_unk_per_batch]))
        unk2utt_rand = np.delete( unk2utt_rand, np.arange(0, n_unk_per_batch) )
            
        files = [utt2file[u] for u in utts ]
###
        data  = load_data(scp_info, ivec_dir, stats_dir, feat_dir, stats_order, 
                      max_length, frame_step, output_labs, output_scp_ind, utts)
        
        if ( output_utt_id ):
            data.append(utts)

        if ( out2put_utt2sideInfo ):
            sideInfo = [ utt2sideInfo[u] for u in utts ]
            data.append( sideInfo )
            
        yield data


########################################################################################



