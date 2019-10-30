

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
#from pympler.asizeof import asizeof
import sys
sys.path= sys.path + ['/mnt/matylda6/rohdin/pytel_venv_2.7/venv_20170106/lib/python2.7/site-packages/pympler/']
from asizeof import asizeof
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



# Function for loading 0th and/or 1st order stats
def load_stats(stats_dir, files, stats_order):
    first = True
    for f in files:
        try:        
            with h5py.File(stats_dir + '/' + f + '.h5', 'r', driver='core') as fh:
                if (stats_order == 0):
                    if ( first ):
                        stats = np.array(fh['N'])[None,:]
                        first = False 
                    else:
                        stats = np.concatenate((stats, np.array(fh['N'])[None,:]))
                elif (stats_order == 1):
                    if ( first ):
                        stats = np.array(fh['F'])
                        first = False 
                    else:
                        stats = np.concatenate((stats, np.array(fh['F'])))
                elif (stats_order == 2):
                    if ( first ):
                        stats0 = np.array(fh['N'])[None,:]
                        stats1 = np.array(fh['F'])
                        first = False 
                    else:
                        stats0 = np.concatenate((stats0, np.array(fh['N'])[None,:]))
                        stats1 = np.concatenate((stats1, np.array(fh['F'])))
        except IOError:
            raise Exception("Cannot open stats file [%s] for reading" % f)
    if (stats_order ==2 ):
        return [stats0, stats1]
    else:
        return [stats]


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



########################################################################################

# Functions for arranging speakar IDs. The batch generator
# will call one of these functions whenever the whole data
# set has been looped through.
def keep_spk_id_order(spk_name):
    print("The order of speaker IDs will not be changed")
    n_spk = len(spk_name)
    return np.arange(0, n_spk)    

def randomize_spk_id_order(spk_name, seed=123):
    rng    = np.random.RandomState(seed)
    n_spk  = len(spk_name)
    spk_id = rng.permutation(list(range(0, n_spk)))                   # Randomize the numbers from 0 to N - 1 
    print("The order of the speaker IDs has been randomized with seed" + str(seed))
    return spk_id


# This generator gives batches with all utterances from 
# some speakers. 
# ivec_dir, stats_dir, feat_dir should be either a path or None.
# If None, this data will not be loaded.
# stats_order: [0,1,2] for 0th, 1st, or both respectively.
def gen_mbatch_spk_all_utts(scp_info, ivec_dir, stats_dir, feat_dir, stats_order,
                          frame_step=1, max_length=30000,  
                          max_spk_per_batch=10, max_utt_per_batch=300, 
                          y_function=None, verbose=False, allow_roll_over=False,
                          arrange_spk_fcn=keep_spk_id_order, output_labs=True, output_scp_ind=False, output_utt_id=False):     
    
    # We assume "scp_info" either the scp file name or the
    # info resulting from reading it with "get_scp_info(.)"
    if not isinstance(scp_info, dict): 
        scp_info = get_scp_info(scp_info, ivec_dir, stats_dir, feat_dir) 
 
    utt2file    = scp_info[ 'utt2file' ]
    utt2spk     = scp_info[ 'utt2spk' ]
    utt2scpInd  = scp_info[ 'utt2scpInd' ]
    spk_name    = scp_info[ 'spk_name' ] 
    spk_counts  = scp_info[ 'spk_counts' ] 

    # Number of speakers                          
    n_spk = len( spk_counts )          
   
    i = 0                                                      # Minibatch index
    j = 0                                                      # Speaker index           
    
    while True:
        
        # For checking the time to create the batch 
        if (verbose):
            start_time = time.time()

        # If we have used the last data of our training set,
        # we rearrange the speaker IDs.  
        # Note: This will happen regardless off the value of allow_roll_over
        if ( j == 0 or spk_indices_batch[0] > spk_indices_batch[-1]):
            print("Order the speakers")
            spk_ids    = arrange_spk_fcn( spk_name )
            finish_set = False
            
        # Set the indices for the batch. We will add 1 speaker
        # until we reach the desired number of speakers or the
        # maximum number of utterance per batch limit

        n_spk_batch         = 0       # Number of speaker in the batch
        n_utts_batch        = 0       # Number of utterances in the batch 
        spk_indices_batch   = []      # The speaker indices we will use
        to_many_speakers    = False 
        to_many_utterances  = False 
        finish_set          = False

        while not ((to_many_speakers) or (to_many_utterances) or (finish_set)):
            n_utts_batch += spk_counts[ spk_ids[ j ] ]
            n_spk_batch  += 1
            spk_indices_batch.append( j )
            # Increase the spk index. The modulo is to start over from the 
            # beginning again when we reach the last speaker. 
            j                  = ( j + 1 ) % n_spk    
            # Check criteria for stopping the loop
            finish_set         = ( ( j == 0 ) and ( not allow_roll_over ) )
            to_many_speakers   = ( n_spk_batch  + 1  > max_spk_per_batch)   
            to_many_utterances = ( n_utts_batch + spk_counts[ spk_ids[j] ] > max_utt_per_batch )

        # Make a list of utterances (i-vectors) of the batch corresponding to 
        # all utterance of the selected speaker.
        utts  =  np.hstack([np.where(utt2spk ==s)[0]] for s in spk_ids[ spk_indices_batch ]).squeeze(axis=0)
        #print utts
        #print utts.shape
        files = [ utt2file[u] for u in utts ]

        i += 1  # Increase the batch index 

        data  = load_data(scp_info, ivec_dir, stats_dir, feat_dir, stats_order, 
                          max_length, frame_step, output_labs, output_scp_ind, utts)
        if ( output_utt_id ):
            data.append(utts)

        # Print some info about the batch      
        if (verbose):

            print(" ")
            print("***")
            print(" Preparing batch " + str(i) + " at " + time.strftime("%Y-%m-%d %H:%M:%S"))
            print("  speakers " + str( spk_indices_batch[ 0 ] ) + " - " + str( spk_indices_batch[ -1 ] )  + ", out of (n_spk) " + str(n_spk)) 
            print("  n_spk_batch " + str(n_spk_batch) + " n_utts_batch " + str(n_utts_batch))
            print("  speaker indices " + str( spk_indices_batch ))
            print("  speaker IDs " + str( spk_ids[spk_indices_batch] ))
            print("  sessions per speaker " + str ( spk_counts[spk_ids[spk_indices_batch]] ))

            out_data_size = asizeof(data)
            if (out_data_size > 1073741824):
                print("  The batch size is %0.2f GB" % ( out_data_size / 1073741824.0 ))
            elif (out_data_size > 1048576):
                print("  The batch size is %0.2f MB" % ( out_data_size / 1048576.0 ))
            elif (out_data_size > 1024):
                print("  The batch size is %0.2f KB" % ( out_data_size / 1024.0 ))
            else:
                print("  The batch size is %0.2f B" % ( out_data_size ))    
            
            print("  Time taken to prepare batch: " + str( time.time() - start_time ) + "s")
            print("  Done preparing batch at " + time.strftime("%Y-%m-%d %H:%M:%S"))
            print("***") 
        yield data



########################################################################################
# gen_mbatch_utt_groups
def create_utt_group_1( spk_name, utt2spk, spk_counts, utt2file, utt2scpInd,
                        single_per_multi_groups = 500, rng=np.random):
    s = []
    for i in range(0,len(spk_counts) ):
        # Get the utterances for a speaker.
        spk_utts = np.where(utt2spk ==i )[0][rng.permutation( spk_counts[i])]
        
        # Divide the speakers utterances into groups        
        if (spk_counts[i] ==1 ):
            s.append( [spk_utts[0]] )

        elif (spk_counts[i] % 2 == 0):
            for j in range(0, spk_counts[i], 2):
                s.append( [spk_utts[j], spk_utts[ j +1 ]  ] )

        else:
            s.append([ spk_utts[0], spk_utts[ 1], spk_utts[2] ])
            for j in range(3, spk_counts[i], 2):
                s.append( [spk_utts[j], spk_utts[ j +1 ]  ] )
 
    # Randomize s (otherwise same speakers will always end up together)
    s = [s[k] for k in rng.permutation(len(s))]
     

    # Now groups ("pairs") in s will be grouped together to larger groups. Such groups are later used to form batches.
    # "single_per_multi_groups" is how many "pairs" we want to have per such a group. 
    utts = []
    ids  = []
    n_single_spk_groups = len(s)
    # Number of large groups are rounded down. For example, if n_single_spk_groups=103 
    # and single_per_multi_groups=10, we will use 10 groups.
    n_multi_spk_groups  = n_single_spk_groups / single_per_multi_groups   
    print("n_single_spk_groups:" + str(n_single_spk_groups))
    print("n_multi_spk_groups:" + str(n_multi_spk_groups))

    # Here we group the "pairs" together. With the example numbers above, we get the following:
    # idx = ceil( 10.3 * [1,2,..,10] ) = ceil([10.3,...,103]) = [ 11, 21, 31, 42, 52, 62, 73, 83, 93, 103]
     
    # This can fail in rare cases. Something numerical makes the last index x.000000001 which is ceiled.
    # ---NEED TO BE FIXED
    # As long as this does not happen, it should give the desired output.
    # The outer "np.ceil" seems unecessary. Remove it?
    idx = np.ceil(np.ceil(n_single_spk_groups/float(n_multi_spk_groups) * (np.arange(0, n_multi_spk_groups)+1)).reshape(n_multi_spk_groups,1) )
    
    print(idx.T)
    idx = np.concatenate((np.array(0).reshape(1,1), idx), axis=0).astype(int)
    print(idx.T)
    for i in range(1, n_multi_spk_groups +1):
       
        u = np.hstack( np.array(s[j]) for j in range(idx[i-1], idx[i]))
        utts.append([u])
        ids.append(utt2spk[ u ]) 
        
    # Final shuffing to avoid all longer batches in the beginning
    r = rng.permutation(len(utts))
    utt = [ utts[k] for k in r ] 
    ids = [ ids [k] for k in r ] 
    return [utts, ids]


def make_create_utt_group_1( single_per_multi_groups, rng=np.random ):

    def create_utt_group(spk_name, utt2spk, spk_counts, utt2file, utt2scpInd ): 
        return create_utt_group_1( spk_name, utt2spk, spk_counts, utt2file, utt2scpInd, single_per_multi_groups =single_per_multi_groups, rng=rng)

    return create_utt_group
  
# ivec_dir, stats_dir, feat_dir
# stats_order: [0,1,2] for 0th, 1st, or both respectively.
def gen_mbatch_utt_groups(scp_info, ivec_dir, stats_dir, feat_dir, stats_order,
                          frame_step=1, max_length=30000,  
                          y_function=None, verbose=False, batch_selection="diag",
                          create_utt_group_list_fcn =create_utt_group_1, 
                          output_labs=True, output_scp_ind=False, output_utt_id=False,
                          rng=np.random ):     


    # We assume "scp_info" either the scp file name or the
    # info resulting from reading it with "get_scp_info(.)"
    if not isinstance(scp_info, dict): 
        scp_info = get_scp_info(scp_info, ivec_dir, stats_dir, feat_dir) 
 
    utt2file    = scp_info[ 'utt2file' ]
    utt2spk     = scp_info[ 'utt2spk' ]
    utt2scpInd  = scp_info[ 'utt2scpInd' ]
    spk_name    = scp_info[ 'spk_name' ] 
    spk_counts  = scp_info[ 'spk_counts' ] 
    

    b = 0                 # Batch index
    new_epoch = True      # This variable indicates whether a new epoch is about to start
    while True:
        
        # For checking the time to create the batch 
        if (verbose):
            start_time = time.time()
            print(" ")
            print("***")
            print(" Preparing batch " + str(b) + " at " + time.strftime("%Y-%m-%d %H:%M:%S"))


        # If a new epoch is about to start, we will group the utterances.
        # The provided function "create_utt_group_list_fcn" is used for this.
        # Note: This will happen regardless off the value of allow_roll_over
        if ( new_epoch):
            print("Obtaining utterances groups")
            [groups_u, groups_s] = create_utt_group_list_fcn( spk_name, utt2spk, spk_counts, utt2file, utt2scpInd ) # Are all these inputs needed in general???               
            new_epoch = False
            b = 0

            if(batch_selection == "rowwise"):
                [ i_ind, j_ind ] =np.triu_indices( len(groups_s) )     
            elif(batch_selection == "random"):
                [ i_ind, j_ind ] =np.triu_indices( len(groups_s) )     
                r = rng.permutation(len( i_ind ))
                i_ind = i_ind[r]
                j_ind = j_ind[r]

        # Depending on batch selection method, we load/prepare the data
        # differently
        if (batch_selection == "diag"):
            # If "diag" we only need to load the data once
            utts  = groups_u[b][0]
            files = [utt2file[u] for u in utts ]

            n_spk_batch         = len(np.unique(groups_s[b]))    # Number of speaker in the batch
            n_utts_batch        = len(groups_u[b][0] )           # Number of utterances in the batch 
 
            data  = load_data(scp_info, ivec_dir, stats_dir, feat_dir, stats_order, 
                          max_length, frame_step, output_labs, output_scp_ind, utts)
            if ( output_utt_id ):
                data.append(utts)


            if (verbose):                
                print("  i and j = : " + str(b)) 
                print("  n_spk_batch:   " + str(n_spk_batch) + " n_utts_batch " + str(n_utts_batch))
                print("  Speakers:      " + str( groups_s[b] )) 
                print("  Utterances:    " + str( groups_u[b] ))

            b += 1  # Increase the batch index 
            if b == len(groups_s):
                new_epoch = True
        

        elif(batch_selection == "rowwise"):
            # In this case, we only load a new "i" batch if it change from
            # last time but we always reload the "j" batch.
            if ( (b == 0) or ( i_ind[ b] != i_ind[ b] ) ):
                utts_i  = groups_u[i_ind[ b ]][0]
                files_i = [utt2file[u] for u in utts_i ]
                data_i  = load_data(scp_info, ivec_dir, stats_dir, feat_dir, stats_order, 
                                    max_length, frame_step, output_labs, output_scp_ind, utts_i)
                if ( output_utt_id ):
                    data_i.append(utts_i)


            utts_j = groups_u[j_ind[ b ]][0]
            files_j = [utt2file[u] for u in utts_j ]

            data_j  = load_data(scp_info, ivec_dir, stats_dir, feat_dir, stats_order, 
                                max_length, frame_step, output_labs, output_scp_ind, utts_j)
            if ( output_utt_id ):
                data_j.append(utts_j)

            n_spk_i        = len( np.unique( groups_s[ i_ind[b] ] ))    # Number of speaker in i
            n_spk_j        = len( np.unique(groups_s[ i_ind[b] ] ))     # Number of speaker in the j
            n_utt_i        = len( groups_u[ i_ind[b] ][0] )             # Number of utterances in i 
            n_utt_j        = len( groups_u[ j_ind[b] ][0] )             # Number of utterances in j 

            data = [data_i, data_j]

            if (verbose):
                print("i  " + str(i_ind[b]) +  ", j = " + str(j_ind[b]))
                print("  n_spk_i:   " + str(n_spk_i) + " n_utt_i " + str(n_utt_i))
                print("  n_spk_j:   " + str(n_spk_j) + " n_utt_j " + str(n_utt_j))
                print("  Speakers i:      " + str( groups_s[ i_ind[b] ] )) 
                print("  Speakers j:      " + str( groups_s[ j_ind[b] ] )) 
                print("  Utterances i:    " + str( groups_u[ i_ind[b] ] ))
                print("  Utterances j:    " + str( groups_u[ j_ind[b] ] ))

            b += 1  
            if b == len(i_ind):
                new_epoch = True
             
        elif(batch_selection == "random"):

            # In this case, we usually have to load both the "i" and 
            # "j" data.
            if (i_ind[ b ] == j_ind[ b ]):
                utts_i  = groups_u[i_ind[ b ]][0]
                files_i = [utt2file[u] for u in utts_i ]
                data_i  = load_data(scp_info, ivec_dir, stats_dir, feat_dir, stats_order, 
                                    max_length, frame_step, output_labs, output_scp_ind, utts_i)
                if ( output_utt_id ):
                    data_i.append(utts_i)
                utts_j = utts_i
                data_j = data_i
            else:
                utts_i  = groups_u[i_ind[ b ]][0]
                files_i = [utt2file[u] for u in utts_i ]
                data_i  = load_data(scp_info, ivec_dir, stats_dir, feat_dir, stats_order, 
                                    max_length, frame_step, output_labs, output_scp_ind, utts_i)
                if ( output_utt_id ):
                    data_i.append(utts_i)
                utts_j  = groups_u[i_ind[ b ]][0]
                files_j = [utt2file[u] for u in utts_i ]
                data_j  = load_data(scp_info, ivec_dir, stats_dir, feat_dir, stats_order, 
                                    max_length, frame_step, output_labs, output_scp_ind, utts_j)
                if ( output_utt_id ):
                    data_j.append(utts_j)

            n_spk_i        = len( np.unique( groups_s[ i_ind[b] ] ))    # Number of speaker in i
            n_spk_j        = len( np.unique(groups_s[ i_ind[b] ] ))     # Number of speaker in the j
            n_utt_i        = len( groups_u[ i_ind[b] ][0] )                # Number of utterances in i 
            n_utt_j        = len( groups_u[ j_ind[b] ][0] )                # Number of utterances in j 

            data = [data_i, data_j]

            if (verbose):
                print("i  " + str(i_ind[b]) +  ", j = " + str(j_ind[b]))
                print("  n_spk_i:   " + str(n_spk_i) + " n_utt_i " + str(n_utt_i))
                print("  n_spk_j:   " + str(n_spk_j) + " n_utt_j " + str(n_utt_j))
                print("  Speakers i:      " + str( groups_s[ i_ind[b] ] )) 
                print("  Speakers j:      " + str( groups_s[ j_ind[b] ] )) 
                print("  Utterances i:    " + str( groups_u[ i_ind[b] ] ))
                print("  Utterances j:    " + str( groups_u[ j_ind[b] ] ))

            b += 1  # Increase the batch index 
            if b == len(i_ind):
                new_epoch = True

        # Print some info about the batch      
        if (verbose):
            out_data_size = asizeof(data)
            if (out_data_size > 1073741824):
                print("  The batch size is %0.2f GB" % ( out_data_size / 1073741824.0 ))
            elif (out_data_size > 1048576):
                print("  The batch size is %0.2f MB" % ( out_data_size / 1048576.0 ))
            elif (out_data_size > 1024):
                print("  The batch size is %0.2f KB" % ( out_data_size / 1024.0 ))
            else:
                print("  The batch size is %0.2f B" % ( out_data_size ))    
            
            print("  Time taken to prepare batch: " + str( time.time() - start_time ) + "s")
            print("  Done preparing batch at " + time.strftime("%Y-%m-%d %H:%M:%S"))
            print("***") 
        yield data




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
        

# This class generates batches from an an iterator like the one above.
# It creates an additional thread which is used to load data will
# the training is ongoing. The maximum number of batches it can keep
# in que is given by "batch_que_length".
class batch_iterator(object):

    def __init__(self, it_tr, train_scp_info, load_feats_train, annoying_train=True, batch_que_length=2, batch_number=0, use_mpi=False, mpi_size=1, mpi_rank=0  ):
        self.delete           = False
        self.it_tr            = it_tr
        self.train_scp_info   = train_scp_info
        self.batch_que_length = batch_que_length
        self.qued_batches     = []
        self.batch_number     = batch_number
        self.use_mpi          = use_mpi
        self.mpi_size         = mpi_size
        self.mpi_rank         = mpi_rank
        self.load_feats_train = load_feats_train
        self.annoying_train   = annoying_train
        #self.prep_batches(break_loop=True) # To make sure they are filled from the beginning    

        if (self.batch_que_length > 0 ):
            self.batch_thread = threading.Thread( target =self.prep_batches )
            self.batch_thread.daemon = True # This will make the process die if the main process dies I THINK...???
            self.batch_thread.start()
        #else:
        #    self.batch_que_length =1  
    """    
    #def __del__(self):
    #    self.delete = True # This will stop the loop and thus finish the thread
    #    #time.sleep(5)
    #    self.batch_thread.join()
    #    print "Batch iterator thread done"
    """    

    def prep_batches(self, break_loop=False):
        while not self.delete:
            if ( (len(self.qued_batches) < self.batch_que_length) or self.batch_que_length == 0 ):
                log.debug( "Only " + str( len(self.qued_batches) ) + " batches in the que. Increasing it." )

                # [X, Y, U]  = self.it_tr.next()
                BB = next(self.it_tr)
                if len(BB)==3:
                    [X, Y, U] = BB
                elif len(BB)==4 :
                    [X, Y, U, S] = BB
                else:
                    log.error("ERROR: Wrong output from iterator")
                    
                if isinstance(U, list):
                    control_nb = U[0][0]
                else:
                    control_nb = U[0]

                if self.use_mpi:
                    # Divede utterances of the batch. Which one this worker will
                    # process depends on the mpi_rank (process number) of the process.    
                    N = U.shape[0]       # Total number of files           
                    job_indices  = np.round(np.linspace(0 , N, self.mpi_size + 1))
                    start        = int( job_indices[self.mpi_rank] ) 
                    end          = int( job_indices[self.mpi_rank + 1] ) 
                    N_this_batch = end - start
                    X = X[start:end]
                    Y = Y[start:end]
                    U = U[start:end]
                    S = U[start:end]                                   
                else:
                    start = 0
                    end   = len(Y)

                tr_files            = [ self.train_scp_info['utt2file'][u] for u in  U] 
                if not self.annoying_train:
                    [tr_feats, tr_idx ] = self.load_feats_train( tr_files )

                    bad_utts     = np.where(tr_idx[1:] - tr_idx[0:-1] == 0 )[0] # For tensor input
                    bad_tr_files = []
                    if (  len( bad_utts ) > 0 ):
                        log.info(" Got a one or more zero-length utterances. This should not happen. This utterance will be discarded but this means batch for this speaker might have been suboptimal. Should be fixed Utterance(s): ")
                        for bu in bad_utts[::-1]:
                            log.info( tr_files[bu] )
                            bad_tr_files.append(tr_files[bu])
                            Y = np.delete(Y, bu)
                            U = np.delete(U, bu)
                            if len(BB)==4 :
                                S = np.delete(S, bu)
                            tr_idx     = np.delete(tr_idx, bu)
                            # Note of-course, we don't need to remove anything from the tr_feats and tr_feats_o, since 
                            # obviously no features have been added for the uttereances where there were no features :)

                    self.batch_number += 1
                    #batch = [[X,Y,U], bad_tr_files, [tr_feats, tr_idx ], self.batch_number, control_nb, start, end]
                    batch = [BB, bad_tr_files, [tr_feats, tr_idx ], self.batch_number, control_nb, start, end]
                    log.debug("X =" + str(X) + ", Y =" + str(Y[0]) + ", U =" + str(U[0]) )
                    log.debug("tr_idx= " + str(tr_idx[0]) + ", self.batch_number= " + str(self.batch_number) + ", control_nb= " + str(control_nb) )
                else:
                    # This is for tensor input
                    self.batch_number += 1
                    tr_feats = self.load_feats_train( tr_files )
                    bad_tr_files = []
                    tr_idx = None
                    #batch = [[X,Y,U], bad_tr_files, [tr_feats, tr_idx ], self.batch_number, control_nb, start, end]
                    batch = [BB, bad_tr_files, [tr_feats, tr_idx ], self.batch_number, control_nb, start, end]


                log.debug("X =" + str(X) + ", Y =" + str(Y[0]) + ", U =" + str(U[0]) )
                log.debug("self.batch_number= " + str(self.batch_number) + ", control_nb= " + str(control_nb) )
                log.debug("self.batch_number= " + str(self.batch_number) + ", control_nb= " + str(control_nb) )
                self.qued_batches.append( batch )
                if ( break_loop ):
                    break

            else:
                if ( break_loop ):
                    break
                time.sleep(1)

    def get_batch(self):
        # The stuff commented out below may interfere in the other thread that
        # runs prep_batches. Had problems with this so keep it here as a warning.
        """
        if (len( self.qued_batches ) ==0 ):
            self.prep_batches(break_loop=True)
        """
        # This should work though, toghether with the changes above.
        while(len( self.qued_batches ) ==0 ):
            if (self.batch_que_length == 0):
                #print "A"
                self.prep_batches(break_loop=True)
            else:
                time.sleep(1)

        b = self.qued_batches.pop(0)
        log.info("Will process data %d to %d in batch." % (b[5], b[6]))                            
        return b[0:5]


# As above but takes a list of iterators and and scp info corresponding to different sets.
# Each set will be used once per batch
class batch_iterator_multi_set(object):

    def __init__(self, it, scp_info, load_feats, annoying_train=True, batch_que_length=2, batch_number=0, use_mpi=False, mpi_size=1, mpi_rank=0  ):
        self.delete           = False
        self.it               = it
        self.scp_info         = scp_info
        self.batch_que_length = batch_que_length
        self.qued_batches     = []
        self.batch_number     = batch_number
        self.use_mpi          = use_mpi
        self.mpi_size         = mpi_size
        self.mpi_rank         = mpi_rank
        self.load_feats       = load_feats
        self.annoying_train   = annoying_train

        self.n_sets  = len(self.it)
        assert len(self.it)  == len(self.scp_info)  
        
        #self.prep_batches(break_loop=True) # To make sure they are filled from the beginning    

        if (self.batch_que_length > 0 ):
            self.batch_thread = threading.Thread( target =self.prep_batches )
            self.batch_thread.daemon = True # This will make the process die if the main process dies I THINK...???
            self.batch_thread.start()
        #else:
        #    self.batch_que_length =1  
    """    
    #def __del__(self):
    #    self.delete = True # This will stop the loop and thus finish the thread
    #    #time.sleep(5)
    #    self.batch_thread.join()
    #    print "Batch iterator thread done"
    """    

    def prep_batches(self, break_loop=False):
        while not self.delete:
            if ( (len(self.qued_batches) < self.batch_que_length) or self.batch_que_length == 0 ):
                log.debug( "Only " + str( len(self.qued_batches) ) + " batches in the que. Increasing it." )

                X = []
                Y = []
                U = []
                S = []
                
                for it in self.it:
                    #[x, y, u]  = it.next()
                    B = next(it) 
                    if len(B) ==3:
                        X.append(B[0])
                        Y.append(B[1])
                        U.append(B[2])
                    elif len(B) ==4:
                        X.append(B[0])
                        Y.append(B[1])
                        U.append(B[2])
                        S.append(B[3])
                    else:
                        log.error("ERROR: Wrong output from iterator")

                if (len(S)>0):
                    BB = [X,Y,U,S]
                else:
                    BB = [X,Y,U]
                    
                control_nb = U[0][0]  
                                

                if self.use_mpi:
                    log.error("Multi set training not supported with MPI training.")
                    """
                    # Divede utterances of the batch. Which one this worker will
                    # process depends on the mpi_rank (process number) of the process.    
                    N = U.shape[0]       # Total number of files           
                    job_indices  = np.round(np.linspace(0 , N, self.mpi_size + 1))
                    start        = int( job_indices[self.mpi_rank] ) 
                    end          = int( job_indices[self.mpi_rank + 1] ) 
                    N_this_batch = end - start
                    X = X[start:end]
                    Y = Y[start:end]
                    U = U[start:end]
                    """
                else:
                    start = 0
                    #end   = len(Y)
                    end   = len(Y[0])
    
                #tr_files            = [ self.train_scp_info['utt2file'][u] for u in  U]
                
                files    = []
                for i in range(self.n_sets):
                    files += [ self.scp_info[i]['utt2file'][u] for u in  U[i]] 

                if not self.annoying_train:
                    [tr_feats, tr_idx ] = self.load_feats( files )

                    bad_utts     = np.where(tr_idx[1:] - tr_idx[0:-1] == 0 )[0] # For tensor input
                    bad_files = []
                    if (  len( bad_utts ) > 0 ):
                        log.info(" Got a one or more zero-length utterances. This should not happen. This utterance will be discarded but this means batch for this speaker might have been suboptimal. Should be fixed Utterance(s): ")
                        for bu in bad_utts[::-1]:
                            log.info( files[bu] )
                            bad_files.append(files[bu])
                            Y = np.delete(Y, bu)
                            U = np.delete(U, bu)
                            if len(BB)==4 :
                                S = np.delete(S, bu)
                            tr_idx     = np.delete(tr_idx, bu)
                            # Note of-course, we don't need to remove anything from the tr_feats and tr_feats_o, since 
                            # obviously no features have been added for the uttereances where there were no features :)

                    self.batch_number += 1
                    batch = [BB, bad_files, [tr_feats, tr_idx ], self.batch_number, control_nb, start, end]
                    log.debug("X =" + str(X) + ", Y =" + str(Y[0]) + ", U =" + str(U[0]) )
                    log.debug("tr_idx= " + str(tr_idx[0]) + ", self.batch_number= " + str(self.batch_number) + ", control_nb= " + str(control_nb) )
                else:
                    # This is for tensor input
                    self.batch_number += 1
                    tr_feats = self.load_feats( files )
                    bad_files = []
                    tr_idx = None
                    batch = [BB, bad_files, [tr_feats, tr_idx ], self.batch_number, control_nb, start, end]

                log.debug("X[0][0:3] =" + str(X[0][0:3]) + ", Y[0][0:3] =" + str(Y[0][0:3]) + ", U[0][0:3] =" + str(U[0][0:3]) )
                log.debug("self.batch_number= " + str(self.batch_number) + ", control_nb= " + str(control_nb) )
                self.qued_batches.append( batch )
                if ( break_loop ):
                    break

            else:
                if ( break_loop ):
                    break
                time.sleep(1)

    def get_batch(self):
        # The stuff commented out below may interfere in the other thread that
        # runs prep_batches. Had problems with this so keep it here as a warning.
        """
        if (len( self.qued_batches ) ==0 ):
            self.prep_batches(break_loop=True)
        """
        # This should work though, toghether with the changes above.
        while(len( self.qued_batches ) ==0 ):
            if (self.batch_que_length == 0):
                #print "A"
                self.prep_batches(break_loop=True)
            else:
                time.sleep(1)

        b = self.qued_batches.pop(0)
        log.info("Will process data %d to %d in batch." % (b[5], b[6]))                            
        return b[0:5]
    

