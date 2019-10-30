import copy
import numpy as np
from pytel.scoring import compute_results_sre16_extra

from utils.misc import get_logger
log = get_logger()


def get_eval_info(eval_conditions, list_dir, ivec_dir=None, stats_dir=None, feat_dir=None, ivec_suffix=None, stats_suffix=None, feat_suffix=None):
    from utils.mbatch_generation import check_data_exists
    # We can't load the features for all test set so they will be loaded when
    # needed. Here we check whether the features exists.
    # NOTE: This will not deal with multiple enrollment sessions properly
    eval_info = {'enroll':{},'test':{}}
    for cnd in eval_conditions:
       
        for p in ['enroll', 'test']:
            n_missing = 0
            # Load the scp
            scp = list_dir + cnd + '.' + p + '.scp'
            f = open(scp, 'r')
            spk= []
            fname = []
            for line in f:
                [s, fn]  = line.rstrip().split("=")    
                missing = check_data_exists( ivec_dir=ivec_dir, stats_dir=stats_dir, feat_dir=feat_dir, f_name=fn,
                                             ivec_suffix=ivec_suffix, stats_suffix=stats_suffix, feat_suffix=feat_suffix)
                if missing:
                    missing_x = check_data_exists( ivec_dir=ivec_dir, stats_dir=stats_dir, feat_dir=feat_dir, f_name=fn.rstrip("-x"),
                                                   ivec_suffix=ivec_suffix, stats_suffix=stats_suffix, feat_suffix=feat_suffix)
                    if not missing_x:
                        log.warning("Have removed -x so that file is not missing.")
                        fn = fn.rstrip("-x")
                        log.warning("  ---%s", fn)
                        missing = False
                if ( missing ):
                    log.debug("ivec_dir=%s, stats_dir=%s, feat_dir=%s", ivec_dir, stats_dir, feat_dir) 
                    log.debug( p + " data " + fn + " is missing. Skipping it." )
                    n_missing += 1
                else:
                    spk.append(s)
                    fname.append(fn)
            f.close()
            
            
            eval_info[p][cnd]  =[spk,  fname] 
            log.info( cnd + " " + p + " n_missing:" + str(n_missing))

    return eval_info


def evaluate(feat2ivec, dplda_test, eval_conditions=['sre16_evl_all'], mode='dev', multisession=False):
    if ( mode == 'dev' ):
        load_feats = load_feats_dev
    elif ( mode == 'test' ):
        load_feats = load_feats_test
    elif ( mode == 'train' ):
        load_feats = load_feats_train

    results = {}
    for cnd in eval_conditions:

        # First, get the embeddings.
        if (use_mpi):
            for p in ['enroll', 'test']:
                N            = len( eval_info[ p ][cnd][1] )
                job_indices  = np.round(np.linspace(0, N, mpi_size+1)).astype('int')
                start        = job_indices[mpi_rank]  
                end          = job_indices[mpi_rank + 1]  

                files = eval_info[ p ][cnd][1][start:end]
                [ivecs, bad_utts, bad_utts_idx] = extract_embeddings(load_feats, feat2embd, embedding_A_size, files, start )

                if (mpi_rank !=0):
                    mpi_comm.send([ivecs, bad_utts, bad_utts_idx], dest=0 )
                else:
                    for i in range(1, mpi_size):
                        [ivecs_r, bad_utts_r, bad_utts_idx_r] = mpi_comm.recv( source=i )

                        ivecs         = np.vstack((ivecs, ivecs_r))
                        bad_utts     += bad_utts_r 
                        bad_utts_idx += bad_utts_idx_r 

                if ( p == 'enroll' ):
                    e_ivecs = ivecs
                else:
                    t_ivecs = ivecs
        else:

            [e_ivecs, e_bad_utts, e_bad_utts_idx] = extract_embeddings(load_feats, feat2embd, embedding_A_size,
                                                                       eval_info['enroll'][cnd][1])
            [t_ivecs, t_bad_utts, t_bad_utts_idx] = extract_embeddings(load_feats, feat2embd, embedding_A_size,
                                                                       eval_info['test'][cnd][1])

            # e_ivecs = load_embeddings('/mnt/matylda6/rohdin/expts/runs/feat_2_score_nn_tf_pytel/test_1/e_ivecs.h5')[0]
            # t_ivecs = load_embeddings('/mnt/matylda6/rohdin/expts/runs/feat_2_score_nn_tf_pytel/test_1/t_ivecs.h5')[0]


        # Now score all trials and check the results.           
        if (is_master):
            if ( piggyback ):
                log.error('Piggyback training is not yet implemented in TF DPLDA (but almost)')
                #e_ivecs_pb = eval_ivec_pb['enroll'][cnd][0:n_e]
                #t_ivecs_pb = eval_ivec_pb['test'][cnd][0:n_t]
                #scr_mx     = dplda_test.score([ e_ivecs, t_ivecs, e_ivecs_pb, t_ivecs_pb], [ M1_, M2_, M_pb_1_, M_pb_2_ ] )
                #scr          = pytel.scoring.Scores(eval_info['enroll'][cnd][0], eval_info['test'][cnd][0], scr_mx)
            else:
                if multisession:
                    # Get average i-vector and the counts, then re-arrange the data a bit.
                    [e_spk, e_utt2spk]       = np.unique(eval_info['enroll'][cnd][0], return_inverse=True)
                    enr_spk                  = list(e_spk)
                    Fe, Ne                   = evaluation.get_ivecs_stats(e_ivecs, e_utt2spk )
                    [counts_u, idx, mapping] = get_multi_enroll_idx( Ne )
                    e_ivecs_new              = Fe[mapping]
                    enr_spk_new              = [ enr_spk[mp] for mp in  mapping ]

                    scr_mx       = dplda_test.score([ e_ivecs_new, t_ivecs, counts_u, idx ], [M1_p, M2_p])
                    scr          = pytel.scoring.Scores(enr_spk_new, eval_info['test'][cnd][0], scr_mx)
                else:
                    scr_mx     = dplda_test.score([ e_ivecs, t_ivecs ], [ M1_p, M2_p ] )                 
                    scr          = pytel.scoring.Scores(eval_info['enroll'][cnd][0], eval_info['test'][cnd][0], scr_mx)

            results[cnd] = evaluation.check_results(scr, cnd, keys, scr_2_train_obj, P_eff)

            # If there are any subconditions, we check them as well.
            if ( cnd in list(cnd_subsets.keys()) ):
                for c in cnd_subsets[ cnd ]: 
                    results[ c ] = evaluation.check_results(scr, c, keys, scr_2_train_obj, P_eff)

        if (use_mpi):
            if (is_master):
                for i in range(1, mpi_size):
                    mpi_comm.send(results, dest=i )
            else:
                results = mpi_comm.recv( source=0 )
    return results 


    
def check_evl_loss_fcn(tar_s, non_s, scr_2_train_obj, P_eff=None):
        
    n_tar = len( tar_s )
    n_non = len( non_s )

    # Not sure if we want to normalize like this for other objectives than
    # cllr. So if P_eff is None (the default) we do not apply the normalization. 
    if (P_eff == None):
        P_eff = 0.5
        
    cllr_norm        = -( P_eff * np.log(P_eff) + (1-P_eff) * np.log(1-P_eff) )
    tar_weight       = P_eff/(n_tar * cllr_norm)
    non_tar_weight   = (1-P_eff)/(n_non * cllr_norm)
        
    #L_tar = np.sum(tar_weight     * scr_2_train_obj( -1*(tar_s - tau) ) )   
    #L_non = np.sum(non_tar_weight * scr_2_train_obj(     non_s - tau  ) )
    L_tar = np.sum(tar_weight     * scr_2_train_obj( tar_s,  1.0 ) )   
    L_non = np.sum(non_tar_weight * scr_2_train_obj( non_s, -1.0 ) )         

    L_tot = L_tar + L_non
        
    return [L_tot, L_tar, L_non]

def check_results(scr, cnd, keys, scr_2_train_obj, P_eff=None):
    print(" ")
    print(cnd) 
    r = compute_results_sre16_extra( [ scr ], [keys[ cnd ]] )
    print(r)
    # Check the loss we use as training objective
    [tar_s, non_s]        = keys[ cnd ].get_tar_non(scr)
    [L_tot, L_tar, L_non] = check_evl_loss_fcn(tar_s, non_s, scr_2_train_obj, P_eff=P_eff)            
    r['L']                = L_tot
    print("Training objective: Tar =" + str(L_tar) + ", non-tar  =" + str(L_non) + ", total  =" + str(L_tot))
    return r



def kaldi_style_ivec_proc(ivecs, kaldi_plda, glob_mean =np.array([]), transform =np.array([]), utt2spk =np.array([]), do_kaldi_norm=True):
    ivec_dim = ivecs.shape[1]

    assert( isinstance(kaldi_plda, list) )
    assert( len(kaldi_plda) ==3 )
    plda_mean   = kaldi_plda[0]
    plda_trans  = kaldi_plda[1]    # Transform for simultaneous diagonalization
    plda_bc_var = kaldi_plda[2]    # Between-class variance 

    # Average the i-vectors for each speaker. Also, check counts, i.e.,  #i-vectors per spaker.
    if ( len(utt2spk) != 0 ):
        n            = len(np.unique(utt2spk))
        e  = np.zeros([n, ivec_dim])
        counts = np.zeros([n, 1], dtype=int)
        for i in range( n ):
            tmp = ivecs[np.where(utt2spk==i),:].reshape(-1, ivec_dim)
            counts[i] = tmp.shape[0]
            e[i]  = np.mean(tmp, axis=0)
        max_count = int(max(counts)[0] )
    else:
        n = len(ivecs)
        max_count = 1
        counts = np.ones([n,1], dtype=int)
        e  = copy.copy(ivecs)

        
    if (len(glob_mean) != 0):
        e -= glob_mean
        
    if (len(transform) != 0):
        assert( isinstance(transform, list) )
        assert( len(transform) ==2 )
        e  = e.dot(transform[0].T) - transform[1]  

    # Length-norm, kaldi style
    e /= np.sqrt(( e **2).sum(axis=1)[:,np.newaxis]) # This would make the lengths equal to one
    e *= np.sqrt(e.shape[1])                         # This is the Kaldi style.

    # This transforms the i-vector into a space where the mean is zero and
    # both within-class covariance and between-class covariance are diagonal.
    e = (e-plda_mean).dot(plda_trans.T)

    if do_kaldi_norm:
        print("Applying kaldi style normalization where the normalization depends on the number of sessions")
        dim = e.shape[1]
        norm_vector = np.zeros([max_count, dim ])
        for i in range(max_count):
            norm_vector[i] = 1.0/(plda_bc_var + 1.0/(i+1))

        nn = 0
        for i in range(e.shape[0]):
            d = (e[i,:] **2).dot(norm_vector[ counts[i] -1].squeeze() )
            n = np.sqrt( dim /( d ))
            e[i,:] = e[i,:] * n
            nn += n
        print("Average normalization factor: " + str( nn / e.shape[0] ))
       
    return e, counts


def kaldi_style_ivec_proc_full(ivecs, kaldi_plda, glob_mean =np.array([]), transform =np.array([]), utt2spk =np.array([]), do_kaldi_norm=True):
    ivec_dim = ivecs.shape[1]

    assert( isinstance(kaldi_plda, list) )
    assert( len(kaldi_plda) ==3 )
    plda_mean   = kaldi_plda[0]
    plda_B  = kaldi_plda[1]    # Between-class variance 
    plda_W = kaldi_plda[2 ]    # Within-class variance 
    
    if ( len(utt2spk) != 0 ):
        n            = len(np.unique(utt2spk))
        e  = np.zeros([n, ivec_dim])
        counts = np.zeros([n, 1], dtype=int)
        for i in range( n ):
            tmp = ivecs[np.where(utt2spk==i),:].reshape(-1, ivec_dim)
            counts[i] = tmp.shape[0]
            e[i]  = np.mean(tmp, axis=0)
        max_count = int(max(counts)[0] )
    else:
        n = len(ivecs)
        max_count = 1
        counts = np.ones([n,1], dtype=int)
        e  = copy.copy(ivecs)

    if (len(glob_mean) != 0):
        e -= glob_mean
        
    if (len(transform) != 0):
        assert( isinstance(transform, list) )
        assert( len(transform) ==2 )
        e  = e.dot(transform[0].T) - transform[1]  

    # Length-norm, kaldi style
    e /= np.sqrt(( e **2).sum(axis=1)[:,np.newaxis]) # This would make the lengths equal to one
    e *= np.sqrt(e.shape[1])                         # This is the Kaldi style.

    # This transforms the i-vector into a space where the mean is zero and
    # both within-class covariance and between-class covariance are diagonal.
    e = (e-plda_mean)

    if do_kaldi_norm:
        print("Applying kaldi style normalization where the normalization depends on the number of sessions")
        dim = e.shape[1]
        norm_matrix = {} #np.zeros([max_count, dim ])
        for i in np.unique(counts):
            norm_matrix[i] = np.linalg.inv(plda_B + plda_W / i )

        nn = 0
        for i in range(e.shape[0]):
            d = (e[i,:].dot(norm_matrix[ counts[i][0] ] )).dot(e[i,:].T)
            #print norm_matrix[ counts[i][0] ][0,0]
            #print e[i,:][0]
            #print d
            n = np.sqrt( dim /( d ))
            e[i,:] = e[i,:] * n
            nn += n
        print("Average normalization factor: " + str( nn / e.shape[0] ))

        
    return e, counts


def kaldi_style_ivec_proc_simple(ivecs, glob_mean, transform, plda_mean):
    ivec_dim = ivecs.shape[1]
    e  = copy.copy(ivecs)        

    if (len(glob_mean) != 0):
        e -= glob_mean
        
    if (len(transform) != 0):
        assert( isinstance(transform, list) )
        assert( len(transform) ==2 )
        e  = e.dot(transform[0].T) - transform[1]  

    # Length-norm, kaldi style
    #e /= np.sqrt(( e **2).sum(axis=1)[:,np.newaxis]) # This would make the lengths equal to one
    #e *= np.sqrt(e.shape[1])                         # This is the Kaldi style.

    # This transforms the i-vector into a space where the mean is zero and
    # both within-class covariance and between-class covariance are diagonal.
    #e = (e-plda_mean)

        
    return e


def get_ivecs_stats(ivecs, utt2spk):
    ivec_dim = ivecs.shape[1]

    assert( len(utt2spk) == ivecs.shape[0])

    n            = len(np.unique(utt2spk))
    e  = np.zeros([n, ivec_dim])
    counts = np.zeros([n, 1], dtype=int)
    for i in range( n ):
        tmp = ivecs[np.where(utt2spk==i),:].reshape(-1, ivec_dim)
        counts[i] = tmp.shape[0]
        e[i]  = np.mean(tmp, axis=0)
    max_count = int(max(counts)[0] )
       
    return e, counts
