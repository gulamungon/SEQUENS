
from utils.misc import get_logger
log = get_logger()

import threading, time
import utils.kaldi_io
import numpy as np

def load_and_prep_dev_set_tar(egs_dir, nnet3_copy_egs_to_feats):
    dev_feat_rsp        = "ark:" + nnet3_copy_egs_to_feats + " ark:" + egs_dir + "/valid_egs.1.ark ark:- |"
    dev_feats_generator = utils.kaldi_io.read_mat_ark(dev_feat_rsp)
    dev_set             = list(dev_feats_generator)
    dev_lab             = np.array([int(dev_set[i][0].split("-")[-1]) for i in range(len(dev_set))])
    dev_feat            = np.vstack([dev_set[i][1][np.newaxis,:,:] for i in range(len(dev_set))])
    dev_idx             = list(range(0, dev_feat.shape[0]*(dev_feat.shape[1]+1), dev_feat.shape[1]))
    
    return dev_lab, dev_feat.reshape(1,-1,dev_feat.shape[2]), dev_idx


class egsBatchQue(object):

    def __init__(self, nnet3_copy_egs_to_feats, archive_path, b_size, n_archives, ark_que_length, feat_size=40, do_shuffle=False):
        self.delete = False

        self.nnet3_copy_egs_to_feats = nnet3_copy_egs_to_feats
        self.archive_path = archive_path
        self.b_size = b_size
        self.n_archives = n_archives
        self.ark_que_length = ark_que_length
        self.feat_size = feat_size
        self.do_shuffle = do_shuffle
            
        if self.do_shuffle:
            self.archive_list =  np.random.permutation( len( self.n_archives ) ) +1 # +1 Because egs indices starts from 1
            log.debug( "Shuffled the archive list " )                                
        else:
            self.archive_list =  np.arange( self.n_archives ) +1                    # np.random.permutation( len( self.n_archives ) )
            
        self.qued_archives = []
        self.archive_idx = 0       # Index for which archive to process
        self.batch_idx_ark = 0     # Index for where to start the batch within the current archive
            
        self.batch_number = 1
            
        self.batch_thread = threading.Thread( target =self.prep_archives )
        self.batch_thread.daemon = True # This will make the process die if the main process dies I THINK...???
        self.batch_thread.start()


    def prep_archives( self ):
        while not self.delete:
            if ( len(self.qued_archives ) < self.ark_que_length ):
                log.info( "Loading new archive" ) # self.qued_archives
                # If we have reached the last archive.
                if self.archive_idx == len( self.archive_list ) -1:
                    self.archive_idx = 0
                    if self.do_shuffle:
                        self.archive_list =  np.random.permutation( len( self.n_archives ) ) + 1
                        log.debug( "Shuffled the archive list " )
                feat_rsp="ark:" + self.nnet3_copy_egs_to_feats + " ark:" + self.archive_path + "/egs." + str( self.archive_list[ self.archive_idx ] ) +  ".ark ark:- |"
                feats_generator=utils.kaldi_io.read_mat_ark(feat_rsp)

                a=list(feats_generator)
                lab=np.array([int(a[i][0].split("-")[-1]) for i in range(len(a))])
                feat=np.vstack([a[i][1][np.newaxis,:,:] for i in range(len(a))])
                log.debug("loading archive done.")
                
                self.archive_idx += 1
        
                if self.do_shuffle:
                    idx=np.random.permutation( len(lab) )
                    feat = feat[idx]
                    lab = lab[idx]
                    log.debug( "Shuffled the loaded archive." )

                self.qued_archives.append( [lab,feat] )


    def get_batch(self):
        X=[]
        U=[]
        bad_tr_files = []
        tr_idx = None
        control_nb = 0
        
        log.debug ( "Retrieving batch" )

        while len(self.qued_archives) < 1:
            time.sleep(1)
    
        print("  " + str(len( self.qued_archives[0][0] )))
        print("  " + str(self.batch_idx_ark + self.b_size))
            
        if len( self.qued_archives[0][0] ) >= self.batch_idx_ark + self.b_size:
            assert ( len( self.qued_archives[0][1] ) == len( self.qued_archives[0][0] ) )

            start = self.batch_idx_ark
            end   = self.batch_idx_ark + self.b_size
            tr_feats = self.qued_archives[0][1][start:end]
            Y        = self.qued_archives[0][0][start:end]
            
            self.batch_idx_ark = end

        else:

            assert ( len( self.qued_archives[0][1] ) == len( self.qued_archives[0][0] ) )
            start = self.batch_idx_ark
            end   = self.batch_idx_ark +  self.b_size        # Will be beyond last index but this is OK
            tr_feats = self.qued_archives[0][1][start:end]
            Y        = self.qued_archives[0][0][start:end]
            self.qued_archives.pop(0)
            
            n_needed = self.b_size - Y.shape[0]

            while len(self.qued_archives) < 1 :
                time.sleep(1)

            assert ( len( self.qued_archives[0][1] ) == len( self.qued_archives[0][0] ) )
            log.debug( tr_feats.shape )
            log.debug( self.qued_archives[0][1][0:n_needed].shape )
            log.debug( Y.shape )
            log.debug( self.qued_archives[0][0][0:n_needed].shape )

            tr_feats = np.vstack( (tr_feats, self.qued_archives[0][1][0:n_needed]) )
            Y        = np.hstack( (Y, self.qued_archives[0][0][0:n_needed]) )
            
            self.batch_idx_ark = n_needed

        self.batch_number += 1
        return [[X, Y, U], [bad_tr_files], [tr_feats.astype('float32'), tr_idx],  self.batch_number, control_nb]
                                                                                                                                                                                                                                            




                
