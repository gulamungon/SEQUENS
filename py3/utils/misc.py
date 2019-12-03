# Functions for


#import mpi4py
#import mpi4py.MPI # Needed for Anaconda ????
import logging
import os
import numpy as np


def get_logger():
    # Prepare a logger. If we are using MPI, the logger should contain
    # info about which worker the message comes from.
    """
    mpi_comm = mpi4py.MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    if (mpi_size > 1):
        logging.basicConfig(format='%(levelname)s %(asctime)s %(name)s:  %(message)s', level=logging.DEBUG)
        log = logging.getLogger("worker-" + str(mpi_rank) + "/" + str(mpi_size) +" " +__name__)
        # THe above doesn't seemd to work with the anaconda python.
    else:
        #logging.basicConfig(format='%(asctime)s %(module)s %(lineno)d:  %(message)s', level=logging.DEBUG)
        logging.basicConfig(format='%(asctime)s:  %(message)s', level=logging.DEBUG)
        log = logging.getLogger(__name__)
    """
    logging.basicConfig(format='%(asctime)s:  %(message)s', level=logging.DEBUG)
    log = logging.getLogger(__name__)
    
    return log

from utils.misc import get_logger
log = get_logger()

def make_dir_mpi( is_master, dir_to_make ):
    if is_master and not os.path.exists( dir_to_make ): 
        os.makedirs( dir_to_make )
    else:
        while not os.path.exists( dir_to_make ):
            time.sleep(1)


#### Functions for extracting, loading, saving embeddings etc.
def extract_embeddings(load_files_proc_fkn, extr_embd_fkn, embedding_size, files_all, info_offset=0, floatX='float32', b_size=20, n_embds=1):
    #b_size = 20 # Number of utterances per batch. More is probably faster but needs more memory.

    n_e   = len(files_all)
    #embds = np.zeros( (n_e , embedding_size), dtype=floatX )
    embds = [np.zeros( (n_e , embedding_size), dtype=floatX ) for i in range(n_embds) ]

    all_bad_utts     = []
    all_bad_utts_idx = []

    if (info_offset == 0):
        log.info( "Loading feats and extraction embeddings: ")

    start_f = 0
    for i in range(0, n_e, b_size):
        start = i 
        end   = np.min((start + b_size, n_e))
        #print str(start + info_offset ) + "-" + str(end + info_offset) + " --- ",
        files = files_all[start:end]

        #[feats, feats_o, e_idx ] = load_and_proc_feats_expand( e_files )                        
        #ivecs[e_start :e_end, :] = feat2embd( feats, feats_o, e_idx )[0]

        [feats, idx ] = load_files_proc_fkn( files )                        

        bad_utts = np.where( idx[1:] - idx[0:-1] == 0 )[0]
        if (  len( bad_utts ) > 0 ):
            log.debug( "Got a one or more zero-length utterances. Will be discarded. Utterance(s): ")
            for bu in bad_utts[::-1]:
                log.debug( files[bu] )
                all_bad_utts.append(files[bu])
                all_bad_utts_idx.append( start + bu )
                idx     = np.delete(idx, bu)
                # Note of-course, we don't need to remove anything from the tr_feats and tr_feats_o, since 
                # obviously no features have been added for the uttereances where there were no features :)

        embds_tmp = extr_embd_fkn( feats, idx ) # [0] In the Theano code, this came out an array inside a list.
                                                # Here it is not inside a list        
        end_f     = start_f + embds_tmp[0].shape[0]

        for i in range(n_embds):
            embds[i] [start_f :end_f, :] = embds_tmp[i]

        log.debug("Org. Index: " + str(start + info_offset ) + "-" + str(end + info_offset) + ", New index: "  + str(start_f + info_offset ) + "-"  + str(end_f + info_offset ) )
        start_f = end_f
        
    for i in range(n_embds):
            embds[i] = embds[i][0:end_f, :]
        
    #return embds[0:end_f, :], all_bad_utts, all_bad_utts_idx
    return embds, all_bad_utts, all_bad_utts_idx

def load_embeddings( embd_file ):
    if os.path.exists( embd_file ):
        log.info("Embeddings exists, loading them.")
        try:
            with h5py.File( embd_file, 'r', driver='core') as f:
                data = [ np.array( f['/embds'] ) ]
                if ( '/lab' in list(f.keys()) ):
                    log.info("No lab in data. Skipping it!")
                    data += [ np.array( f['/lab'] ) ]
                if ( '/m' in list(f.keys()) ):
                    data += [ np.array( f['/m'] ) ]
                    log.info("No m in data. Skipping it!")
                if ( '/s' in list(f.keys()) ):
                    data += [ np.array( f['/s'] ) ]
                    log.info("No s in data. Skipping it!")
            return data

        except IOError:
            raise Exception("Cannot open data file [%s] for reading" % embd_file)
    else:
        return []

def save_embeddings(out_file, embds, lab=None, m=None, s=None ):
    log.info("Saving embedding etc..")
    try:
        with h5py.File(out_file, 'w', driver='core') as f:
            f.create_dataset('embds', data=embds)
            if (lab != None):
                f.create_dataset('lab', data=lab)
            if (m != None):
                f.create_dataset('m', data=m)
            if (s != None):
                f.create_dataset('s', data=s)
    except IOError:
        raise Exception("Cannot open file [%s] for writing" % out_file )


