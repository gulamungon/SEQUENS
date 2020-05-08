#!/usr/bin/env python

# This script loads and Kaldi x-vector model and extracts x-vectors.
#
# NOTES
# * A suffix "_" denotes a TF symbolic variable and "_p" denotes a TF placeholder variable


floatX='float32'

import sys, os, logging, time, logging, h5py, copy, subprocess, argparse, pickle, re 
import numpy as np
import tensorflow as tf

from utils.mbatch_generation import *
#from utils.load_data import *
from tensorflow_code.load_save import load_tf_model
from utils.misc import get_logger, extract_embeddings, save_embeddings
import kaldi_io

def apply_instance_norm(X):
    Xs = np.reshape(X,(X.shape[0],X.shape[1]*X.shape[2]))
    M = np.mean(Xs, axis=1)
    S = np.std(Xs, axis=1)
    return (X - M[:,np.newaxis,np.newaxis])/S[:,np.newaxis,np.newaxis]

        
def get_configuration():
    
    parser = argparse.ArgumentParser(
        description = 'Transforms i-vectors with a neural network' )
    parser.add_argument(
        '-o', '--out_dir',     type =str, help ='Directory where to dump the xvectors',     required =True)
    parser.add_argument(
        '-v', '--vad_scp',     type =str, help ='scp for  VAD labels is',                   required =False, default="None")
    parser.add_argument(
        '-l', '--scp',         type =str, help ='scp',                                      required =True)
    parser.add_argument(
        '-m', '--model',       type =str, help ='Path to model file',                       required =True)
    parser.add_argument(
        '-w', '--window_size', type =int, help ='Size of window in frames',                 required =False, default=200)
    parser.add_argument(
        '-s', '--shift',       type =int, help ='Shift in frames',                          required =False, default=150)
    parser.add_argument(
        '-nc', '--n_cores',    type =int, help ='Number of cores. -1 means let TF decide.', required =False, default=-1)
    parser.add_argument(
        '-x', '--extract',     type =str, help ='Name of variable to extract. Comma separate if more than one.', required =True)
    parser.add_argument(
        '-si', '--side_info',  type =str, help ='side_info',  required =False, default="None")
    parser.add_argument(
        '-so', '--store_option', type =str, help ='Whether to store different embds from an utterance (e.g. embd_A or embd_B) concatenated or separately [concat,separate]', required =False, default="separately")
    parser.add_argument(
        '-sf', '--store_format', type =str, help ='Format for storing. htk=one htk file per utterance. h5=one h5 for all utts in the scp.',  required =False, default="htk")
    parser.add_argument(
        '-c', '--context', type =int, help ='Context. Only utterances with more frames than this will be extracted.',  required =False, default=22)
    parser.add_argument(
        '-a', '--architecture', type =str, help ='Architecture, "tdnn" or "resnet". Affects input format.',  required =False, default="tdnn")
    parser.add_argument('--use_gpu', dest='use_gpu', default=False, action='store_true')
    parser.add_argument('--instance_norm', dest='instance_norm', default=False, action='store_true')
    args              = parser.parse_args()

    return(args.vad_scp, args.scp, args.out_dir+"/", args.model, args.window_size,
           args.shift, args.n_cores, args.extract, args.side_info, args.store_option,
           args.store_format, args.context, args.architecture, args.use_gpu, args.instance_norm)
        
if ( __name__ == "__main__" ):

    kaldi_src = "/mnt/matylda6/rohdin/software/kaldi_20190309/src/"
    
    # Input arguments
    [vad_scp, scp, output_dir, model, window, shift, n_cores, extract_var_name, side_info,
     store_option, store_format, context, arch, use_gpu, instance_norm] = get_configuration()

    
    # Check hostname and cpu info. Will be printed in log below. Cpu info just checks
    # the first cpu on the machine, not necessarily the one we use but normally they
    # are the same.
    host_name = os.uname()[1]
    cpu_info  = subprocess.check_output("cat /proc/cpuinfo | grep 'model name' | head -n 1 ", shell=True).decode('utf-8').split(':')[1]

    log = get_logger()
    log.info("host_name  " + host_name)
    log.info("cpu_info   " + cpu_info)
    log.info("vad_scp:   " + vad_scp)
    log.info("scp:       " + scp)
    log.info("out_dir:   " + output_dir)
    log.info("model:  " + model)    
    log.info("Window:    " + str(window))
    log.info("Shift:     " + str(shift))
    log.info("n_cores:   " + str(n_cores))
    log.info("variables to extract:   " + extract_var_name)
    log.info("side info:   " + side_info)
    log.info("store option:   " + store_option)
    log.info("store format:   " + store_format)
    log.info("context:   " + str(context) )
    log.info("architecture:  " + arch )
    log.info("use_gpu:  " + str(use_gpu) )
    log.info("instance_norm:  " + str(instance_norm) )
    
    log.info("Extracting embeddings")
    overlap = window - shift
    if ( vad_scp != "None" ):
        log.info("Will apply VAD to the features.")
        feat_rsp="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:"+scp+" ark:- | select-voiced-frames ark:- scp:"+vad_scp+" ark:- |"
        feats_generator=kaldi_io.read_mat_ark(feat_rsp)
    else:
        log.info("Assuming VAD has already been applied to the features.")
        feat_rsp="scp:"+scp 
        feats_generator=kaldi_io.read_mat_scp(feat_rsp)


    if use_gpu:
        # Detect which GPU to use
        command='nvidia-smi --query-gpu=memory.free,memory.total --format=csv |tail -n+2| awk \'BEGIN{FS=" "}{if ($1/$3 > 0.98) print NR-1}\''
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = subprocess.check_output(command, shell=True).decode('utf-8').rsplit('\n')[0]
            log.info("CUDA_VISIBLE_DEVICES " + os.environ["CUDA_VISIBLE_DEVICES"])
        except subprocess.CalledProcessError:
            log.info("No GPU seems to be available")        
        sess            = tf.Session()
        
    else:
        if (n_cores == -1):
            log.info("Using all available cores")
            sess = tf.Session()
        else:
            log.info("Using " + str(n_cores) + " cores" )
            session_conf=tf.ConfigProto(
                intra_op_parallelism_threads=n_cores,
                inter_op_parallelism_threads=n_cores)
            sess = tf.Session(config=session_conf)

        
    ### --- Set up the model ------------------------------------------ ###
    #saver = tf.train.import_meta_graph(re.sub('-\d+','-0.meta', model))
    model_dir = os.path.dirname( model )
    model_name = os.path.basename( model )
    saver = tf.train.import_meta_graph(model_dir + "/" + re.sub('-\d+','-0.meta', model_name))
    graph = tf.get_default_graph()

    saver.restore(sess, os.path.relpath(model))   # Due to a bug in TF, path must be relative. Seems to be fixed in later versions of TF.
    X1_p             = graph.get_tensor_by_name('X1_p:0')
    if ( arch == "tdnn"):
        C1_p             = graph.get_tensor_by_name('C1_p:0')
    if ( side_info != "None" ):
        side_info, side_info_value = side_info.split(":")
        S1_p             = graph.get_tensor_by_name(side_info + ':0')
        
    is_test_p        = graph.get_tensor_by_name('is_test_p:0')


    ### --- Extract the required embeddings ------------------------ ###
    if ( arch == "tdnn"):
        vars_to_extrct_  = [graph.get_tensor_by_name(v+':0') for v in extract_var_name.split(",") ]
        if ( side_info != "None" ):
            log.info("Adding %s to %s for all utterances " % (side_info_value, str(S1_p) ))
            g  = lambda X1, C1: sess.run(vars_to_extrct_, {X1_p: X1, C1_p:C1, S1_p:np.array([side_info_value]).astype(floatX).reshape(1,1),
                                                           is_test_p:True})
        else:
            g  = lambda X1, C1 : sess.run(vars_to_extrct_, {X1_p: X1, C1_p:C1, is_test_p:True})    
    elif( arch == "resnet"):
        vars_to_extrct_  = [graph.get_tensor_by_name('resnet_model/' + v + ':0') for v in extract_var_name.split(",") ]
        g  = lambda X1, C1: sess.run(vars_to_extrct_, {X1_p: X1[:,:,:,np.newaxis], is_test_p:True})
    else:
        raise ("Unsuported architcture %s" % arch)
        
    data_info = {"f_name":[], "f_path":[]}

        
    first = True   
    # We process the files one by one
    log.info("Extracting embeddings")
    for f_path,feats in feats_generator:

        if ( side_info == "None" ):
            log.info("Processing: " + f_path)
        else:
            log.info("Processing: " + f_path + ", sideinfo: " +  side_info +  ", sideinfo_value: " +  side_info_value )

        n_frames = len(feats)
        log.info("# feats: " + str(n_frames) )

        feats = feats[np.newaxis,:,:]
        if instance_norm:
            feats = apply_instance_norm(feats)

        try: 
            if ( n_frames > context ):

                idx    = np.arange(0, n_frames-overlap, shift)
                if ( n_frames - idx[-1] > context ): 
                     idx    = np.append(idx, n_frames-overlap)
                    
                out  = g(feats, idx)    

                if (store_option == 'separately'):
                    if (store_format == "htk"):
                        log.error("HTK format currently not supported.")
                        sys.exit(-1)
                        for o,v in zip(out, extract_var_name.split(",")):
                            pytel.htk.writehtk(output_dir + f_path + "." + v, o )

                    elif (store_format == "h5"):
                        if (first):
                            h5_output = []
                            h5_f_path = []
                            for o in out:
                                h5_output.append([o])
                            h5_f_path.append(f_path)
                            first = False
                        else:
                            for i,o in enumerate(out):
                                h5_output[i].append(o)
                            h5_f_path.append(f_path)

                elif (store_option == 'concat'):
                    if (store_format == "h5"):
                        log.error("h5 not supported for concatenated embeddings")
                    elif (store_format == "h5"):
                        log.error("HTK format currently not supported.")
                        sys.exit(-1)
                        out = np.hstack(out)
                        pytel.htk.writehtk(output_dir + f_path + "." + extract_var_name, out )
                        
                else:
                    log.error("ERROR: Wrong store option")

            else:
                log.info("File too short. Skipping it"  )
        except Exception as e:
            #log.warning("Failed: ", f_path)
            #log.warning(e.__doc__)
            #log.warning(e.message)
            raise type(e)(str(e) + ' for file ' + f_path )
    
        
    if (store_format == "h5"):
        h5_file = output_dir + "/" + os.path.basename(scp) + '.h5'
        log.info("Saving embeddings to " + h5_file)
        
        var_names = extract_var_name.split(",")
        with h5py.File(h5_file, 'w', 'core') as f:
            f.create_dataset('Physical', data=[a.encode('utf-8') for a in h5_f_path ], dtype=h5py.special_dtype(vlen=str) )
            if ( len(var_names) == 1 ):
                f.create_dataset('Data', data=np.squeeze(h5_output[0]) )
            else:
                for i in range(len(var_names)):
                    f.create_dataset('Data'+ var_names[i] , data=h5_output[i] )

