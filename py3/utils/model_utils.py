import h5py
import numpy as np

from utils.misc import get_logger
log = get_logger()

# Natural sort. From
# https://stackoverflow.com/questions/2545532/python-analog-of-natsort-function-sort-a-list-using-a-natural-order-algorithm
import re
def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    #return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
    return [int(s) if s.isdecimal() else s for s in re.split(r'(\d+)', string_)]

def load_model(load_model_file):
    para = []
    try:
        with h5py.File( load_model_file, 'r', driver='core') as f:

            for k in  sorted(list(f.keys()), key=natural_key) :
                p =k # k.encode('ascii') 
                m  =re.search('_(\d+)_(\d+)', p)
                i1 = int(m.group(1))
                print("loading " + p)            
                if i1 +1 > len(para):
                    para.append([])
                para[i1].append(np.array( f[p] ))
    except IOError:
        raise Exception("Failed to load model [%s] " %  load_model_file )

    return para


def save_model(file_name, params):
    print("Saving model to " + file_name)  
    try:
        with h5py.File(file_name, 'w', driver='core') as f:
            for i in range(0,len(params )):            
                for p in range(0, len( params[i] )):
                    name='para_' + str(i) +'_' + str(p)
                    f.create_dataset(name, data =params[i][p])                
    except IOError:
        raise Exception("Cannot open file [%s] for writing" % f_name)



# Loads a Kaldi nnet3 model in text format. This function was used for
# loading David's model and will not work for general models.
def load_davids_kaldi_model(model_file):
    with open (model_file) as f:
        weights         = []
        biases          = []
        batchnorm_means = []
        batchnorm_vars  = []

        r1 = re.compile('\<ComponentName\> (tdnn[1-9]*.affine) .*\[')
        r2 = re.compile('\<ComponentName\> (tdnn[1-9]*.batchnorm) .*\[')
        r3 = re.compile('\<ComponentName\> (output.affine) .*\[')

        l = f.readline()
        while l:
            m1 = r1.match( l )
            m3 = r3.match( l )
            if (m1 != None ) or (m3 != None ):
                if (m1 != None ):
                    m = m1
                else:
                    m=  m3

                print("Loading component " + m.group(1))            
                w = []
                while True:
                    l = f.readline()
                    a = l.rsplit(' ')
                    w.append(np.array(a[2:-1],dtype='float32'))

                    if ( a[-1] == ']\n' ):
                        weights.append(np.array(w))
                        print(" Weights loaded")
                        break

                l = f.readline()
                a = l.rsplit(' ')                
                biases.append(np.array(a[3:-1], dtype='float32'))
                print(" Bias loaded")

            l = f.readline()

            m2 = r2.match( l )
            if (m2 != None ):
                print("Loading batchnorm " + m.group(1))
                a = l.rsplit(' ')                
                batchnorm_means.append(np.array(a[18:-1], dtype='float32'))
                print(" Batchnorm mean loaded")

                l = f.readline()
                a = l.rsplit(' ')                
                batchnorm_vars.append(np.array(a[3:-1], dtype='float32'))
                print(" Batchnorm mean loaded")

    return [ weights, biases, batchnorm_means, batchnorm_vars ]

def load_davids_kaldi_model_2(model_file):
    with open (model_file) as f:
        weights         = []
        biases          = []
        batchnorm_means = []
        batchnorm_vars  = []

        r1 = re.compile('\<ComponentName\> (tdnn[1-9]*\w?.affine) .*\[')
        r2 = re.compile('\<ComponentName\> (tdnn[1-9]*\w?.batchnorm) .*\[')
        r3 = re.compile('\<ComponentName\> (output.affine) .*\[')

        l = f.readline()
        while l:
            m1 = r1.match( l )
            m3 = r3.match( l )
            if (m1 != None ) or (m3 != None ):
                if (m1 != None ):
                    m = m1
                else:
                    m=  m3

                print("Loading component " + m.group(1))            
                w = []
                while True:
                    l = f.readline()
                    a = l.rsplit(' ')
                    w.append(np.array(a[2:-1],dtype='float32'))

                    if ( a[-1] == ']\n' ):
                        weights.append(np.array(w))
                        print(" Weights loaded")
                        print(" Shape " + str(np.array(w).shape))
                        break

                l = f.readline()
                a = l.rsplit(' ')                
                biases.append(np.array(a[3:-1], dtype='float32'))
                print(" Bias loaded")

            l = f.readline()

            m2 = r2.match( l )
            if (m2 != None ):
                print("Loading batchnorm " + m.group(1))
                a = l.rsplit(' ')                
                batchnorm_means.append(np.array(a[18:-1], dtype='float32'))
                print(" Batchnorm mean loaded")

                l = f.readline()
                a = l.rsplit(' ')                
                batchnorm_vars.append(np.array(a[3:-1], dtype='float32'))
                print(" Batchnorm mean loaded")

    return [ weights, biases, batchnorm_means, batchnorm_vars ]

def load_kaldi_xvec_para(kaldi_txt_mdl, feat_dim=40, n_lay_before_pooling=12, n_lay_after_pooling=2,
                         feat_norm=True, pool_norm=True):

    mdl = load_davids_kaldi_model_2( kaldi_txt_mdl ) 
    
    para = [[],[]]
    if feat_norm:
        para[0].append( np.zeros([1,1,feat_dim]) )
        para[0].append( np.ones([1,1,feat_dim]) )
        para[0].append( np.zeros([1,1,feat_dim]) )
        para[0].append( np.ones([1,1,feat_dim]) )

    for i in range( n_lay_before_pooling ):

        para[0].append( mdl[0][i].T )
        para[0].append( mdl[1][i] )
        para[0].append( mdl[2][i][np.newaxis,np.newaxis,:] )
        para[0].append( mdl[3][i][np.newaxis,np.newaxis,:] )
        para[0].append( np.zeros_like(mdl[2][i].squeeze() ) )
        para[0].append( np.ones_like(mdl[3][i].squeeze()) )

    pool_size = mdl[1][n_lay_before_pooling-1].shape[0] * 2
    if pool_norm:
        para[0].append( np.zeros([1,1,pool_size]) )
        para[0].append( np.ones([1,1,pool_size]) )
        para[0].append( np.zeros([1,1,pool_size]) )
        para[0].append( np.ones([1,1,pool_size]) )

    for i in range( n_lay_after_pooling ):

        para[0].append( mdl[0][n_lay_before_pooling +i ].T )
        para[0].append( mdl[1][n_lay_before_pooling +i ] ) 
        para[0].append( mdl[2][n_lay_before_pooling +i ][np.newaxis,np.newaxis,:] )
        para[0].append( mdl[3][n_lay_before_pooling +i ][np.newaxis,np.newaxis,:] )
        para[0].append( np.zeros_like(mdl[2][n_lay_before_pooling +i ].squeeze() ) )
        para[0].append( np.ones_like(mdl[3][n_lay_before_pooling +i ].squeeze()) )

    para[0].append( mdl[0][n_lay_before_pooling + n_lay_after_pooling].T )
    para[0].append( mdl[1][n_lay_before_pooling + n_lay_after_pooling] )

    ### For comparing the loaded shapes with those of the model. Not tested.    
    # a = get_para()
    # for i in range(len(a[0])):
    #    if (a[0][i].shape != para[0][i].shape):
    #        #print str(i)  + " " + str(a[0][i].shape == para[0][i].shape)
    #        print str(i)  + " " + str(a[0][i].shape) +" " +  str(para[0][i].shape)
    
    ### Format used inside the x-vector model. Not tested.
    #tdnn_para_bp = {'b_' + str(i+1): mdl[1][i] for i in range(n_lay_before_pooling) }
    #tdnn_para_bp.update( {'W_' + str(i+1): mdl[0][i] for i in range(n_lay_before_pooling)} )

    #tdnn_para_ap = {'b_' + str(i+1): mdl[1][i+n_lay_before_pooling] for i in range(n_lay_after_pooling) }
    #tdnn_para_ap.update( {'W_' + str(i+1): mdl[0][i+n_lay_before_pooling] for i in range(n_lay_after_pooling)} )

    #bn_para_bp = {'m_' + str(i+1): mdl[2][i] for i in range(n_lay_before_pooling) }
    #bn_para_bp.update( {'v_' + str(i+1): mdl[3][i] for i in range(n_lay_before_pooling)} )

    #bn_para_bp = {'m_' + str(i+1): mdl[2][i+n_lay_before_pooling] for i in range(n_lay_after_pooling) }
    #bn_para_bp.update( {'v_' + str(i+1): mdl[3][i+n_lay_before_pooling] for i in range(n_lay_after_pooling)} )
        
    return para


def load_kaldi_xvec_para_est_stat(mdl, Stats_, it_tr_que, sess, set_para, X1_p, C1_p, is_test_p, feat_dim=40, n_lay_before_pooling=12, n_lay_after_pooling=2, feat_norm=True, pool_norm=True, S1_p=None):

    #mdl = load_davids_kaldi_model_2( kaldi_txt_mdl )
    para = [[],[]]
    if feat_norm:
        para[0].append( np.zeros([1,1,feat_dim]) )
        para[0].append( np.ones([1,1,feat_dim]) )
        para[0].append( np.zeros([1,1,feat_dim]) )
        para[0].append( np.ones([1,1,feat_dim]) )

    for i in range( n_lay_before_pooling ):

        para[0].append( mdl[0][i].T )
        para[0].append( mdl[1][i] )
        para[0].append( mdl[2][i][np.newaxis,np.newaxis,:] )
        para[0].append( mdl[3][i][np.newaxis,np.newaxis,:] )
        para[0].append( np.zeros_like(mdl[2][i].squeeze() ) )
        para[0].append( np.ones_like(mdl[3][i].squeeze()) )

    if pool_norm:
        pool_size = mdl[1][n_lay_before_pooling-1].shape[0] * 2
        para[0].append( np.zeros([1,1,pool_size]) )
        pool_mean_idx = len(para[0]) -1 
        para[0].append( np.ones([1,1,pool_size]) )
        pool_var_idx = len(para[0])  -1
        para[0].append( np.zeros([1,1,pool_size]) )
        para[0].append( np.ones([1,1,pool_size]) )

    first_a_pool_w_idx = len(para[0])
    first_a_pool_b_idx = len(para[0]) + 1 
    for i in range( n_lay_after_pooling ):

        para[0].append( mdl[0][n_lay_before_pooling +i ].T )
        para[0].append( mdl[1][n_lay_before_pooling +i ] ) 
        para[0].append( mdl[2][n_lay_before_pooling +i ][np.newaxis,np.newaxis,:] )
        para[0].append( mdl[3][n_lay_before_pooling +i ][np.newaxis,np.newaxis,:] )
        para[0].append( np.zeros_like(mdl[2][n_lay_before_pooling +i ].squeeze() ) )
        para[0].append( np.ones_like(mdl[3][n_lay_before_pooling +i ].squeeze()) )

    para[0].append( mdl[0][n_lay_before_pooling + n_lay_after_pooling].T )
    para[0].append( mdl[1][n_lay_before_pooling + n_lay_after_pooling] )

    set_para(para)
    if pool_norm:
        log.info("Estimating statistics of pool output")
        #g_stat      = lambda X1: sess.run(Stats_, {X1_p: X1, C1_p: np.array([]), is_test_p:False})
        ss          = np.zeros([0,pool_size])


        log.info("Calculating mean and standard deviation of pooling output")
        for i in range(10):
            #[X, Y, U], _, [feats, tr_idx ], it_batch_nb, it_ctrl_nb = it_tr_que.get_batch()
            #ss    = np.concatenate([ss,g_stat(feats).squeeze()], axis=0)
            I, _, [feats, tr_idx ], it_batch_nb, it_ctrl_nb = it_tr_que.get_batch()
            if ( len(I) == 3 ):
                [X, Y, U] = I
                g_stat = lambda X1: sess.run(Stats_, {X1_p: X1, C1_p: np.array([]), is_test_p:False})
                ss     = np.concatenate([ss,g_stat(feats).squeeze()], axis=0)
            elif ( len(I) == 4 ):
                [X, Y, U, S] = I
                S=np.vstack([int(s) for s in S] )
                g_stat = lambda X1,S1: sess.run(Stats_, {X1_p: X1, C1_p: np.array([]), is_test_p:False, S1_p:S1})
                ss     = np.concatenate([ss,g_stat(feats, S).squeeze()], axis=0)
                
        ### Apply the normalization as a batchnorm layer
        mean_pool = np.mean(ss,axis=0)[np.newaxis,np.newaxis,:]
        para[0][pool_mean_idx] = mean_pool
        var_pool  = np.var(ss,axis=0)[np.newaxis,np.newaxis,:]
        para[0][pool_var_idx] = var_pool
        std_pool  = np.std(ss,axis=0)[np.newaxis,np.newaxis,:]

        para[0][first_a_pool_b_idx] += np.dot(mean_pool.squeeze(), para[0][first_a_pool_w_idx] )
        para[0][first_a_pool_w_idx]  = para[0][first_a_pool_w_idx] * std_pool.squeeze(0).T

        set_para(para)
    return para


    
def load_plda_model( plda_dir ):
    if ( os.path.isfile(plda_dir + '/params/cov_mat_mean.mat') ):
        log.info("PLDA model file name is cov_mat_mean.mat, assuming this is a MATLAB generative model")
        gplda_model = scipy.io.loadmat(plda_dir +  '/params/cov_mat_mean.mat')
        m = gplda_model['model']['mu'][0][0]
        V = gplda_model['model']['V'][0][0]
        U = gplda_model['model']['U'][0][0]
        D = gplda_model['model']['D'][0][0]

        B = V.dot(V.T)
        W = U.dot(U.T) + np.diag( 1 / D.ravel(),)

        return [B, W, m]

    elif( os.path.isfile( plda_dir + '/model_pytel_gplda.h5') ):
        log.info("PLDA model file name is model_pytel_gplda.h5, assuming this is a Pytel generative model")
        model_pytel_gplda = plda_dir + '/model_pytel_gplda.h5' 
        try:
            with h5py.File( model_pytel_gplda, 'r', driver='core') as f:
                B = np.array( f['/B'] )
                W = np.array( f['/W'] )
                m = np.array( f['/m'] )

                return [m, B, W]

        except IOError:
            raise Exception( "Cannot open data file [%s] for reading" % model_pytel_gplda )

    elif( os.path.isfile(plda_dir + '/model_pytel_dplda.h5') ):
        log.info("PLDA model file name is model_pytel_dplda.h5, assuming this is a Pytel discriminative model")
        model_pytel_dplda = plda_dir + '/model_pytel_dplda.h5' 
        try:
            with h5py.File( model_pytel_dplda, 'r', driver='core') as f:
                if '/Lambda' in f:
                    PP = np.array( f['/Lambda'] )
                else:
                    PP = np.array( f['/P'] )
                if '/Gamma' in f:
                    QQ = np.array( f['/Gamma'] )
                else:
                    QQ = np.array( f['/Q'] )

                cc = np.array( f['/c'] )
                kk = np.array( f['/k'] )

                return [PP.astype( floatX ), QQ.astype( floatX ), cc.astype( floatX ), kk.astype( floatX )]

        except IOError:
            raise Exception("Cannot open data file [%s] for reading" % model_pytel_dplda)

    else:
        log.error("No PLDA model found.")
        return []


def save_gplda_model(B,W,m, gplda_model_file):
    try:
        with h5py.File(gplda_model_file, 'w', driver='core') as f:
            f.create_dataset('B', data=B)
            f.create_dataset('W', data=W)
            f.create_dataset('m', data=m)
    except IOError:
        raise Exception("Cannot open file [%s] for writing" % f_name)


def get_n_para(var_list):
    n = 0
    for v in var_list:
        n_tmp =1
        for i in range(len(v.shape) ):
            n_tmp *= v.shape[i].value
        n += n_tmp    
    return n


def find_best_model(load_model_prefix):
    avl_models = glob.glob(load_model_prefix+'_epoch-[0-9]*.*.h5')
    next_epoch = len(avl_models)
    if next_epoch == 0:
      log.warning( "WARNING: You requested to load an existing model but no model was found. Training will start from Epoch 0." )
      load_old_model = False
    else:

        # Find the best model by looking at the dev loss.
        prog = re.compile('.*/model_feat2score_epoch-([0-9]*)_lr-([-\.0-9e]*)_lossTr-([-\.0-9ex]*)_lossDev-([-\.0-9e]*).h5')
        prev_lr      = np.ones((next_epoch, 1))*np.NaN   # *** Not all nans will be overwritten below 
        prev_lossTr  = np.ones((next_epoch, 1))*np.NaN
        prev_lossDev = np.ones((next_epoch, 1))*np.NaN
        for m in avl_models:
            match                   = prog.match(m)
            epoch_tmp               = int(match.group(1))
            prev_lr[epoch_tmp]      = match.group(2)    # *** if same epoch number used for different models.
            print(epoch_tmp)
            if (epoch_tmp > 0):
                print(epoch_tmp)
                prev_lossTr[epoch_tmp]  = match.group(3)
            prev_lossDev[epoch_tmp] = match.group(4)

        best_epoch = np.argmin(prev_lossDev).squeeze()
        if ( best_epoch == next_epoch -1 ):
            lr_first = prev_lr[best_epoch]
        else:
            lr_first = prev_lr[next_epoch -1] / 2

        if ( next_epoch  > 0) :
            if (best_epoch > 0):    
                load_model_file = load_model_prefix+ '_epoch-' + str(best_epoch) + '_lr-' + str(prev_lr[best_epoch][0]) + '_lossTr-' + str(prev_lossTr[best_epoch][0]) + '_lossDev-' + str(prev_lossDev[best_epoch][0]) + '.h5'
                log.info( "%d models found. Loading model %s and starting from next epoch." %(next_epoch, load_model_file) )
            else:
                log.info( "%d models found. But best model is the initial model. Will not load any model")
                load_old_model = False

    return load_old_model, lr_first, load_model_file, best_epoch # Should it be best_epoch + 1 ???


def h5_to_kaldi_nnet3(model_file_h5_in, model_file_kaldi_nnet3_out, tdnn_sizes_before_pooling, tdnn_sizes_after_pooling,
                      activations_before_pooling, activations_after_pooling, do_feat_norm, do_pool_norm):
    para = load_model( model_file_h5_in)
    header = "<Nnet3>\n" + \
    "input-node name=input dim=23\n" + \
    "component-node name=tdnn1.affine component=tdnn1.affine input=Append(Offset(input, -2), Offset(input, -1), input, Offset(input, 1), Offset(input, 2))\n" + \
    "component-node name=tdnn1.relu component=tdnn1.relu input=tdnn1.affine\n" + \
    "component-node name=tdnn1.batchnorm component=tdnn1.batchnorm input=tdnn1.relu\n" + \
    "component-node name=tdnn2.affine component=tdnn2.affine input=Append(Offset(tdnn1.batchnorm, -2), tdnn1.batchnorm, Offset(tdnn1.batchnorm, 2))\n" + \
    "component-node name=tdnn2.relu component=tdnn2.relu input=tdnn2.affine\n" + \
    "component-node name=tdnn2.batchnorm component=tdnn2.batchnorm input=tdnn2.relu\n" + \
    "component-node name=tdnn3.affine component=tdnn3.affine input=Append(Offset(tdnn2.batchnorm, -3), tdnn2.batchnorm, Offset(tdnn2.batchnorm, 3))\n" + \
    "component-node name=tdnn3.relu component=tdnn3.relu input=tdnn3.affine\n" + \
    "component-node name=tdnn3.batchnorm component=tdnn3.batchnorm input=tdnn3.relu\n" + \
    "component-node name=tdnn4.affine component=tdnn4.affine input=tdnn3.batchnorm\n" + \
    "component-node name=tdnn4.relu component=tdnn4.relu input=tdnn4.affine\n" + \
    "component-node name=tdnn4.batchnorm component=tdnn4.batchnorm input=tdnn4.relu\n" + \
    "component-node name=tdnn5.affine component=tdnn5.affine input=tdnn4.batchnorm\n" + \
    "component-node name=tdnn5.relu component=tdnn5.relu input=tdnn5.affine\n" + \
    "component-node name=tdnn5.batchnorm component=tdnn5.batchnorm input=tdnn5.relu\n" + \
    "component-node name=stats-extraction-0-10000 component=stats-extraction-0-10000 input=tdnn5.batchnorm\n" + \
    "component-node name=stats-pooling-0-10000 component=stats-pooling-0-10000 input=stats-extraction-0-10000\n" + \
    "component-node name=tdnn6.affine component=tdnn6.affine input=Round(stats-pooling-0-10000, 1)\n" + \
    "component-node name=tdnn6.relu component=tdnn6.relu input=tdnn6.affine\n" + \
    "component-node name=tdnn6.batchnorm component=tdnn6.batchnorm input=tdnn6.relu\n" + \
    "component-node name=tdnn7.affine component=tdnn7.affine input=tdnn6.batchnorm\n" + \
    "component-node name=tdnn7.relu component=tdnn7.relu input=tdnn7.affine\n" + \
    "component-node name=tdnn7.batchnorm component=tdnn7.batchnorm input=tdnn7.relu\n" + \
    "component-node name=output.affine component=output.affine input=tdnn7.batchnorm\n" + \
    "component-node name=output.log-softmax component=output.log-softmax input=output.affine\n" + \
    "output-node name=output input=output.log-softmax objective=linear\n"
    print(header)

    c_num = 0

    print ("<NumComponents> 25")
    
    if feat_norm:
        #para[0].append( np.zeros([1,1,feat_dim]) )
        #para[0].append( np.ones([1,1,feat_dim]) )
        #para[0].append( np.zeros([1,1,feat_dim]) )
        #para[0].append( np.ones([1,1,feat_dim]) )
        offset = 2
    else:
        offset = 0

    for i in range( n_lay_before_pooling ):


        print ("<ComponentName> tdnn%d.affine <NaturalGradientAffineComponent> <MaxChange> 0.75 <LearningRate> 0.0008 <LinearParams>  [" % i)
        weight = para[ i*4 + offset ]
        for r in range(weight.shape[0]-1):
            print (weight[r,:])
        print (weight[r,-1], end = " ]")

        bias = para[ i*4 +1  + offset ]
        print ( "<BiasParams>  [", end="" )
        print (bias, end= " ]" )

        
        para[0].append( mdl[0][i].T )
        para[0].append( mdl[1][i] )
        para[0].append( mdl[2][i][np.newaxis,np.newaxis,:] )
        para[0].append( mdl[3][i][np.newaxis,np.newaxis,:] )
        para[0].append( np.zeros_like(mdl[2][i].squeeze() ) )
        para[0].append( np.ones_like(mdl[3][i].squeeze()) )

        

    pool_size = mdl[1][n_lay_before_pooling-1].shape[0] * 2
    if pool_norm:
        para[0].append( np.zeros([1,1,pool_size]) )
        para[0].append( np.ones([1,1,pool_size]) )
        para[0].append( np.zeros([1,1,pool_size]) )
        para[0].append( np.ones([1,1,pool_size]) )

    for i in range( n_lay_after_pooling ):

        para[0].append( mdl[0][n_lay_before_pooling +i ].T )
        para[0].append( mdl[1][n_lay_before_pooling +i ] ) 
        para[0].append( mdl[2][n_lay_before_pooling +i ][np.newaxis,np.newaxis,:] )
        para[0].append( mdl[3][n_lay_before_pooling +i ][np.newaxis,np.newaxis,:] )
        para[0].append( np.zeros_like(mdl[2][n_lay_before_pooling +i ].squeeze() ) )
        para[0].append( np.ones_like(mdl[3][n_lay_before_pooling +i ].squeeze()) )

    para[0].append( mdl[0][n_lay_before_pooling + n_lay_after_pooling].T )
    para[0].append( mdl[1][n_lay_before_pooling + n_lay_after_pooling] )
    
    
    return para

