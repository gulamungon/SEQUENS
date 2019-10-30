# Translated from the theano version 2017-12-14. At this date
# there were only minor differences from the Theano version, e.g.,
# how input arguments are provided. Later on consistency between
# the Theano and Tensorflow version may not be kept.

# Notes
# 1: Probably alpha should better be included in self.S_reg since
#    otherwise effect of regularization will probably dissappear
#    as explained in note for score regularization where I included it.


import h5py, logging

import tensorflow as tf


import numpy as np
from numpy.linalg import inv, slogdet



# Notations:
# 
# Assume the (two-covariance) PLDA model:
# m = u + y + x
#
# y ~ N(0, B)
# x ~ N(0, W)
#
# m: i-vector                             (d x 1) 
# W: within-class covariance matrix       (d x d)
# B: between-class covariance matrix      (d x d)
# u: mean                                 (d x 1)
#
# PLDA LLR-score:
# s_ij  =  m_i' * PP * m_j  +  m_j' * PP * m_i  
#       +  m_i' * Q  * m_i  +  m_j' * Q  * m_j  
#       +  (m_i + m_j)'  *  c   +   k
#
# where
#  
# PP = 0.5 * invT * B * invS 
# Q  = 0.5 * (invT - invS)
# c  =  -2 * (PP + Q) * m
# k  = 0.5 * ( logDet(T) - logDet(S) ) - m.T * c        # Note:  - m.T * c = 2 m.T * (PP + Q) * m
#
# T  = B + W                  # Total covariance 
#
#                                                   [T B]
# S  = T - B * invT * B       # Schur complement of [B T]  (...I think)
#
#  NOTE: In the code, i-vectors will be collected in a matrix 
#  M of dimensions (n x d), not (d x n) for consistency with how 
#  most neural network tools seems to organize the data.
#
# NOTES ON CONSISTENCY WITH PYTEL'S DPLDA
# In order to obtain equivalent loss to Pytel's DPLDA for the same 
# parameters and regularization, we need to pay attention to the 
# following:
#  
# - Since PP is symmetric, we can rewrite
#    m_i' * PP * m_j  +  m_j' * PP * m_i  = 2 * m_i' * PP * m_j 
#                                         = m_i' * P * m_j
#    where, P = 2 * PP. 
#   THE CHOICE OF PAREMETERIZATION AFFECT THE RESULT WHEN REGULARIZATION 
#   IS USED!!! --- If a factor 2 is included in P, its regularization 
#   penalty will be 4 times larger. Anyway, we allow the possibility 
#   to use different regularization penalty for P, Q, c, k so it doesn't
#   matter much. But, for consistency with Pytel, we include the factor 2,  
#   in the model parameter i.e. we use "P" instead of "PP" as parameter.
#   Note: In my (Johan's) old work I did not do this and I used the same 
#   regularization parameter for P, Q, c, k 
# 
# - Again, since P and Q are symmetric we can think of e.g., P_ij and 
#   P_ji as the same parameter. Accordingly, they should be added to the 
#   regularization only once instead of twice as it would be with the 
#   above. To adjust for this, we multiply P and Q with a factor 0.25 
#   instead of 0.5 and add the diagonal times 0.25. This could probably
#   be done in a better way. E.g., using T.triu() = TODO.
#   Note: I did not do this in my old work.
#
# - In Pytel, as well as in my old work, we exclude the trials where
#   where the same i-vector is scored against itself corresponding to 
#   the diagonal in the score matrix. However when making the trial 
#   weight matrix, it seems Pytel includes these trials in the target
#   trial counts. This is not a big deal since anyway, the balance 
#   between target and non-target trials in the loss is tuned via 
#   P_eff. This parameter also affects the decision threshold, tau, 
#   so there will be small incosistency. However, this incosistency 
#   can be completely compensated by the training of k, at least when
#   no regularization is used.
#   In the function "labMat2weight" we include and option "increase_n_tar"
#   which is False by default but if set to True will replicate Pytel
#   Update: This has now been changed in Pytel so increase_n_tar=False
#   should give identical results to Pytel.
#
# - Pytel normalizes the loss to be the CLLR. We let this is the default 
#   in the weight calculation here too. 
#   Note: I didn't do this in my old work.


logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.DEBUG)
log = logging.getLogger(__name__)

def lab_2_ntar_nnon( lab_vector, increase_n_tar = False ):

    counts  = np.bincount( lab_vector )
    counts2 = counts **2

    tot   = len( lab_vector ) **2
    n_tar = np.sum( counts2 )
    n_non = tot - n_tar

    # Now we don't want to double count trials
    n_non = n_non/2
    
    if ( increase_n_tar ):
        # If diagonal trials are counted.
        n_tar   = (n_tar + np.sum(counts))/2
    else:
        # If diagonal trials are not counted.
        n_tar   = (n_tar - np.sum(counts))/2   

    return [ n_tar, n_non ]

def labMat2weight_external_weights(Y, tar_weight, non_tar_weight):
    W = (Y == 1)*tar_weight + (Y == -1)*non_tar_weight
    return W

# Converts PLDA parameters, m, B and W, to "DPLDA" parameters,
# P, Q, c, k. Can be used for, e.g, initialize DPLDA training.
def mBW_2_PQck(m, B, W):

    T = B + W                    

    invT = inv(T)
                                 
    S    = T - np.dot(B, np.dot( invT,  B ))
    invS = inv(S)

    #P = 0.5 * np.dot(invT, np.dot(B,  invS ))
    P = np.dot(invT, np.dot(B,  invS ))
    P = np.dot(invT, np.dot(B,  invS ))
    Q = 0.5 * (invT - invS)
    #c =  -2 * np.dot((P + Q), m )
    c =  - np.dot((P + 2*Q), m )

    [s, logdet_T ] = slogdet(T)
    assert s > 0                      # This should be the case since the matrix should be positive definite
    [s, logdet_S ] = slogdet(S)
    #print s
    assert s > 0               

    k = 0.5 * ( logdet_T - logdet_S ) - np.dot(m.T, c)

    return [P, Q, c, k]

# Convert the Label vector to a label matrix of size (n x n)
# with elements equal to 1 for target trials and -1 for non-
# target trials. Written in a way that avoids "repeat" so that
# this code can be translated to Theano code without getting 
# gradient problems.However, the comparison (Y == Y.T) probably
# doesn't work on the GPU??
def lab2matrix(lab):
    n = lab.shape[0]
    I_1xn = np.ones((1, n))
    Y = np.dot(lab[:,None], I_1xn )
    Y = 2.0*(Y == Y.T) - np.ones((n, n))
    return Y

 
def lab2matrix_2(lab1, lab2):
    n1 = lab1.shape[0]
    n2 = lab2.shape[0]
    I_1xn1 = np.ones((1, n1))
    I_1xn2 = np.ones((1, n2))
    Y1 = np.dot(lab1[:,None], I_1xn2 )
    Y2 = np.dot(lab2[:,None], I_1xn1 )
    Y = 2.0*(Y1 == Y2.T) - np.ones((n1, n2))
    return Y



# Make a weight matrix from a label matrix.
# The weights are set so that the total weight
# for the target trials euqals p_eff and the 
# total weight for the non-target trials equals
# (1 - p_eff). The diagonal, i.e., i-vectors
# scored against itself get weight 0.
def labMat2weight(Y, p_eff=0.5, cllr_norm=True, include_diag_trials=False, increase_n_tar=False):
    n         = Y.shape[0]
 
    n_tar     = np.sum(Y == 1)          # Number of target trials, will be divided by 2 below 
    n_non     = (n * n - n_tar)/2       # Number of non-target trials 
    
    if (include_diag_trials) or ( increase_n_tar ):
        n_tar = (n_tar +n)/2 
    else:
        n_tar = (n_tar -n)/2 

    w  = p_eff*(Y == 1)/n_tar + (1- p_eff)*(Y == -1)/n_non  
 
    # The off-diagonal trials should be halved since element ij and ji in 
    # the score matrix correspond to the same trial.
    w = 0.5 * np.triu(w) + 0.5 * np.tril(w)

    if (not include_diag_trials):
        np.fill_diagonal(w,0)

    if (cllr_norm):
        cllr_norm = -( p_eff * np.log(p_eff) + (1-p_eff) * np.log(1-p_eff) )
        w = w / cllr_norm 

    return w


# Calculates the effective prior, p_eff.
def p_eff(C_FA, C_FR, P_tar): 
    assert(0.0 < P_tar < 1.0)     # I guess  "<="/">=" would be ok but for P_tar/the costs
    assert(C_FA > 0)              # in principle. But it would not be very meaningful and
    assert(C_FR > 0)              # would lead to infitly high or low LLR threshold.
    return P_tar * C_FR / (P_tar * C_FR + (1 - P_tar) * C_FA)

# Calculates llr threshold for a given p_eff
def llrThreshold(p_eff):
    assert(0.0 <= p_eff <= 1.0)
    return -np.log(p_eff/(1-p_eff))


# This function takes a list of counts (with the same lenght as the number of speakers).
# It returns
# counts_u: unique counts, e.g. [1, 3] meaning that some speakers has 1 session and some
#           have 3 sessions.
# idx     : array e.g., array([  0, 602, 802]) saying that first 602 speakers have 1 enroll
#           session and the following 200 has three in this case. (After sorted according to
#           mapping see below.
# mapping : A "mapping" such that ivec_new =ivec[mapping] and spk_new =[ spk[mp] for mp in  mapping ]
#           gives the i-vectors and spk (names) sorted so that all with 1 session comes first and so on.
def get_multi_enroll_idx( counts ):
    counts_u  = np.unique( counts )
    idx       = [0]
    mapping  = np.array([])
    for c in counts_u:
        m = np.where(counts == c)[0]
        mapping = np.concatenate( (mapping, m) )
        idx.append( len( m ) )
    idx = np.cumsum( idx )
    return counts_u, idx, mapping.astype(int)

 
class dplda_model(object):

    # Also, the input TF tensor, M (or placeholder), should be given.
    # --- Making two input ,M1 M2
    # M:  Either a TF tensor or a list containing
    #     one or two tensors. This will be the "enroll"
    #     and "test" data respectively. If only one is provided
    #     we will use it as both "enroll" and "test" data. 
    #def __init__(self, m, W, B, M, score_offset=None, score_scaling=None):
    # def __init__(self, m, W, B=None, M=None, score_offset=None, score_scaling=None):
    def __init__(self, tf_session, param, M, score_offset=None, score_scaling=None, floatX='float32', is_test_p=None):

        self.session = tf_session
        self.floatX  = floatX

        if ( is_test_p != None ):
            self.is_test_p = is_test_p
        else:
            self.is_test_p = None
            
        # Originally, the input
        if (len(param) == 3):
            [P, Q, c, k] = mBW_2_PQck(param[0], param[1], param[2])
        elif(len(param) == 4):
            [P, Q, c, k] = param
        else:
            log.error("ERROR: Number of parameters in DPLDA initialization should be 3 or 4 collected in a list.")

        if isinstance(P, tf.Variable):    
            self.P = P
            self.Q = Q
            self.c = c
            self.k = k
        else:
            self.P = tf.Variable( P.astype( self.floatX ), name='dplda_P' )
            self.Q = tf.Variable( Q.astype( self.floatX ), name='dplda_Q' )
            self.c = tf.Variable( c.astype( self.floatX ), name='dplda_c' )
            self.k = tf.Variable( k.astype( self.floatX ), name='dplda_k' )

            
        # Input data (M1="enroll", M2="test")
        if (not isinstance(M, list)):  
            M = [M]                                     
        if (len(M) == 1 ):
            self.M1 = M[0]    
            self.M2 = M[0]
        else:
            self.M1 = M[0]
            self.M2 = M[1]
 
        self.score_function = None   

        # If piggyback_S is provided (a Theano matrix), we will use Niko's piggyback recipe 
        # with this matrix as the baseline scores.
        # We also added a "boarback" which will be multiplied with the model scores
        self.score_offset  = score_offset
        self.score_scaling = score_scaling

        ### --- LLR score ---------------------- ###
        # These are the formulas from e.g. "Pairwise
        # Discriminative Speaker Verification in the
        # I-vector space", IEEE 2013, Cumani et al.
        # (with multiplication by 1_1xn at some place
        # in order to do repetition, avoiding Theano's
        # "repeat" which seems to have some gradient
        # problems)

        I_1xn1 = tf.transpose( tf.ones_like( self.M1[:,0:1] ) ) # "0:1" is just so that shape becomes (?,1) instead of (?,)
        I_1xn2 = tf.transpose( tf.ones_like( self.M2[:,0:1] ) )
        
        # In the "normal" training where M1 = M2, P and Q will
        # remain symmetric during training but not sure if this
        # will happen if M1 and M2 are different sets.
        P_sym  = 0.5*(self.P + tf.transpose( self.P ) ) 
        Q_sym  = 0.5*(self.Q + tf.transpose( self.Q ) ) 
        
        # This should also work (better??) but the regularization
        # terms in the loss below need to be adjusted accordingly.
        #P_sym  = T.tril(self.P) + T.tril(self.P.T, k=1).T 
        #Q_sym  = T.tril(self.Q) + T.tril(self.Q.T, k=1).T 

        ##SP  = tf.tensordot(self.M1, tf.tensordot(P_sym, tf.transpose(self.M2) ))     # M1 * P * M2'
        SP  = tf.matmul( self.M1, tf.matmul(P_sym, self.M2,transpose_b=True) )     # M1 * P * M2'

        #SQ1_1 = T.dot(self.M1, T.dot(Q_sym, self.M1.T))                 # M1 * Q * M1'                 (n1 x n1)       
        #SQ2_1 = T.dot(T.diagonal(SQ1_1)[:,None], I_1xn2 )               # (diag(M1*Q*M1.T)) * I_1xn    (n1 x 1 ) * (1 x n2)
        #SQ1_2 = T.dot(self.M2, T.dot(Q_sym, self.M2.T))                 # M2 * Q * M2'                 (n2 x n2)       
        #SQ2_2 = T.dot(T.diagonal(SQ1_2)[:,None], I_1xn1)                # (diag(M2*Q*M2.T)) * I_1xn    (n2 x 1 ) * (1 x n1)
        
        # This is equivalent to the above but avoids some unecessary calculations. 
        SQ2_1 = tf.matmul( tf.reduce_sum(((tf.matmul(self.M1, Q_sym)) * self.M1 ), axis=1 )[:, None], I_1xn2 )
        SQ2_2 = tf.matmul( tf.reduce_sum(((tf.matmul(self.M2, Q_sym)) * self.M2 ), axis=1 )[:, None], I_1xn1 ) 

        SQ    = SQ2_1 + tf.transpose( SQ2_2 )

        Sc_1 = tf.matmul(self.M1, tf.matmul(self.c[:, None], I_1xn2))   # M*c   repeated n2 times      (n1 x n2) 
        Sc_2 = tf.matmul(self.M2, tf.matmul(self.c[:, None], I_1xn1))   # M*c   repeated n1 times      (n1 x n2) 
        Sc   = Sc_1 + tf.transpose( Sc_2 )

        Sk  = self.k


        # The final score from the model. This score will be regularized in case of 
        # score regularization so weep it in "S_reg"
        self.S     = SP + SQ + Sc + Sk
        self.S_reg = self.S 
        # Final score
        if (self.score_offset != None):
            
            # Include score_offsets and scaling if wanted. For now, we do not allow 
            # the case where to do scaling without offset
            if (self.score_scaling != None):
                self.S = self.S * self.score_scaling 

            # Weights for new model. Not sure if 0.5 is the best inital value.
            self.alpha = tf.Variable( np.array(0.5, dtype=floatX), name='dplda_alpha' )  
            self.beta  = tf.Variable( np.array(0.5, dtype=floatX), name='dplda_beta' )    

            self.S = self.alpha * self.S  + self.beta * self.score_offset
            

    def get_parameters(self):
        if (self.score_offset == None):
            return [self.P, self.Q, self.c, self.k ]
        else:
            return [self.P, self.Q, self.c, self.k, self.alpha, self.beta ]

    def get_parameter_values(self):
        params = []
        with self.session.as_default(): 
          for p in self.get_parameters():
            params += [ p.eval() ]
        return params

    def set_parameter_values(self, params):
        #with self.session.as_default():
        self.session.run( tf.assign( self.P, params[0] ) )
        self.session.run( tf.assign( self.Q, params[1] ) )
        self.session.run( tf.assign( self.c, params[2] ) )
        self.session.run( tf.assign( self.k, params[3] ) )

        if (self.score_offset != None):
            if (len(params) == 4 ):
                print("WARNING: score_offset is used but I was asked to set only 4 params ---alpha and beta will not be set")
            else:
                self.session.run( tf.assign( self.alpha, params[4] ) )
                self.session.run( tf.assign( self.beta,  params[5] ) )
                      
    def load(self, model_file):
        try:
            with h5py.File(model_file, 'r', driver='core') as f:
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

                if '/alpha' in f:
                    alpha = np.array( f['/alpha'] )
                    beta = np.array( f['/beta'] )
                else:
                    aa = None
                    bb  = None 
                

        except IOError:
            raise Exception("Cannot open data file [%s] for reading" % model_pytel_gplda)
        if (aa == None):
            self.set_parameter_values([PP.astype(floatX), QQ.astype(floatX), cc.astype(floatX), kk.astype(floatX).squeeze()])
        else:
            self.set_parameter_values([PP.astype(floatX), QQ.astype(floatX), cc.astype(floatX), kk.astype(floatX).squeeze(), 
                                       aa.astype(floatX).squeeze(), bb.astype(floatX).squeeze()])



    # Makes a score function if it doesn't already
    # exists. Then calculates the PLDA score.   
    # It is possible to specify an external TF variable, X,
    # such that the i-vectors, M,  are given by M = f(X). That is, 
    # X could be some other inputs, e.g., stats, feats, and "f"
    # is a neural netork that produces i-vectors. The score function
    # will then let its input be transformed by f before calculating
    # scores. In order for this to work, it is necessary that
    # M, given above, can be calculated from X.
    # That is, M = f( X ) have been specified somewhere else and M
    # is given as input when the model is initialized, i.e, 
    # model = dplda_model(m, W, B, M).
    # THIS FUNCTION SHOULD BE CLEANED UP. FOR THEANO, THIS WAS NECASSARY
    # TO COMPILE THE FUCNTION ONLY ONCE BUT TF CACHES SESSIONS I THINK
    # SO THIS MESS NOT NECESSARY HERE.
    def score(self, MX, X=[], is_test=True):

        if (self.score_function == None):
            if (X == []):
                # Perhaps add some check that M is a placeholder and not a standard tensor in this case.
                #self.score_function = theano.function([self.M1, self.M2 ], self.S, allow_input_downcast=True)
                #self.score_function = lambda M1, M2: self.session.run( self.S , {self.M1: M1, self.M2: M2 })
                self.score_function = lambda M1, M2: self.session.run( self.S , {self.M1: M1, self.M2: M2, self.is_test_p:is_test})
            else:
                # Perhaps add check the M is a standard tensor and not a placeholder.
                if (not isinstance(X, list)):  
                    X = [X]                                     
                #self.score_function = theano.function( X, self.S, allow_input_downcast=True)
                #self.score_function = lambda X1, X2: self.session.run( self.S , {X[0]: X1, X[1]: X2 })
                self.score_function = lambda X1, X2: self.session.run( self.S , {X[0]: X1, X[1]: X2, self.is_test_p:is_test})
        

        # Input data (M1="enroll", M2="test")
        if (not isinstance(MX, list)):  
            MX = [MX]
        return self.score_function( *MX )

  
    # Makes a loss "function". 
    # Inputs:
    # loss_fcn  loss function                   Should be increasing, i.e., punish large values. 
    #                                           "softlplus" will give the logistic regression loss. 
    # W         Trial weights                   (n x n)
    # Y         Trial labels                    (n x n)
    # tau       Threshold to remove from scores (scalar)
    def make_loss(self, loss_fcn, W, Y, tau, l2_reg_P, l2_reg_Q, l2_reg_c, l2_reg_k, reg_to=[], l2_reg_score=0.0, sep_reg=False, batch_score_norm=False):
    #def make_loss(self, loss_fcn, W, Y, tau):

        # Need to think about whether formulating
        # it like this works for any choice of loss
        # (It means target and non-target trials 
        # are treated symmetrically around tau)
        if batch_score_norm:
            p_eff    = 0.0075
            
            # tar_mask =  (Y + 1) /2   
            # n_tar = tf.reduce_sum( tar_mask ) - tf.cast(tf.shape(Y)[0], self.floatX)    # Subtact the diagonal
            # This is a bug because ones are still on the diagonal of tar_mask and will be included in some calculations
            
            tar_mask =  (Y + 1) /2     - tf.eye(tf.shape(Y)[0])    # Subtact the diagonal
            n_tar = tf.reduce_sum( tar_mask ) 
            
            non_mask = -(Y - 1) /2
            n_non = tf.reduce_sum( non_mask )

            m_e = ( tf.reduce_sum( tar_mask * self.S ) ) / n_tar
            m_d = ( tf.reduce_sum( non_mask * self.S ) ) / n_non

            v = p_eff * ( tf.reduce_sum( tar_mask * (self.S - m_e)**2 ) ) / n_tar
            v += (1- p_eff) * ( tf.reduce_sum( non_mask * (self.S - m_d)**2 ) ) / n_non

            a = (m_e - m_d) / v
            b = -a *(m_e + m_d)/2
            
            self.L = tf.reduce_sum( W * loss_fcn( -Y * ((a*self.S +b)- tau)))
        else:                
            self.L = tf.reduce_sum( W * loss_fcn( -Y * (self.S - tau)))

        # Add L2 regularization. We include a factor 0.5.
        # Also, we use an adjustment to avoid off-diagonal elements
        # of P and Q to be double penalized.
        # If reg_to is given, we do regularization towards these 
        # parameters instead of towards 0.
        self.R = tf.constant(0.0, dtype=self.floatX)
        if ( reg_to == [] ):
            if ( np.sum(np.abs( l2_reg_P )) > 0 ):    # This should work both it l2_reg_P is a scalar or a matrix
                self.R = self.R +  0.25 * ( tf.reduce_sum( np.diagonal(np.atleast_2d(l2_reg_P)) *
                                                           ( tf.diag_part( self.P )  **2) ) +
                                            tf.reduce_sum( l2_reg_P *(  self.P  **2 )))
            if ( np.sum(np.abs( l2_reg_Q )) > 0 ):
                self.R = self.R + 0.25 * ( tf.reduce_sum( np.diagonal(np.atleast_2d(l2_reg_Q)) *
                                                          ( T.diag_part( self.Q )  **2) ) +  
                                           tf.reduce_sum( l2_reg_Q *(  self.Q  **2 )))
            if ( np.sum(np.abs( l2_reg_c )) > 0 ):
                self.R = self.R + 0.5 * tf.reduce_sum(l2_reg_c *(self.c **2) )
            if (l2_reg_k) > 0:
                self.R = self.R + 0.5 * l2_reg_k *(self.k **2) 
        else:
            assert (len(reg_to) == 4)
            if ( np.sum(np.abs( l2_reg_P )) > 0 ):
                self.R = self.R + 0.25 * ( tf.reduce_sum( np.diagonal(np.atleast_2d(l2_reg_P)) *
                                                          ( ( T.diag_part( self.P ) - T.diagonal( reg_to[0] )) **2) ) +  
                                           tf.reduce_sum( l2_reg_P *( ( self.P - reg_to[0] ) **2 )))
            if ( np.sum(np.abs( l2_reg_Q )) > 0 ):
                self.R = self.R + 0.25 * ( tf.reduce_sum( np.diagonal(np.atleast_2d(l2_reg_Q)) *
                                                          ( ( T.diag_part( self.Q ) - T.diagonal( reg_to[1] )) **2) ) +  
                                           tf.reduce_sum( l2_reg_Q * ( ( self.Q - reg_to[1] ) **2 ) )) 
            if ( np.sum(np.abs( l2_reg_c )) > 0 ):
                self.R = self.R + 0.5 * tf.reduce_sum( l2_reg_c * ( (self.c - reg_to[2]) **2) )
            if (l2_reg_k) > 0:
                self.R = self.R + 0.5 * l2_reg_k * ( (self.k - reg_to[3]) **2) 

        # Score regularization
        # Note 1, alpha needs to be included in the regularization otherwise
        # even if regularization forces the score before multiplication with
        # alpha, i.e., self.S_reg, to be small, this regularization could
        # completely be compensated for by increasing alpha.
        # Note 2, we probably don't want to do this if an external score scaling
        # is used since it wouldn't help for the same reason. 
        if ( l2_reg_score > 0.0 ):
            #self.R = self.R + tf.reduce_sum( l2_reg_score * W * (( self.S_reg ) **2 ) )
            if (self.score_scaling != None ):
                log.warning( "WARNING: You are regularizing the score while using an external score scaling. Is this what you want? " )
            self.R = self.R + tf.reduce_sum( l2_reg_score * W * ( ( self.alpha * self.S_reg ) **2 ) )

        if ( sep_reg ):
            return [ self.L, self.R]
        else:
            return self.L + self.R






        
# This model the same as above except that the score is calculated based on
# the original model parameters m, B, and W. The score calculation is based
# on Ioffe's paper and similar to how it is done in Kaldi. This means we are
# training the parameters m, B, and W instead of P, Q, c, k. As as consequence,
# we can do training/testing with multiple enrollment sessions.
# It is not really a good organization that this one inherits from the previous
# one because they are equally fundamental. But since one was written later,
# it was the simplest to do in this way. 
class dplda_model_constrained(dplda_model):

    # The class should be initialized with normal PLDA 
    # parameters m, W, B.
    # Also, the input TF tensor, M (or placeholder), should be given.
    # --- Making two input ,M1 M2
    # M:  Either a TF tensor or a list containing
    #     one or two tensors. This will be the "enroll"
    #     and "test" data respectively. If only one is provided
    #     we will use it as both "enroll" and "test" data. 
    #def __init__(self, m, W, B, M, score_offset=None, score_scaling=None):
    # def __init__(self, m, W, B=None, M=None, score_offset=None, score_scaling=None):
    def __init__(self, tf_session, param, M, count_info =[], score_offset=None, score_scaling=None, floatX='float32', kaldi_norm=False):

        self.session    = tf_session
        self.floatX     = floatX
        self.kaldi_norm = kaldi_norm

        # Input data (M1="enroll", M2="test")
        if (not isinstance(M, list)):  
            M = [M]                                     
        if (len(M) == 1 ):
            self.M1 = M[0]    
            self.M2 = M[0]
        else:
            self.M1 = M[0]
            self.M2 = M[1]
        
        # The count_info is used for looping over all enroll models with 1 session, all with 2, all with 3 and so on
        if (len(count_info) == 0):
            self.enr_unique_counts = tf.constant(np.array([1], dtype='int32'))
            self.enr_counts_offset = tf.stack([0, tf.shape(self.M1)[0]])
            self.multisession = False
        else:
            assert(len(count_info) == 2)    
            self.enr_unique_counts = count_info[0]  # E.g. [3, 5]
            self.enr_counts_offset = count_info[1]  # Vector e.g. [0,4,9] meaning that enr. spk 0-4 has 3 utts, 4-9 has 5 utts.
            self.multisession = True
            
        # Originally, the input
        if (len(param) != 3):
            log.error("ERROR: Number of parameters in DPLDA initialization should be 3 collected in a list.")

        if isinstance(param[0], tf.Variable):    
            self.m = param[0]
            self.B = param[1]
            self.W = param[2]                
        else:
            self.m = tf.Variable( param[0].astype( self.floatX ), name='plda_m' )
            self.B = tf.Variable( param[1].astype( self.floatX ), name='plda_B' )
            self.W = tf.Variable( param[2].astype( self.floatX ), name='plda_W' )

        # Check that the shape of the parameters is correct and whether we assume
        # diagonal W and B (in which case they are vectors).
        assert (len(self.m.shape) == 1)
        if (len(self.B.shape) == 2):
            log.info("Using full covariance matrices")
            self.diag_cov = False
            assert (len(self.W.shape) == 2)
        else:
            log.info("Using diagonal covariance matrices")
            self.diag_cov = True
            assert (len(self.W.shape) == 1)
            assert (len(self.B.shape) == 1)
                 
        self.score_function = None   

        # If score_offset is provided (a TF varaiable), we will use Niko's piggyback recipe 
        # with this matrix as the baseline scores.
        # We also added a "score_scaling" which will be multiplied with the model scores
        self.score_offset  = score_offset
        self.score_scaling = score_scaling

        self.dim = self.m.get_shape().dims[0].value        
        ### --- LLR score ---------------------- ###

        dim = self.m.shape[0]
        
        # Necessary ??
        W = 0.5*(self.W + tf.transpose( self.W ) )
        B = 0.5*(self.B + tf.transpose( self.B ) )


        I_1xn2 = tf.transpose( tf.ones_like( self.M2[:,0:1] ) )

        
        if ( self.kaldi_norm ):
            norm_vector = 1.0/( B + 1.0 / 1 ) 
            norm        = tf.reduce_sum( (self.M2 ** 2) * norm_vector, axis=1, keep_dims=True )
            norm        = tf.sqrt(self.dim / norm )
            M2 = self.M2 * norm
        else:
            M2 = self.M2
        
        ######################################
        # mean !!!!!!!!1
        # NEED TO REPLACE W WITH I IN THE DIAGONAL CASE, OR ADD CHECK THAT W=I
        ######################################
        def body(Sin, i):
            # We distinguish the case of diagonal and full covariance matrices
            # because computations can be made more efficiently for diagonal ones.
            n     = self.enr_unique_counts[i[-1]]                 # Number of sessions
            start = self.enr_counts_offset[i[-1]]
            end   = self.enr_counts_offset[i[-1] +1]
            M1    = self.M1[start:end,:]
            I_1xn1 = tf.transpose( tf.ones_like( M1[:,0:1] ) )    # "0:1" is just so that shape becomes (?,1) instead of (?,)

            if ( self.kaldi_norm ):
                norm_vector = 1.0/( B + 1.0 / tf.cast(n, self.floatX) ) 
                norm        = tf.reduce_sum( (M1 ** 2) * norm_vector, axis=1, keep_dims=True )
                norm = tf.sqrt(self.dim / norm)                        # Get dim!!!!!!!!!
                M1 = M1 * norm
            
            if self.diag_cov:
                P0 = 1.0 / (W + B / W )                           # Take this and the below out from "body" ??
                C0 = 0.5 * tf.reduce_sum( tf.log( P0 ) )          # We don't include 2pi^(0.5d) since it will be cancelled out.
                G   = B / (tf.cast(n, self.floatX)*B + W )        # ( d x d )  Need this at two places
                M1n = tf.cast(n, self.floatX)* M1 * G             # ( n1 x d ) The "normalized" enrollemnt means
                P   = 1.0 / (W + G)                               # Precision 

                C = 0.5 * tf.reduce_sum( tf.log( P ) )

                   
                # For each normalized enrolment ivector, m1n, and each test i-vector, m2, we would like
                # to calculate (m1n - ,2)P(m1n - ,2)'. We do that with a couple of matrix products rather
                # than a loop. It could be worth thinking about whether it can be done with one operation
                # involving 3D tensors.
                S1   = tf.matmul( M1n,  tf.transpose(M2 * P ) )

                S2_1 = tf.matmul( tf.reduce_sum((( M1n * P) *     M1n ), axis=1 )[:, None], I_1xn2 )
                S2_2 = tf.matmul( tf.reduce_sum(((M2 * P) * M2 ), axis=1 )[:, None], I_1xn1 ) 

                S2_2_0 = tf.matmul( tf.reduce_sum(((M2 *  P0) * M2 ), axis=1 )[:, None], I_1xn1 )

            else:
                # THIS IS WRONG
                P0 = tf.matrix_inverse(W + tf.matmul(B, tf.matrix_inverse(W)))
                C0 = 0.5 * tf.reduce_sum(tf.log(tf.diag_part(P0)))   # We don't include 2pi^(0.5d) since it will be cancelled out.


                #n   = 1                                             # Number of sessions
                G   = tf.matmul(B, tf.matrix_inverse(n*B + W) )      # ( d x d )  Need this at two places
                M1n = n * tf.matmul(M1, G)                           # ( n1 x d ) The "normalized" enrollemnt means
                P   = tf.matrix_inverse(W + G)                       # Precision 

                #S, C = 0.5*tf.slogdet(P) # We don't include 2pi^(0.5d) since it will be cancelled out.
                C = 0.5 * tf.reduce_sum(tf.log(tf.diag_part(P)))

                # For each normalized enrolment ivector, m1n, and each test i-vector, m2, we would like
                # to calculate (m1n - ,2)P(m1n - ,2)'. We do that with a couple of matrix products rather
                # than a loop. It could be worth thinking about whether it can be done with one operation
                # involving 3D tensors.
                S1   = tf.matmul( M1n, tf.matmul(P, M2, transpose_b=True) )
                S2_1 = tf.matmul( tf.reduce_sum(((tf.matmul(    M1n, P)) *     M1n ), axis=1 )[:, None], I_1xn2 )
                S2_2 = tf.matmul( tf.reduce_sum(((tf.matmul(M2, P)) * M2 ), axis=1 )[:, None], I_1xn1 ) 

                S2_2_0 = tf.matmul( tf.reduce_sum(((tf.matmul(M2, P0)) * M2 ), axis=1 )[:, None], I_1xn1 )        

                
            S = 0.5 * (2*S1 - S2_1 - tf.transpose(S2_2) + tf.transpose(S2_2_0) ) + C - C0        
            S = tf.concat([Sin,S ], axis=0)

            return [S, tf.add(i, 1) ]

        ####
        # Initial tensors for counter, i, and embeddings, M,
        i0_ = tf.constant(np.array([0]), dtype='int32', name='pool_loop_index')
        S0_ = tf.transpose(tf.zeros_like(self.M2[:,0:0] ))

        #print tf.shape(self.enr_unique_counts)[0]
        self.S = tf.while_loop(cond =lambda S, i: tf.less(i, tf.shape(self.enr_unique_counts)[0] )[0],
                               body=body, loop_vars=[S0_, i0_ ],
                               shape_invariants=[tf.TensorShape([None, M2.shape[0]]), i0_.get_shape()],
                               parallel_iterations=1, swap_memory=False)[0]
        
        ###
        # The final score from the model. This score will be regularized in case of 
        # score regularization so keep it in "S_reg"
        self.S_reg = self.S 
        # Final score
        if (self.score_offset != None):
            
            # Include score_offsets and scaling if wanted. For now, we do not allow 
            # the case where to do scaling without offset
            if (self.score_scaling != None):
                self.S = self.S * self.score_scaling 

            # Weights for new model. Not sure if 0.5 is the best inital value.
            self.alpha = tf.Variable( np.array(0.5, dtype=floatX), name='dplda_alpha' )  
            self.beta  = tf.Variable( np.array(0.5, dtype=floatX), name='dplda_beta' )    

            self.S = self.alpha * self.S  + self.beta * self.score_offset

    def get_parameters(self):
        if (self.score_offset == None):
            return [self.m, self.B, self.W ]
        else:
            return [self.m, self.B, self.W, self.alpha, self.beta ]

    def get_parameter_values(self):
        params = []
        with self.session.as_default(): 
          for p in self.get_parameters():
            params += [ p.eval() ]
        return params

    def set_parameter_values(self, params):
        #with self.session.as_default():
        self.session.run( tf.assign( self.m, params[0] ) )
        self.session.run( tf.assign( self.B, params[1] ) )
        self.session.run( tf.assign( self.W, params[2] ) )

        if (self.score_offset != None):
            if (len(params) == 4 ):
                print("WARNING: score_offset is used but I was asked to set only 4 params ---alpha and beta will not be set")
            else:
                self.session.run( tf.assign( self.alpha, params[3] ) )
                self.session.run( tf.assign( self.beta,  params[4] ) )
                      
    def load(self, model_file):
        try:
            with h5py.File(model_file, 'r', driver='core') as f:
                m = np.array( f['/m'] )
                B = np.array( f['/B'] )
                W = np.array( f['/W'] )

                if '/alpha' in f:
                    aa = np.array( f['/alpha'] )
                    bb = np.array( f['/beta'] )
                else:
                    aa = None
                    bb  = None 
                
        except IOError:
            raise Exception("Cannot open data file [%s] for reading" % model_pytel_gplda)
        if (aa == None):
            self.set_parameter_values([m.astype(floatX), B.astype(floatX), W.astype(floatX) ])
        else:
            self.set_parameter_values([m.astype(floatX), B.astype(floatX), W.astype(floatX),
                                       aa.astype(floatX).squeeze(), bb.astype(floatX).squeeze()])
            

    def score(self, MX, X=[]):

        if (self.score_function == None):
            if ( X == [] ):
                # Perhaps add some check that M is a placeholder and not a standard tensor in this case.
                if ( not self.multisession ):
                    self.score_function = lambda M1, M2: self.session.run( self.S , {self.M1: M1, self.M2: M2})
                else:
                    self.score_function = lambda M1, M2, CU, IDX: self.session.run( self.S , {self.M1: M1, self.M2: M2,
                                                                                              self.enr_unique_counts:CU,
                                                                                              self.enr_counts_offset:IDX})
            else:
                # Perhaps add check the M is a standard tensor and not a placeholder.
                if (not isinstance(X, list)):  
                    X = [X]                                     
                if ( not self.multisession ):                        
                    self.score_function = lambda X1, X2: self.session.run( self.S , {X[0]: X1, X[1]: X2})
                else:
                    self.score_function = lambda X1, X2, CU, IDX: self.session.run( self.S , {X[0]: X1, X[1]: X2,
                                                                                              self.enr_unique_counts:CU,
                                                                                              self.enr_counts_offset:IDX})

        # Input data (M1="enroll", M2="test")
        if (not isinstance(MX, list)):  
            MX = [MX]                                     
        return self.score_function( *MX )



class dplda_simple(object):

    def __init__(self, session, m, B, W, floatX):

        self.session = session
        self.floatX  = floatX
        
        [P, Q, c, k] = mBW_2_PQck(m, B, W)
        self.P_ = tf.Variable( P.astype( floatX ), name='dplda_P' )
        self.Q_ = tf.Variable( Q.astype( floatX ), name='dplda_Q' )
        self.c_ = tf.Variable( c.astype( floatX ), name='dplda_c' )
        self.k_ = tf.Variable( k.astype( floatX ), name='dplda_k' )
        self.P_sym_ = 0.5*(self.P_ + tf.transpose(self.P_))
        self.Q_sym_ = 0.5*(self.Q_ + tf.transpose(self.Q_))


    def __call__(self, M1_, M2_):   

          SP_  = tf.reduce_sum(tf.tensordot(M1_, self.P_sym_, axes=[[2], [0]]) * M2_, axis=2 )
          SQ1_ = tf.reduce_sum(tf.tensordot(M1_, self.Q_sym_, axes=[[2], [0]]) * M1_, axis=2 )
          SQ2_ = tf.reduce_sum(tf.tensordot(M2_, self.Q_sym_, axes=[[2], [0]]) * M2_, axis=2 )
          Sc_  = tf.tensordot( M1_ + M2_, self.c_, axes=[[2],[0]])

          S_ = SP_ + SQ1_ + SQ2_ + Sc_ + self.k_
          
          return S_
          
        
    def get_parameters(self):
        return [self.P_, self.Q_, self.c_, self.k_]

    def get_parameter_values(self):
        params = []
        with self.session.as_default(): 
            for p in self.get_parameters():
                params += [ p.eval() ]
        return params

    def set_parameter_values(self, params):
        for i, p in enumerate( self.get_parameters() ):
            self.session.run( tf.assign( p, params[i] ) )

          
