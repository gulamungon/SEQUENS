class jfa(object):

    def __init__(self, tf_session, param, stats1, stats2=None, floatX='float32'):
        self.session = tf_session
        self.floatX  = floatX

        if ( len(param)== 3 ): 
            if isinstance(P, tf.Variable):    
                self.m = param[0]
                self.V = param[1]
                self.U = param[2]
            else:
                self.m = tf.Variable( param[0].astype( self.floatX ), name='jfa_m' )
                self.V = tf.Variable( param[1].astype( self.floatX ), name='jfa_V' )
                self.U = tf.Variable( param[2].astype( self.floatX ), name='jfa_U' )
        if ( len(param)== 4 ): 
            if isinstance(P, tf.Variable):    
                self.m = param[0]
                self.V = param[1]
                self.U = param[2]
                self.D = param[3]
            else:
                self.m = tf.Variable( param[0].astype( self.floatX ), name='jfa_m' )
                self.V = tf.Variable( param[1].astype( self.floatX ), name='jfa_V' )
                self.U = tf.Variable( param[2].astype( self.floatX ), name='jfa_U' )
                self.D = tf.Variable( param[3].astype( self.floatX ), name='jfa_D' )
        else:
            log.error("ERROR: The number of parameters to JFA should be either 3 or 4")


        # Input sufficient statistics, (stat1="enroll", stat2="test")
        # If stat2==None, stat1 is scored against itself.
        self.N1 = stats1[0]    # (G,)   # Where G is the number of Gaussians
        self.F1 = stats1[1]    # (G, F) # F is the feature dimension
         if (stats2 == None):
            log.info("JFA: scoring stats1 against itself")
            self.N1 = stats1[0]
            self.F1 = stats1[1]
        else:
            log.info("JFA: scoring stats1 against stats2")
            self.N1 = stats2[0]
            self.F1 = stats2[1]
            
        self.score_function = None   

        # Now calculate the score
        
        
    def __call__():
        
            
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

