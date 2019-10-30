import numpy as np

def init_params_simple_he(sizes, input_mean=[], input_std=[], floatX='float32'):

  params_dict = {}

  if ( input_mean != [] ):
    params_dict["input_mean"] = input_mean.astype( floatX)

  if ( input_std != [] ):
    params_dict["input_std"] = input_std.astype( floatX )

  for ii in range(1, len( sizes )):   
    s = 1.0/np.sqrt((sizes[ii-1] )/2.0) 
    params_dict[ 'W_'+str(ii) ] = np.random.randn( sizes[ii-1], sizes[ii]).astype( floatX )*s

  for ii in range(1, len(sizes )): 
    #params_dict[ 'b_bfr_pool_'+str(ii) ] = np.random.random(sizes[ii]).astype(T.config.floatX)*0.0
    params_dict[ 'b_'+str(ii) ] = np.zeros(sizes[ii]).astype( floatX )

  return params_dict

def init_params_simple_he_uniform(sizes, input_mean=[], input_std=[], floatX='float32', use_bug=False):

  params_dict = {}

  if ( input_mean != [] ):
    params_dict["input_mean"] = input_mean.astype( floatX)

  if ( input_std != [] ):
    params_dict["input_std"] = input_std.astype( floatX )

  if use_bug:  
    for ii in range(1, len( sizes )):
      print "Buggy init"
      params_dict[ 'W_'+str(ii) ] = np.random.uniform(-np.sqrt(6)/sizes[ii-1], np.sqrt(6)/sizes[ii-1],
                                                      (sizes[ii-1], sizes[ii]) ).astype( floatX )
  else:
    print "Bug free init"
    for ii in range(1, len( sizes )):   
      params_dict[ 'W_'+str(ii) ] = np.random.uniform(-np.sqrt(6.0/sizes[ii-1]), np.sqrt(6.0/sizes[ii-1]),
                                                      (sizes[ii-1], sizes[ii]) ).astype( floatX )
    
  for ii in range(1, len(sizes )): 
    params_dict[ 'b_'+str(ii) ] = np.zeros(sizes[ii]).astype( floatX )

  return params_dict


def init_params_simple_he_uniform_full_spec(sizes, input_mean=[], input_std=[], floatX='float32', use_bug=False):

  params_dict = {}

  if ( input_mean != [] ):
    params_dict["input_mean"] = input_mean.astype( floatX)

  if ( input_std != [] ):
    params_dict["input_std"] = input_std.astype( floatX )

  if use_bug:
    print "Buggy init"
    for ii in range(0, len( sizes ) ):   
      params_dict[ 'W_'+str(ii +1 ) ] = np.random.uniform(-np.sqrt(6)/sizes[ii][0], np.sqrt(6)/sizes[ii][0],
                                                          (sizes[ii][0], sizes[ii][1]) ).astype( floatX )
  else:
    print "Bug free init"
    for ii in range(0, len( sizes ) ):   
      params_dict[ 'W_'+str(ii +1 ) ] = np.random.uniform(-np.sqrt(6.0/sizes[ii][0]), np.sqrt(6.0/sizes[ii][0]),
                                                          (sizes[ii][0], sizes[ii][1]) ).astype( floatX )

    
  for ii in range(0, len(sizes )): 
    params_dict[ 'b_'+str(ii + 1) ] = np.zeros(sizes[ii][1]).astype( floatX )
  #for ii in range(0, len(sizes )): 
  #  params_dict[ 'b_'+str(ii + 1) ] = np.random.uniform(-1, 1, ( sizes[ii][1]) ).astype( floatX )

    
  return params_dict
