#!/usr/bin/env python

import argparse, h5py, glob, sys, os.path
import numpy as np
from utils.misc import get_logger

if ( __name__ == "__main__" ):

    parser = argparse.ArgumentParser(
        description = 'Collects embeddings from many h5 files into one.' )
    parser.add_argument(
        '-d', '--data_dir', type =str, help ='Directory with the h5 files',
        required =True)
    parser.add_argument(
        '-n', '--file_name', type =str, help ='File_Name of h5 files to collect in form x.d, x is the file_name, d is a index',
        required =True)
    parser.add_argument(
        '-s', '--scp', type =str,
        help ='Comma separated Subset list of scp files for the desired sets. Need to have both name and physical.',
        required =True)
    parser.add_argument(
        '-ns', '--name_sep', type =str,
        help ='Separator for speaker name in front of filename. If provided, speaker name will be reomoved.',
        required =False, default="")
    parser.add_argument(
        '-as', '--augmentation_sep', type =str,
        help ='Separator for augmentation following filename. If provided, it will be changed to :. Also, : will be added to the end of file names without augmenation.',
        required =False, default="")

    
    args = parser.parse_args()

    data_dir = args.data_dir
    file_name = args.file_name
    scp = args.scp
    name_sep = args.name_sep 
    aug_sep = args.augmentation_sep 
    
    log = get_logger()
    
    log.info("data_dir:  " + data_dir)
    log.info("file_name: " + file_name)
    log.info("scp:       " + scp)
    log.info("name_sep:  " + name_sep )
    log.info("aug_sep:   " + aug_sep )


    in_h5_files = sorted(glob.glob(data_dir + "/" + file_name + '.[0-9]*'))

    n_in_h5_files = len( in_h5_files )

    if ( n_in_h5_files == 0 ):
        log.error("Error didn't find any h5 files")
        sys.exit(-1)

    
    log.info("Found %d h5 files" % n_in_h5_files)
    #for h in in_h5_files:
    #    print ( h) 
    
    ######################################################################
    # Read in the content of the h5 files
    data = []
    physical = []
    for h in in_h5_files:
        try:
            with h5py.File( h, 'r', driver='core') as f:
                data_tmp     = np.array( f['/Data'] ) 
                physical_tmp =  list(f['/Physical'] )
        except IOError:
            raise Exception("Cannot load model")

        if ( name_sep != "" ):
            physical_tmp = [ name_sep.join( p.split( name_sep )[1:] ) for p in physical_tmp ]
        if ( aug_sep != "" ):
            physical_tmp = [ p + ":"  for p in physical_tmp ]
            # Note: the below shoud be safe because the only : in the name should be the one added above
            for i,p in enumerate( physical_tmp) :
                #if i < 10:
                #    print(physical_tmp[i])
                physical_tmp[i] = physical_tmp[i].replace(aug_sep + "noise:", ":noise")
                physical_tmp[i] = physical_tmp[i].replace(aug_sep + "reverb:", ":reverb")
                physical_tmp[i] = physical_tmp[i].replace(aug_sep + "music:", ":music")
                physical_tmp[i] = physical_tmp[i].replace(aug_sep + "babble:", ":babble")
                physical_tmp[i] = physical_tmp[i].replace(aug_sep + "comp:", ":comp")
                #if i < 10:
                #    print(physical_tmp[i])

            
        n_embds =  data_tmp.shape[0] 
        assert( n_embds == len(physical_tmp) ) 
        #log.debug("Read %d embeddings from %s" %(n_embds,h) )

        data.append( data_tmp )
        physical += physical_tmp 
        
    data = np.vstack(data)
    assert( data.shape[0] == len(physical) )
    log.info("Read a total of %d embeddings from %s.*" %(data.shape[0], data_dir + "/" + file_name))

    ######################################################################
    # Process the scps
    
    scp = scp.rstrip().split(",")
    for ss in scp:
        sss = os.path.splitext(ss)[0].split("/")[-1]
        log.info("Writing h5 files and scp for subset %s" %ss )
        
        # Read in the scp
        physical_2_name = {}
        n_scp_entry = 0
        with open(ss, 'r') as f: 
            for line in f:
                # The below is to accept either " " or "=" as separator.
                scp_info   = line.rstrip().replace("="," ").split(" ")    
                assert( len(scp_info) == 2 )
                
                if ( name_sep != "" ):
                    scp_info[1] = name_sep.join( scp_info[1].split( name_sep )[1:] )

                if ( aug_sep != "" ):
                    #print (scp_info[1])
                    scp_info[1] = scp_info[1] + ":"
                    #print (scp_info[1])
                    scp_info[1] = scp_info[1].replace(aug_sep + "noise:", ":noise")
                    scp_info[1] = scp_info[1].replace(aug_sep + "reverb:", ":reverb")
                    scp_info[1] = scp_info[1].replace(aug_sep + "music:", ":music")
                    scp_info[1] = scp_info[1].replace(aug_sep + "babble:", ":babble")
                    scp_info[1] = scp_info[1].replace(aug_sep + "comp:", ":comp")
                    #print (scp_info[1])
                    

                physical_2_name[scp_info[1]] = scp_info[0]
                n_scp_entry += 1
        log.info("Read %d entries from scp %s " %(n_scp_entry, ss) )

        indices = []
        for i,p in enumerate(physical):
            if p in physical_2_name.keys():
                indices.append(i)

        if( len(physical_2_name.keys()) == len(indices) ):
            log.info("All %d physicals found in the embeddings." %len(indices) )
        else:
            log.warning("Only %d physicals out of %d was found in the embeddings." %(len(indices), len(physical_2_name.keys())) )
            
        sub_indices = np.array( indices ).astype(int)
        log.info("Found ")

        # Write scp for the subset
        if ( name_sep == "" ):
            h5_out = data_dir + "/" + sss + '.h5'
        else:
            h5_out = data_dir + "/" + sss.replace("_HZ", "") + '.h5'
             
        log.info("Saving h5 file to " + h5_out )
        with h5py.File(h5_out, 'w', 'core') as f:
            f.create_dataset('Physical', data=[physical[i].encode('utf-8') for i in indices], dtype=h5py.special_dtype(vlen=str) )
            f.create_dataset('Data', data=data[sub_indices] )
            f.create_dataset('Name', data=[physical_2_name[ physical[i] ].encode('utf-8') for i in indices], dtype=h5py.special_dtype(vlen=str) )

        # Write scp for the subset
        if ( name_sep == "" ):
            scp_out = data_dir + "/" + sss + '.scp'
        else:                
            scp_out = data_dir + "/" + sss.replace("_HZ", "") + '.scp'
            
        log.info( "Saving scp to " + scp_out )
        with open(scp_out, 'w') as f:     
            for i in sub_indices :
                f.write( physical_2_name[physical[i]] + " " + physical[i] + "\n" )
        



        
