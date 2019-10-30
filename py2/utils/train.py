


import time, h5py, sys, shutil, os
import numpy as np

from utils.misc import get_logger
log = get_logger()


#def train_nn(n_epoch, n_batch, lr, last_lr, train_batch, check_dev, get_para_func, set_para_func, model_save_file=[], patience=0, batch_count=0, tf_serv_io=[],session=None):
def train_nn(n_epoch, n_batch, lr, last_lr, train_batch, check_dev, get_para_func, set_para_func, model_save_file=[], patience=0, batch_count=0, save_func=None, patience_2=-1, half_every_N_epochs=-1, save_every_epoch=False):

    # Stuff to control the training 
    L_dev = []                                   # Collect dev. set losses after each epoch        
    L_dev.append( check_dev() )                  # We include the initial loss
    L_dev_best = L_dev[-1]                       # The best loss so far (will be updated during training) 
    best_params = get_para_func()                # The corresponding best parameters
    patience_count = 0                           # The number of times dev set loss did not improve.
    
    # Train the model
    start = time.clock()

    # Save the initial model if a file name was provided.
    if ( not model_save_file == [] ):
        i = 0
        L_epoch = "xxxx"
        model_suffix = "_epoch-" + str(i) + "_lr-" + str(lr) + "_lossTr-" + L_epoch + "_lossDev-" + str(L_dev[-1])
        params    = get_para_func()
        file_name = model_save_file + model_suffix  + ".h5"
        log.info( "Saving model to " + file_name )
        try:
            with h5py.File(file_name, 'w', driver='core') as f:
                for i in range(0,len(params )):
                    for p in range(0, len( params[i] )):
                        name='para_' + str(i) +'_' + str(p)
                        f.create_dataset(name, data =params[i][p])

        except IOError:
            raise Exception("Cannot open file [%s] for writing" % f_name)
        
        save_func( model_save_file, 0 )    

    
    for i in range(1, n_epoch +1 ):

        if (patience_2 != -1) and (i == 150):
            log.info(" Epoch count is 150 and patience_2 is given. Changing to patience_2 ( %d% ).", patience_2)
            patience = patience_2

        if (half_every_N_epochs != -1):
            if i%half_every_N_epochs == 0:
                lr = lr/2.0
                log.info("Halving learning rate using fixed number if epochs. New LR is %f.", lr)
            
        n_failed_batches = 0
        L_epoch          = 0
        j                = 0
        L =1000
        while j < n_batch:
            batch_count += 1
            log.info( "Batch %d/%d", j, n_batch -1)
            if ( j % 25 == 0 ):
                tmp_params = get_para_func()
                j_tmp = j
            try:
                L = train_batch(lr, batch_count)
                L_epoch += L
                j       += 1
            except Exception, e:
                log.error("Failed to train on batch, resetting params")
                log.error("Excption: " + str(e))
                n_failed_batches +=1
                set_para_func( tmp_params )
                j = j_tmp
                sys.exc_clear()
            if np.isnan(L):
                log.error("Loss is nan,resetting params")
                n_failed_batches +=1
                set_para_func( tmp_params )
                j = j_tmp
                sys.exc_clear()

                
        # The average training loss for the epoch    
        L_epoch = L_epoch / n_batch 

        # Check the loss on the devolpment set after the epoch
        L_dev.append( check_dev() )
        
        # If the dev. loss has improved, we store the new parameters. If it fails to improve
        # for more than patience times, we halve the learning rate and reset the params to
        # last succesful ones (best_params) 
        if ( L_dev[-1] <  L_dev_best ):
            log.info("Finished epoch: " + str(i) + ", Avg. Training loss: " + str(L_epoch) + " Dev. loss " + str(L_dev[-1]) + " Prev. Best dev. loss " + str(L_dev_best) + "  Development loss has improved")
            if  (patience_count > 0) :
                log.info( "     Reset patient count to 0.")
                patience_count = 0
            L_dev_best = L_dev[-1]
            best_params = get_para_func()
            
            if ( not model_save_file == [] ):
                model_suffix="_epoch-" + str(i) + "_lr-" + str(lr) + "_lossTr-" + str(L_epoch) + "_lossDev-" + str(L_dev[-1]) 
                params    = get_para_func()
                file_name = model_save_file + model_suffix + ".h5"
                log.info( "Saving model to " + file_name )
                try:
                    with h5py.File(file_name, 'w', driver='core') as f:
                        for ii in range(0,len(params )):
                            for p in range(0, len( params[ii] )):
                                name='para_' + str(ii) +'_' + str(p)
                                f.create_dataset(name, data =params[ii][p])

                except IOError:
                    raise Exception("Cannot open file [%s] for writing" % f_name)
                save_func( model_save_file, i )    

        else:
            log.info("Finished epoch: " + str(i) + ", Avg. Training loss: " +str(L_epoch) + " Dev. loss " + str(L_dev[-1]) + " Prev. Best dev. loss " + str(L_dev_best) + "Development loss did not improve")
        #if (n_failed_batches > 0 ):
            if save_every_epoch:
                if ( not model_save_file == [] ):
                    model_suffix="_epoch-" + str(i) + "_lr-" + str(lr) + "_lossTr-" + str(L_epoch) + "_lossDev-" + str(L_dev[-1]) 
                    params    = get_para_func()
                    file_name = model_save_file + model_suffix + ".h5"
                    log.info( "Saving model to " + file_name )
                    try:
                        with h5py.File(file_name, 'w', driver='core') as f:
                            for ii in range(0,len(params )):
                                for p in range(0, len( params[ii] )):
                                    name='para_' + str(ii) +'_' + str(p)
                                    f.create_dataset(name, data =params[ii][p])

                    except IOError:
                        raise Exception("Cannot open file [%s] for writing" % f_name)
                    save_func( model_save_file, i )    

            
            patience_count += 1
            if ( patience_count <= patience ):
                log.info( "  -- Development loss DID NOT improve. Patience count is now " + str(patience_count) )
            elif (half_every_N_epochs == -1):
                lr = lr * 0.5
                set_para_func( best_params )
                log.info( "  -- Development loss DID NOT improve for " + str(patience_count) +" iterations and patience is " + str(patience) + "." )
                log.info( "     Reset params to the best ones. Halv the learning rate to " + str(lr) )
                log.info( "     Reset patient count to 0."       )
                patience_count = 0

        log.error("Number failed batches: " + str(n_failed_batches))

                
        # We stop the training if the learning rate has been
        # reduced below a threshold
        if ( lr < last_lr ):
            break

    end = time.clock()

    log.info ("Training done. Training time: " + str(end - start) + "s" )


# This generates a batch and train on it. Uses the above iterator and 
# the TF train function.
#def get_train_batch_fkn(all_bad_utts, it_tr_que, train_function, P_eff, tau, lr_scale_b_pool=1.0, lr_scale_a_pool=1.0, lr_scale_dplda=1.0, lr_scale_multi=1.0, piggyback=False):                           
def get_train_batch_fkn(it_tr_que, train_function, P_eff, tau, lr_scale_b_pool=1.0, lr_scale_a_pool=1.0, lr_scale_dplda=1.0, lr_scale_multi=1.0, piggyback=False):                           
    from tensorflow_code.dplda import lab2matrix, labMat2weight

    
    
    def train_batch_fkn(lr, batch_count):
        #global batch_count
        global lr_first # We need this so that load and save functions can 
        lr_first = lr   # get it
                             
        #batch_count += 1
        log.info ( "Batch count " + str(batch_count) )

        [[X, Y, U], bad_tr_files, [tr_feats, tr_idx ], it_batch_nb, it_ctrl_nb]     = it_tr_que.get_batch()
        print "A"
        print batch_count
        print it_batch_nb
        print "B"
        assert(batch_count == it_batch_nb)
        #all_bad_utts += bad_tr_files

        YY    = lab2matrix( Y.squeeze() )
        WGT   = labMat2weight( YY, P_eff )
        WGT_m = np.ones(Y.shape)/ float(Y.shape[0])

        # Adjust learn rate if the desired one has changed
        log.info( "Global lr: " + str( lr ) )
        #log.debug( " lr  b_pool: " + str( lr * lr_scale_b_pool ) )
        #log.debug( " lr  a_pool: " + str( lr * lr_scale_a_pool ) )
        #log.debug( " lr  dplda: " + str( lr * lr_scale_dplda ) )
        #log.debug( " lr  multi: " + str( lr * lr_scale_multi ) )

        if ( piggyback ):
            M_pb = tr_ivec_pb[U, :]
            L_multi, L_dplda, L,_    = train_function(tr_feats, tr_idx, M_pb, M_pb, WGT, YY, tau, WGT_m, Y, lr)
        else:
            info, L, _ = train_function(tr_feats, tr_idx, WGT, YY, tau, WGT_m, Y, lr)
        print "Total loss: " + str(L)
        print "Info: " + str(info)
        #print "DPLDA loss: " + str(L_dplda)
        #print "grads: " + str(grads)
        return L
    return train_batch_fkn

def get_train_batch_fkn_mpi(all_bad_utts, lr_scale_b_pool=1.0, lr_scale_a_pool=1.0, lr_scale_dplda=1.0, lr_scale_multi=1.0, piggyback=False):
    time_all             = 0
    clock_all            = 0
    time_batch_prep      = 0
    clock_batch_prep     = 0
    time_embd            = 0
    clock_embd           = 0
    time_train_dplda     = 0
    clock_train_dplda    = 0
    time_train_grad_f2i  = 0
    clock_train_grad_f2i = 0
    time_train_f2i       = 0
    clock_train_f2i      = 0

    def train_batch_fkn_mpi(lr):

        global batch_count
        global lr_first # We need this so that load and save functions can 
        lr_first = lr   # get it

        global time_all
        global clock_all    
        global time_batch_prep 
        global clock_batch_prep   
        global time_embd          
        global clock_embd         
        global time_train_dplda    
        global clock_train_dplda   
        global time_train_grad_f2i 
        global clock_train_grad_f2i
        global time_train_f2i      
        global clock_train_f2i     

        time_b_all    = time.time()
        clock_b_all   = time.clock()

        batch_count += 1

        time_b_batch_prep  = time.time()
        clock_b_batch_prep = time.clock()
        [[X, Y, U], bad_tr_files, [tr_feats, tr_idx ], it_batch_nb, it_ctrl_nb] = it_tr_que.get_batch()
        time_a_batch_prep  = time.time()
        clock_a_batch_prep = time.clock()

        # This is a check that all machines are processing the same batch.
        # And that the first utterance index is the same. So if seed of it_tr
        # differ, it will be detected.
        assert(batch_count == it_batch_nb)
        if (is_master):            
            log.info( "Batch count " + str(batch_count) )
            log.info( "Global lr: " + str( lr ) )
            log.debug( " lr  b_pool: " + str( lr * lr_scale_b_pool ) )
            log.debug( " lr  a_pool: " + str( lr * lr_scale_a_pool ) )
            log.debug( " lr  dplda: " + str( lr * lr_scale_dplda ) )
            log.debug( " lr  multi: " + str( lr * lr_scale_multi ) )

            log.debug( "Control number: " +str(it_ctrl_nb) )
            for i in range(1, mpi_size):
                mpi_comm.send( [batch_count, it_ctrl_nb], dest=i )
        else:
            batch_count_master, it_ctrl_nb_master = mpi_comm.recv( source=0 )
            log.debug( "Batch count " + str(batch_count) )
            log.debug( "Received batch number from Master: " + str(batch_count_master) )
            log.debug( "Control number: " + str(it_ctrl_nb) )
            log.debug( "Received control number from Master: " +str(it_ctrl_nb_master) )               
            assert( batch_count_master == batch_count )
            assert( it_ctrl_nb_master == it_ctrl_nb )

        all_bad_utts += bad_tr_files

        # Get the embeddings
        time_b_embd    = time.time()
        clock_b_embd   = time.clock()
        M              = feat2embd(tr_feats, tr_idx) 
        time_a_embd    = time.time()
        clock_a_embd   = time.clock()

        # If MPI, all except worker 0, sends their embeddings M to worker 0. 
        if ( not is_master ):
            mpi_comm.send( [M, Y], dest=0 )
        else:
            job_indices = np.array([0, Y.shape[0] ])
            for i in range(1, mpi_size):
                [ M_received, Y_received ]  = mpi_comm.recv( source=i )
                log.debug( "Received embeddings from worker %d" % i )

                M = np.concatenate( (M, M_received), axis =0)
                Y = np.concatenate( (Y, Y_received), axis =0)
                job_indices = np.concatenate( (job_indices, Y_received.shape) )

            job_indices = np.cumsum( job_indices )    
            YY    = lab2matrix( Y.squeeze() )
            WGT   = labMat2weight( YY, P_eff )
            WGT_m = np.ones(Y.shape)/ float(Y.shape[0])

        # Train DPLDA part + get gradient of loss with respect to embeddings        
        # This occurs only on worker 0.         
        if ( is_master ):

            time_b_train_dplda    = time.time()
            clock_b_train_dplda   = time.clock()
            if ( piggyback ):
                [L_multi, L_dplda, L, g ] = train_function_i2s(M, m_PB, WGT, YY, tau, WGT_m, Y,lr)
            else:
                [L_multi, L_dplda, L, g ] = train_function_i2s(M, WGT, YY, tau, WGT_m, Y, lr)
            time_a_train_dplda    = time.time()
            clock_a_train_dplda   = time.clock()

            print "Total loss: " + str(L)

            # If MPI, worker 0 sends the gradients of embeddings wrt loss to the other workers
            for i in range(1, mpi_size):
                start_i = int( job_indices[i] ) 
                end_i   = int( job_indices[i + 1] ) 
                mpi_comm.send([L, g[start_i:end_i,:] ] , dest=i )

            # Now reduce g for Master
            start_i = int( job_indices[0] ) 
            end_i   = int( job_indices[1] ) 
            g       = g[start_i:end_i,:]
        else:
            # Slaves recieves the above gradients
            [L, g ] = mpi_comm.recv( source=0 )
            log.debug ( "Received grads from worker 0 with shape %s" % str(g.shape) )

        # Calculate the gradients for the f2i parameters
        time_b_train_grad_f2i  = time.time()
        clock_b_train_grad_f2i = time.clock()
        g_f2i                  = train_function_grad_f2i(tr_feats, tr_idx, g) 
        time_a_train_grad_f2i  = time.time()
        clock_a_train_grad_f2i = time.clock()

        # If MPI, all except worker 0, sends their f2i gradients, g_f2i, to worker 0. 
        if ( not is_master ):
            mpi_comm.send( g_f2i, dest=0 )
        else:
            for i in range(1, mpi_size):
                g_f2i_received = mpi_comm.recv( source=i )
                log.debug( "Received grad_f2i from worker %d" % i )

                g_f2i = [p1+p2 for p1,p2 in  zip(g_f2i,  g_f2i_received) ]  # g_f2i + g_f2i_received 


        # Even if MPI, this occurs only on worker 0.         
        if ( is_master ):
            time_b_train_f2i  = time.time()
            clock_b_train_f2i = time.clock()
            L_f2i             = train_function_f2i(g_f2i,lr)
            time_a_train_f2i  = time.time()
            clock_a_train_f2i = time.clock()

        # If MPI, Finally update the f2i part on each worker, except 0 were we just updated.
        ####        if ( is_master ):            
            para = []
            for p in params_to_update_b_pool_ + params_to_update_a_pool_:
                para.append( sess.run(p) )

            for i in range(1, mpi_size):
                mpi_comm.send( para , dest=i )
        else:
            para  = mpi_comm.recv( source =0 )
            para_ = params_to_update_b_pool_ + params_to_update_a_pool_ 
            for i in range(len(para)):
                sess.run( tf.assign( para_[i] , para[i] ) )
            log.debug ( "Received model from worker 0" )

        time_a_all    = time.time()
        clock_a_all   = time.clock()
        # Summarize the times
        time_all             += time_a_all - time_b_all   
        clock_all            += clock_a_all - clock_b_all   
        time_batch_prep       += time_a_batch_prep - time_b_batch_prep   
        clock_batch_prep      += clock_a_batch_prep - clock_b_batch_prep   
        time_embd            += time_a_embd - time_b_embd   
        clock_embd           += clock_a_embd - clock_b_embd   
        if is_master:
            time_train_dplda     += time_a_train_dplda - time_b_train_dplda   
            clock_train_dplda    += clock_a_train_dplda - clock_b_train_dplda   
        time_train_grad_f2i  += time_a_train_grad_f2i - time_b_train_grad_f2i   
        clock_train_grad_f2i += clock_a_train_grad_f2i - clock_b_train_grad_f2i   
        if is_master:
            time_train_f2i       += time_a_train_f2i - time_b_train_f2i   
            clock_train_f2i      += clock_a_train_f2i - clock_b_train_f2i   

        log.debug("Average times so far: ")
        log.debug(" All steps time:                   " + str(time_all / float(batch_count)) + "s" )
        log.debug(" All steps clock:                  " + str(clock_all / float(batch_count)) + "s" )
        log.debug(" Batch preparation time:           " + str(time_batch_prep / float(batch_count)) + "s" )
        log.debug(" Batch preparation clock:          " + str(clock_batch_prep / float(batch_count)) + "s" )
        log.debug(" Extr. embd. time:                 " + str(time_embd / float(batch_count)) + "s" )
        log.debug(" Extr. embd clock:                 " + str(clock_embd / float(batch_count)) + "s" )
        if is_master:
             log.debug(" Train DPLDA time:                 " + str(time_train_dplda / float(batch_count)) + "s" )
             log.debug(" Train DPLDA clock:                " + str(clock_train_dplda / float(batch_count)) + "s" )
        log.debug(" Calculate grad f2i time:          " + str(time_train_grad_f2i / float(batch_count)) + "s" )
        log.debug(" Calculate grad f2i clock:         " + str(clock_train_grad_f2i / float(batch_count)) + "s" )
        if is_master:
            log.debug(" Train (given grad f2i) f2i time:  " + str(time_train_f2i / float(batch_count)) + "s" )
            log.debug(" Train (given grad f2i) f2i clock: " + str(clock_train_f2i / float(batch_count)) + "s" )

        return L 
    return train_batch_fkn_mpi




def get_train_batch_multi_fkn( it_tr_que, train_function ):                           

    
    def train_batch_fkn(lr, batch_count):
                             
        log.info( "Batch count " + str(batch_count) )

        [[X, Y, U], bad_tr_files, [tr_feats, tr_idx ], it_batch_nb, it_ctrl_nb]     = it_tr_que.get_batch()
        assert(batch_count == it_batch_nb)

        log.debug("Batch size: " + str(tr_feats.shape))
        
        WGT_m = np.ones(Y.shape)/ float(Y.shape[0])

        info, L, _ = train_function(tr_feats, tr_idx, WGT_m, Y, lr)
        log.info( "Total loss: " + str(L) )
        log.info( "Info: " + str(info) )
        return L
    return train_batch_fkn



def get_train_batch_multi_fkn_bn_mlt( it_tr_que, train_function ):                           

    
    def train_batch_fkn(lr, batch_count):
                             
        log.info( "Batch count " + str(batch_count) )

        [[X, Y, U], bad_tr_files, [tr_feats, tr_idx ], it_batch_nb, it_ctrl_nb]     = it_tr_que.get_batch()
        assert(batch_count == it_batch_nb)

        log.debug("Batch size: " + str(tr_feats[0].shape))
        
        WGT_m = np.ones(Y.shape)/ float(Y.shape[0])

        Info, L, _ = train_function(tr_feats[0], tr_idx, WGT_m, Y, tr_feats[1], lr)
        print "Total loss: " + str(L)
        print "Info: " + str(Info)
        return L
    return train_batch_fkn




def get_train_w_g_c_batch_fcn( it_tr_que, get_w_input, train_function_w, n_w_iter, alpha_1, rng_w, train_function_g_c, alpha_2, n_only_critic_batches=0 ): 

    
    def train_batch_fcn(lr, batch_count):

        log.info( "Batch count " + str(batch_count) )
        #[[X, Y, U], bad_tr_files, [tr_feats, tr_idx ], it_batch_nb, it_ctrl_nb]     = it_tr_que.get_batch()
        [BB, bad_tr_files, [tr_feats, tr_idx ], it_batch_nb, it_ctrl_nb]     = it_tr_que.get_batch()
        X = BB[0]
        Y = BB[1]
        U = BB[2]
        if (len(BB)==4):
            S = np.vstack( [np.vstack(BB[3][0]), np.vstack(BB[3][1])] )
        n_out = Y[0].shape[0]
        n_in  = Y[1].shape[0]

        assert(n_out == n_in ) # For now we require this. Generalize it later on.
        assert(batch_count == it_batch_nb)

        log.debug("Batch size: " + str(tr_feats.shape))

        
        ### Critic 
        log.debug("Training w (critic)")
        if (len(BB)==4):
            M = get_w_input(tr_feats, tr_idx, S)  
        else:
            M = get_w_input(tr_feats, tr_idx)
            
        log.info("M is ")
        print M
        log.info("Max M " + str(np.max(M)))
        log.info("Min M " + str(np.min(M)))
        
        for i in range(n_w_iter):

            # A speed-up could be to to this with 
            out_weights = rng_w.uniform(0,1,n_out)[:,np.newaxis]
            in_weights = 1 - out_weights 

            h = out_weights * M[0:n_out][ rng_w.permutation(n_out) ] + in_weights * M[n_out:n_out + n_in] # Permute one of them is enough.
            l_w, l_grad, L_w_grad, _ = train_function_w(M, h, lr * alpha_1)
        
            log.info(" Iteration %d: l_w=%f, l_grad=%f, L_w_grad=%f ", i,  l_w, l_grad, L_w_grad )


        WGT_m = np.ones(n_out) / float(n_out)

        if (batch_count > n_only_critic_batches ):
            # Rest of network. This will do the forward propagation to M again which is unecessary. The only
            # fix I know is messy so skip it for now.
            log.debug("done. Training generator and classifier")
            if (len(BB)==4):
                loss_w, loss_c, Loss_w_c, _ = train_function_g_c(tr_feats, tr_idx, S, WGT_m, Y[0], lr * alpha_2 )
            else:
                loss_w, loss_c, Loss_w_c, _ = train_function_g_c(tr_feats, tr_idx, WGT_m, Y[0], lr * alpha_2 )
            log.info("loss_w= %f, loss_c=%f, Loss_w_c= %f",  loss_w, loss_c, Loss_w_c)
        else:
            log.debug("done. Do not train generator and classifier since batchcount < %d", n_only_critic_batches)                                         
        return  l_grad # Return this one since this is the one where nan's occur!!

    
    return train_batch_fcn


def get_train_w_g_c_batch_fcn_inDomLabs( it_tr_que, get_w_input, train_function_w, n_w_iter, alpha_1, rng_w, train_function_c,
                                         train_function_g_c, alpha_2, n_only_critic_batches=0 ): 

    
    def train_batch_fcn(lr, batch_count):

        log.info( "Batch count " + str(batch_count) )
        [BB, bad_tr_files, [tr_feats, tr_idx ], it_batch_nb, it_ctrl_nb]     = it_tr_que.get_batch()
        X = BB[0]
        Y = BB[1]
        U = BB[2]
        n_out = Y[0].shape[0]
        n_in  = Y[1].shape[0]
        if (len(BB)==4):
            #S = np.vstack( [np.vstack(BB[3][0]), np.vstack(BB[3][1])] )
            S = np.vstack( (np.zeros((n_out,1)), np.ones((n_in,1)) ) )

        assert(n_out == n_in ) # For now we require this. Generalize it later on.
        assert(batch_count == it_batch_nb)

        log.debug("Batch size: " + str(tr_feats.shape))

        ### Critic 
        log.debug("Training w (critic)")
        if (len(BB)==4):
            M, M_B = get_w_input(tr_feats, tr_idx, S)
        else:
            M, M_B = get_w_input(tr_feats, tr_idx)
            
        log.info("M is ")
        #print M
        log.info("Max M " + str(np.max(M)))
        log.info("Min M " + str(np.min(M)))


        #print M_B.shape
        #print M_B
        log.info("Max M_B " + str(np.max(M_B)))
        log.info("Min M_B " + str(np.min(M_B)))
        
        for i in range(n_w_iter):

            # A speed-up could be to to this with 
            out_weights = rng_w.uniform(0,1,n_out)[:,np.newaxis]
            in_weights = 1 - out_weights 

            h = out_weights * M[0:n_out][ rng_w.permutation(n_out) ] + in_weights * M[n_out:n_out + n_in] # Permute one of them is enough.
            l_w, l_grad, L_w_grad, _ = train_function_w(M, h, lr * alpha_1)
        
            log.info(" Iteration %d: l_w=%f, l_grad=%f, L_w_grad=%f ", i,  l_w, l_grad, L_w_grad )


        WGT_m = np.ones(n_out) / float(n_out) # We assume the same weight for both the domains

        if (batch_count > n_only_critic_batches ):
            # Rest of network. This will do the forward propagation to M again which is unecessary. The only
            # fix I know is messy so skip it for now.
            log.debug("done. Training generator and classifiers")
            if (len(BB)==4):
                loss_w, loss_c_out_d, loss_c_in_d, loss_c, Loss_w_c, _ = train_function_g_c(tr_feats, tr_idx, S, WGT_m, Y[0],
                                                                                            Y[1], lr * alpha_2 )
            else:
                loss_w, loss_c_out_d, loss_c_in_d, loss_c, Loss_w_c, _ = train_function_g_c(tr_feats, tr_idx, WGT_m, Y[0], Y[1],
                                                                                            lr * alpha_2 )
            log.info("loss_w= %f, loss_c_out_d=%f, loss_c_in_d= %f, loss_c=%f, Loss_w_c= %f",
                     loss_w, loss_c_out_d, loss_c_in_d, loss_c, Loss_w_c)
        else:
            log.debug("done. Train only in-domain classifier since batchcount < %d", n_only_critic_batches)
            loss_c_in_d, _ = train_function_c(M_B, WGT_m, Y[1], lr * alpha_2 )
            log.info("loss_c_in_d=%f", loss_c_in_d )

        try:
            return  l_grad # Return this one since this is the one where nan's occur!!
        except NameError:
            log.warning("l_grad not defined. (Maybe no critic iterations was done). Returning loss_c_in_d.")
            return loss_c_in_d

    return train_batch_fcn



def get_train_w_g_c_batch_fcn_inDomLabs_bn_mlt( it_tr_que, get_w_input, train_function_w, n_w_iter, alpha_1, rng_w, train_function_ic_bn,
                                                train_function_g_c_bn, alpha_2, n_only_critic_batches=0 ): 

    
    def train_batch_fcn(lr, batch_count):

        log.info( "Batch count " + str(batch_count) )
        [BB, bad_tr_files, [tr_feats, tr_idx ], it_batch_nb, it_ctrl_nb]     = it_tr_que.get_batch()
        X = BB[0]
        Y = BB[1]
        U = BB[2]
        n_out = Y[0].shape[0]
        n_in  = Y[1].shape[0]
        if (len(BB)==4):
            #S = np.vstack( [np.vstack(BB[3][0]), np.vstack(BB[3][1])] )
            S = np.vstack( (np.zeros((n_out,1)), np.ones((n_in,1)) ) )

        assert(n_out == n_in ) # For now we require this. Generalize it later on.
        assert(batch_count == it_batch_nb)

        log.debug("Batch sizes: " + str(tr_feats[0].shape) + " and " + str(tr_feats[1].shape) )

        
        ### Critic 
        log.debug("Training w (critic)")
        if (len(BB)==4):
            M, M_B = get_w_input(tr_feats[0], tr_idx, S)  
        else:
            M, M_B = get_w_input(tr_feats[0], tr_idx)
            
        log.info("M is ")
        #print M
        log.info("Max M " + str(np.max(M)))
        log.info("Min M " + str(np.min(M)))


        #print M_B.shape
        #print M_B
        log.info("Max M_B " + str(np.max(M_B)))
        log.info("Min M_B " + str(np.min(M_B)))
        
        for i in range(n_w_iter):

            # A speed-up could be to to this with 
            out_weights = rng_w.uniform(0,1,n_out)[:,np.newaxis]
            in_weights = 1 - out_weights 

            h = out_weights * M[0:n_out][ rng_w.permutation(n_out) ] + in_weights * M[n_out:n_out + n_in] # Permute one of them is enough.
            l_w, l_grad, L_w_grad, _ = train_function_w(M, h, lr * alpha_1)
        
            log.info(" Iteration %d: l_w=%f, l_grad=%f, L_w_grad=%f ", i,  l_w, l_grad, L_w_grad )


        WGT_m = np.ones(n_out) / float(n_out) # We assume the same weight for both the domains

        if (batch_count > n_only_critic_batches ):
            # Rest of network. This will do the forward propagation to M again which is unecessary. The only
            # fix I know is messy so skip it for now.
            log.debug("done. Training generator and classifiers")
            if (len(BB)==4):
                loss_w, loss_c_out_d, loss_c_in_d, loss_bn, Loss_w_c_bn, _ = train_function_g_c_bn(tr_feats[0], tr_idx, S, WGT_m, Y[0], Y[1], tr_feats[1], lr * alpha_2 )
            else:
                loss_w, loss_c_out_d, loss_c_in_d, loss_bn, Loss_w_c_bn, _ = train_function_g_c_bn(tr_feats[0], tr_idx, WGT_m, Y[0], Y[1], tr_feats[1], lr * alpha_2 )
            log.info("loss_w= %f, loss_c_out_d=%f, loss_c_in_d= %f, loss_bn=%f, Loss_w_c_bn= %f",
                     loss_w, loss_c_out_d, loss_c_in_d, loss_bn, Loss_w_c_bn)
        else:
            log.debug("done. Train only in-domain classifier and bn predictor  since batchcount < %d", n_only_critic_batches)
            if (len(BB)==4):
                #loss_c_in_d, _ = train_function_c(M_B, S, WGT_m, Y[1], lr * alpha_2 )
                loss_c_in_d_, loss_bn,  Loss_ic_bn, _ = train_function_ic_bn(tr_feats[0], S, tr_idx, tr_feats[1], X1_h, WGT_m, Y[1], lr* alpha_2)
            else:
                #loss_c_in_d, _ = train_function_c(M_B, WGT_m, Y[1], lr * alpha_2 )
                loss_c_in_d_, loss_bn,  Loss_ic_bn, _ = train_function_ic_bn(tr_feats[0], tr_idx, tr_feats[1], WGT_m, Y[1], lr* alpha_2)
            log.info("loss_c_in_d=%f, loss_bn=%f, Loss_ic_bn=%f", loss_c_in_d_, loss_bn,  Loss_ic_bn )

        try:
            return  l_grad # Return this one since this is the one where nan's occur!!
        except NameError:
            log.warning("l_grad not defined. (Maybe no critic iterations was done). Returning loss_c_in_d.")
            return loss_c_in_d

    return train_batch_fcn




#def get_train_batch_multi_reco( it_tr_que_embd, it_tr_que_reco, train_function ):                           
def get_train_batch_multi_reco( it_tr_que, train_function ):                           
    
    def train_batch_fkn(lr, batch_count):
                             
        log.info( "Batch count " + str(batch_count))

        #[[X, Y_e, U], bad_tr_files, [tr_feats_e, tr_idx ], it_batch_nb, it_ctrl_nb]     = it_tr_que_embd.get_batch()
        #[[X, Y_r, U], bad_tr_files, [tr_feats_r, tr_idx ], it_batch_nb, it_ctrl_nb]     = it_tr_que_reco.get_batch()
        [[X, Y, U], bad_tr_files, [[tr_feats_e, tr_feats_r], tr_idx ], it_batch_nb, it_ctrl_nb]     = it_tr_que.get_batch()
        assert(batch_count == it_batch_nb)

        log.debug("Batch size: " + str(tr_feats_e.shape) + " " + str(tr_feats_r.shape))
        
        #WGT_m = np.ones(Y_e.shape)/ float(Y_e.shape[0])
        WGT_m = np.ones(Y.shape)/ float(Y.shape[0])

        #loss_c, loss_reco, Loss, _ = train_function(tr_feats_e, tr_idx, WGT_m, Y_e, tr_feats_r, lr)
        loss_c, loss_reco, Loss, _ = train_function(tr_feats_e, tr_idx, WGT_m, Y, tr_feats_r, lr)
        log.info("loss_c=%f, loss_reco=%f, Loss=%f", loss_c, loss_reco,  Loss )
        return Loss
    return train_batch_fkn



def get_train_batch_multi_reco_bn_utt_norm( it_tr_que, train_function, tr_bn_mean, tr_bn_std):                           
    
    def train_batch_fkn(lr, batch_count):
                             
        log.info( "Batch count " + str(batch_count) )

        [[X, Y, U], bad_tr_files, [[tr_feats_e, tr_feats_r], tr_idx ], it_batch_nb, it_ctrl_nb]     = it_tr_que.get_batch()
        assert(batch_count == it_batch_nb)

        tr_bn_mean_batch = tr_bn_mean[U,:]
        tr_bn_std_batch = tr_bn_std[U,:]
        
        log.debug("Batch size: " + str(tr_feats_e.shape) + " " + str(tr_feats_r.shape))
        
        WGT_m = np.ones(Y.shape)/ float(Y.shape[0])

        loss_c, loss_reco, Loss, _ = train_function(tr_feats_e, tr_idx, WGT_m, Y, tr_feats_r, lr,
                                                    tr_bn_mean_batch[:,np.newaxis,:], tr_bn_std_batch[:,np.newaxis,:])
        log.info("loss_c=%f, loss_reco=%f, Loss=%f", loss_c, loss_reco,  Loss )
        return Loss
    return train_batch_fkn


def get_train_batch_multi_reco_lab( it_tr_que, train_function, ret_loss="Loss_1" ):                           
    
    def train_batch_fkn(lr, batch_count):
                             
        log.info( "Batch count " + str(batch_count) )

        [[X, Y, U], bad_tr_files, [[tr_feats_e, tr_feats_r, lab], tr_idx ], it_batch_nb, it_ctrl_nb]     = it_tr_que.get_batch()
        assert(batch_count == it_batch_nb)

        log.debug("Batch size: " + str(tr_feats_e.shape) + " " + str(tr_feats_r.shape) + " " + str(lab.shape))
        
        WGT_m = np.ones(Y.shape)/ float(Y.shape[0])

        tr_output = train_function(tr_feats_e, tr_idx, WGT_m, Y, tr_feats_r, lab, lr)
        if ( len(tr_output) == 4 ):
            loss_c, loss_reco, Loss, _ = tr_output 
            log.info("loss_c=%f, loss_reco=%f, Loss=%f", loss_c, loss_reco,  Loss )
        else:
            loss_c, loss_reco, Loss_1, _, loss_GB = tr_output 
            log.info("loss_c=%f, loss_reco=%f, loss_GB=%f, Loss_1=%f", loss_c, loss_reco, loss_GB, Loss_1)
        if (ret_loss == "loss_reco"):
            return loss_reco
        elif(ret_loss == "loss_c"):
            return loss_c
        else:
            return Loss_1
        
    return train_batch_fkn

def get_train_batch_multi_reco_lab_chn_multi( it_tr_que, train_function, ret_loss="Loss_1" ):                           
    
    def train_batch_fkn(lr, batch_count):
                             
        log.info( "Batch count " + str(batch_count) )

        [[X, Y, U], bad_tr_files, [[tr_feats_e, tr_feats_r, lab], tr_idx ], it_batch_nb, it_ctrl_nb]     = it_tr_que.get_batch()
        assert(batch_count == it_batch_nb)

        log.debug("Batch size: " + str(tr_feats_e.shape) + " " + str(tr_feats_r.shape) + " " + str(lab.shape))
        
        WGT_m = np.ones(Y.shape)/ float(Y.shape[0])

        tr_output = train_function(tr_feats_e, tr_idx, WGT_m, Y, tr_feats_r, lab, lr)
        if ( len(tr_output) == 5 ):
            loss_c, loss_c_chn, loss_reco, Loss, _ = tr_output 
            log.info("loss_c=%f, loss_reco=%f, Loss=%f", loss_c, loss_reco,  Loss )
        else:
            loss_c, loss_c_chn, loss_reco, Loss_1, _, loss_GB = tr_output 
            log.info("loss_c=%f, loss_c_chn=%f, loss_reco=%f, loss_GB=%f, Loss_1=%f", loss_c, loss_c_chn, loss_reco, loss_GB, Loss_1)
        if (ret_loss == "loss_reco"):
            return loss_reco
        elif(ret_loss == "loss_c"):
            return loss_c
        else:
            return Loss_1
        
    return train_batch_fkn



def get_train_batch_multi_fkn_backend( it_tr_que, train_function):                           

    
    def train_batch_fkn(lr, batch_count):
                             
        log.info( "Batch count " + str(batch_count) )

        [[X, Y, U], bad_tr_files, [tr_feats, tr_idx ], it_batch_nb, it_ctrl_nb]     = it_tr_que.get_batch()
        assert(batch_count == it_batch_nb)

        log.debug("Batch size: " + str(tr_feats.shape))
        
        WGT_m = np.ones(Y.shape)/ float(Y.shape[0])

        out  = train_function(tr_feats, tr_idx, WGT_m, Y, lr, batch_count)
        info = out[0]
        L = out[1]
        log.info( "Total loss: " + str(L) )
        log.info( "Info: " + str(info) )
        return L
    return train_batch_fkn
