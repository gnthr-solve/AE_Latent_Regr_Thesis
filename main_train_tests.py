###--- External Library Imports ---###
import torch


###--- Training Functions ---###
from training.training_funcs import *



"""
Main Executions
-------------------------------------------------------------------------------------------------------------------------------------------
"""
if __name__=="__main__":

    # check computation backend to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-device:", device)

    ###--- AE in isolation ---###
    #AE_iso_training_procedure()
    

    ###--- VAE in isolation ---###
    #train_VAE_iso()
    #VAE_iso_training_procedure()
    #VAE_latent_visualisation()


    ###--- Compositions ---###
    #train_joint_seq_AE()
    #train_joint_seq_VAE()
    #train_seq_AE()
    #train_joint_epoch_wise_VAE()
    #train_joint_epoch_wise_VAE_recon()
    #AE_joint_epoch_procedure()
    #VAE_joint_epoch_procedure()


    ###--- Baseline ---###
    #train_linear_regr()
    #train_deep_regr()


    ###--- Testing ---###
    #AE_regr_loss_tests()
    AE_regr_loss_tests2()

    pass