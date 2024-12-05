
import torch

from torch import Tensor
from torch.utils.data import Dataset, Subset

from data_utils.datasets import TensorDataset

class ResultEvaluator:

    def __init__(
            self, 
            dataset: TensorDataset, 
            subsets: dict[str, Subset], 
            split_kind: str = 'label',
            composed: bool = False
        ):

        self.dataset = dataset
        self.split_kind = split_kind
        self.composed = composed
        self.model_results = {}

        if split_kind == 'label':
            test_dataset_l = subsets['test_labeled']
            test_dataset_ul = subsets['test_unlabeled']

            X_test_l = dataset.X_data[test_dataset_l.indices]
            y_test_l = dataset.y_data[test_dataset_l.indices]

            X_test_ul = dataset.X_data[test_dataset_ul.indices]

            self.X_test_l = X_test_l[:, 1:]
            self.y_test_l = y_test_l[:, 1:]

            self.X_test_ul = X_test_ul[:, 1:]
        
        else:
            test_dataset = subsets['test_unlabeled']
            X_test_ul = dataset.X_data[test_dataset.indices]
            self.X_test_ul = X_test_ul[:, 1:]


    def obtain_model_returns(self, model):

        pass

    
    def _obtain_AE_returns(self, model):

        with torch.no_grad():

            Z_batch_ul, X_test_ul_hat = model(self.X_test_ul)
        
        return Z_batch_ul, X_test_ul_hat

    
    



def eval_GVAE_iso(dataset: Dataset, subsets: dict[str, Subset], model, eval_loss, n_epochs, n_iterations):

    test_dataset = subsets['test_unlabeled']
    X_test = dataset.X_data[test_dataset.indices]
    X_test = X_test[:, 1:]

    with torch.no_grad():

        Z_batch, infrm_dist_params, genm_dist_params = model(X_test)

        mu_l, logvar_l = infrm_dist_params.unbind(dim = -1)
        mu_r, logvar_r = genm_dist_params.unbind(dim = -1)

        X_test_hat = mu_r

        loss_eval = eval_loss(X_batch = X_test, X_hat_batch = X_test_hat)
    
    print(
        f'After {n_epochs} epochs with {n_iterations} iterations each\n'
        f'Avg. Loss on mean reconstruction in testing subset: \n{loss_eval.mean()}\n'
    )




def eval_BVAE_iso(dataset: Dataset, subsets: dict[str, Subset], model, eval_loss, n_epochs, n_iterations):

    test_dataset = subsets['test_unlabeled']
    X_test = dataset.X_data[test_dataset.indices]
    X_test = X_test[:, 1:]

    with torch.no_grad():

        Z_batch, infrm_dist_params, genm_dist_params = model(X_test)

        mu_l, logvar_l = infrm_dist_params.unbind(dim = -1)
        alpha_r, beta_r = genm_dist_params.unbind(dim = -1)

        X_test_hat = alpha_r / (alpha_r + beta_r)

        loss_eval = eval_loss(X_batch = X_test, X_hat_batch = X_test_hat)
    
    print(
        f'After {n_epochs} epochs with {n_iterations} iterations each\n'
        f'Avg. Loss on mean reconstruction in testing subset: \n{loss_eval.mean()}\n'
    )




def eval_AE_iso(dataset: Dataset, subsets: dict[str, Subset], model, eval_loss, n_epochs, n_iterations):

    test_dataset = subsets['test_unlabeled']
    X_test = dataset.X_data[test_dataset.indices]
    X_test = X_test[:, 1:]

    with torch.no_grad():

        Z_batch_hat, X_test_hat = model(X_test)

        loss_eval = eval_loss(X_batch = X_test, X_hat_batch = X_test_hat)
    
    print(
        f'After {n_epochs} epochs with {n_iterations} iterations each\n'
        f'Avg. Loss on mean reconstruction in testing subset: \n{loss_eval.mean()}\n'
    )




def eval_AE_Regr_joint(
        dataset: Dataset, 
        subsets: dict[str, Subset], 
        ae_model, regr_model, 
        ae_eval_loss, regr_eval_loss, 
        n_epochs, 
        n_iterations_ae,
        n_iterations_regr,
    ):

    ae_test_ds = subsets['test_unlabeled']
    regr_test_ds = subsets['test_labeled']

    X_test_ul = dataset.X_data[ae_test_ds.indices]

    X_test_l = dataset.X_data[regr_test_ds.indices]
    y_test_l = dataset.y_data[regr_test_ds.indices]

    X_test_ul = X_test_ul[:, 1:]

    X_test_l = X_test_l[:, 1:]
    y_test_l = y_test_l[:, 1:]

    with torch.no_grad():

        Z_batch_ul, X_test_ul_hat = ae_model(X_test_ul)

        loss_reconst = ae_eval_loss(X_batch = X_test_ul, X_hat_batch = X_test_ul_hat)

        Z_batch_l, X_test_l_hat = ae_model(X_test_l)
        y_test_l_hat = regr_model(Z_batch_l)

        loss_regr = regr_eval_loss(y_batch = y_test_l, y_hat_batch = y_test_l_hat)

    print(
        f"Autoencoder:\n"
        f"---------------------------------------------------------------\n"
        f"After {n_epochs} epochs with {n_iterations_ae} iterations each\n"
        f"Avg. Loss on unlabelled testing subset: {loss_reconst}\n\n"
        
        f"Regression End-To-End:\n"
        f"---------------------------------------------------------------\n"
        f"After {n_epochs} epochs with {n_iterations_regr} iterations each\n"
        f"Avg. Loss on labelled testing subset: {loss_regr}\n"
    )



def eval_VAE_Regr_joint(
        dataset: Dataset, 
        subsets: dict[str, Subset], 
        ae_model, regr_model, 
        ae_eval_loss, regr_eval_loss, 
        n_epochs, 
        n_iterations_ae,
        n_iterations_regr,
    ):

    ae_test_ds = subsets['test_unlabeled']
    regr_test_ds = subsets['test_labeled']

    X_test_ul = dataset.X_data[ae_test_ds.indices]

    X_test_l = dataset.X_data[regr_test_ds.indices]
    y_test_l = dataset.y_data[regr_test_ds.indices]

    X_test_ul = X_test_ul[:, 1:]

    X_test_l = X_test_l[:, 1:]
    y_test_l = y_test_l[:, 1:]

    with torch.no_grad():
        
        #--- Apply VAE to labelled and unlabelled data ---#
        Z_batch_l, infrm_dist_params_l, genm_dist_params_l = ae_model(X_test_l)
        Z_batch_ul, infrm_dist_params_ul, genm_dist_params_ul = ae_model(X_test_ul)

        #--- Reconstruction  ---#
        mu_r, _ = genm_dist_params_ul.unbind(dim = -1)
        X_test_ul_hat = mu_r

        #--- Regression ---#
        y_test_l_hat = regr_model(Z_batch_l)

        #--- Loss ---#
        loss_reconst = ae_eval_loss(X_batch = X_test_ul, X_hat_batch = X_test_ul_hat)
        loss_regr = regr_eval_loss(y_batch = y_test_l, y_hat_batch = y_test_l_hat)

    print(
        f"Autoencoder:\n"
        f"---------------------------------------------------------------\n"
        f"After {n_epochs} epochs with {n_iterations_ae} iterations each\n"
        f"Avg. Loss on unlabelled testing subset: {loss_reconst}\n\n"
        
        f"Regression End-To-End:\n"
        f"---------------------------------------------------------------\n"
        f"After {n_epochs} epochs with {n_iterations_regr} iterations each\n"
        f"Avg. Loss on labelled testing subset: {loss_regr}\n"
    )