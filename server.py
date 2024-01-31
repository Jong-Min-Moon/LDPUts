from abc import ABC, ABCMeta, abstractmethod
from client import client
import utils
import torch
from scipy.stats import chi2
import numpy
import gc

class server(ABC):
    def __init__(self, privacy_level):
        
        #self.cuda_device_z = cuda_device_z
        self.privacy_level = privacy_level
        self.n_1 = torch.tensor(0)
        self.n_2 = torch.tensor(0)

    def load_private_data_multinomial_y(self, data_y, alphabet_size):
        self.n_1 = torch.tensor(utils.get_sample_size(data_y))
        self.alphabet_size = alphabet_size
        self.chisq_distribution = chi2(self.alphabet_size - 1)
 
        if self.is_integer_form(data_y):
            self.data_y = torch.nn.functional.one_hot( data_y , self.alphabet_size).float()
        else:
            self.data_y = data_y
        
        self.data_y_square_colsum = self.data_y.square().sum(1)
        self.cuda_device_y = data_y.device
        del data_y
        gc.collect()
        torch.cuda.empty_cache()

    def load_private_data_multinomial_z(self, data_z, alphabet_size):
        if not self.is_y_loaded:
            raise Exception("Load Y data first")
        
        if self.alphabet_size != alphabet_size:
            raise Exception("different alphabet sizes")
            
              
        self.n_2 = torch.tensor(utils.get_sample_size(data_z))
        
        if self.is_integer_form(data_z):
            self.data_z = torch.nn.functional.one_hot( data_z , self.alphabet_size).float()
        else:
            self.data_z = data_z

        self.data_z_square_colsum = self.data_z.square().sum(1)
        self.cuda_device_z = data_z.device
        del data_z
        gc.collect()
        torch.cuda.empty_cache()

    def release_p_value_permutation(self, n_permutation):       
        original_statistic = self.get_original_statistic()
        permuted_statistic_vec = torch.empty(n_permutation).to(self.cuda_device_z)
       
        for i in range(n_permutation):
            permuted_statistic_vec[i] = self._get_statistic( torch.randperm(self.n_1 + self.n_2) )
      
        return(self.get_p_value_proxy(permuted_statistic_vec, original_statistic))
    
    def get_original_statistic(self):
       original_statistic = self._get_statistic( torch.arange(self.n_1 + self.n_2) )
       return(original_statistic)

    @abstractmethod
    def _get_statistic(self, idx_1, idx_2):
        return(statistic)
    
    def get_p_value_proxy(self, stat_permuted, stat_original):
        p_value_proxy = (1 +
                         torch.sum(
                             torch.gt(input = stat_permuted, other = stat_original)
                         )
                        ) / (stat_permuted.size(dim = 0) + 1)
        return(p_value_proxy)
    
    def delete_data(self):
        del self.data_y
        4
        del self.data_z  
        gc.collect()
        torch.cuda.empty_cache()

    def is_y_loaded(self):
        return( self.n_1.equal(0) )

    def is_integer_form(self, data):
        return( utils.get_dimension(data) == 1 )

    def get_group_sum(self, perm):
        perm_toY_fromY, perm_toY_fromZ, perm_toZ_fromY, perm_toZ_fromZ = utils.split_perm(perm, self.n_1)

        return(
            self.data_y[perm_toY_fromY].sum(0).to(self.cuda_device_z ).add( self.data_z[perm_toY_fromZ].sum(0) ),
            self.data_y[perm_toZ_fromY].sum(0).to(self.cuda_device_z ).add( self.data_z[perm_toZ_fromZ].sum(0) )
        )
        
        


class server_ell2(server):
    def _get_statistic(self, perm):
        perm_toY_fromY, perm_toY_fromZ, perm_toZ_fromY, perm_toZ_fromZ = utils.split_perm(perm, self.n_1) 
        y_row_sum, z_row_sum =  self.get_group_sum(perm)
        y_sqrsum = self.data_y_square_colsum[perm_toY_fromY].sum().to(self.cuda_device_z ).add(
            self.data_z_square_colsum[perm_toY_fromZ].sum()
        ) # scalar

        z_sqrsum = self.data_y_square_colsum[perm_toZ_fromY].sum().to(self.cuda_device_z ).add(
            self.data_z_square_colsum[perm_toZ_fromZ].sum()
        ) # scalar
        
        one_Phi_one = y_row_sum.dot(y_row_sum) #scalar
        one_Psi_one = z_row_sum.dot(z_row_sum) #scalar
        cross = y_row_sum.dot(z_row_sum) #scalar

        one_Phi_tilde_one = one_Phi_one - y_sqrsum #scalar
        one_Psi_tilde_one = one_Psi_one - z_sqrsum #scalar

        n_1 = self.n_1.to(torch.float)
        n_2 = self.n_2.to(torch.float)
        # y only part. log calculation in case of large n1
        sign_y = torch.sign(one_Phi_tilde_one) #scalar
        abs_u_y = torch.exp(torch.log(torch.abs(one_Phi_tilde_one)) - torch.log(n_1) - torch.log(n_1 - 1) ) #scalar
        u_y = sign_y * abs_u_y #scalar


        # z only part. log calculation in case of large n2
        sign_z = torch.sign(one_Psi_tilde_one) #scalar
        abs_u_z = torch.exp(torch.log(torch.abs(one_Psi_tilde_one)) - torch.log(n_2) - torch.log(n_2- 1) ) #scalar
        u_z = sign_z * abs_u_z #scalar

        # cross part 
        sign_cross = torch.sign(cross) #scalar
        abs_cross = torch.exp(torch.log(torch.abs(cross)) +torch.log(torch.tensor(2).to(torch.float))- torch.log(n_1) - torch.log(n_2) ) #scalar
        u_cross = sign_cross * abs_cross #scalar
        statistic = u_y + u_z - u_cross #scalar
        return(statistic) 



class server_multinomial_bitflip(server):
    def release_p_value(self):
        test_stat = self.get_original_statistic().cpu().numpy().item()      
        return(self.chisq_distribution.sf(test_stat))
    
    def _get_statistic(self, perm):
        y_row_sum, z_row_sum =  self.get_group_sum(perm)

        mu_hat_diff_proj = self.get_proj_orth_one_space().matmul(
            torch.sub(
                y_row_sum.div(self.n_1),
                z_row_sum.div(self.n_2)
                ).view([self.alphabet_size,1])
            )

         
        total_mean =  y_row_sum.add( z_row_sum ).div(self.n_1 + self.n_2)
        cov_est = self.data_y.sub(total_mean).t().matmul(
            self.data_y.sub(total_mean)
            ).to(self.cuda_device_z ).add(
                self.data_z.sub(total_mean).t().matmul(self.data_y.sub(total_mean))
            ).div(self.n_1 + self.n_2-1)
        
        scaling_constant = torch.reciprocal( torch.add( self.n_1.reciprocal(), self.n_2.reciprocal() ) )
        statistic = mu_hat_diff_proj.t().matmul(
            torch.linalg.solve(cov_est, mu_hat_diff_proj)
            ).mul(scaling_constant)
        
        return(statistic)
    
    def get_proj_orth_one_space(self):
        matrix_iden = torch.eye(self.alphabet_size)
        one_one_t = torch.ones( torch.Size([self.alphabet_size, self.alphabet_size]) )
        one_one_t_over_d = one_one_t.div(self.alphabet_size)
        one_projector = matrix_iden.sub(one_one_t_over_d)
        return(one_projector.to(self.cuda_device_z))

        
class server_multinomial_genrr(server_multinomial_bitflip):
    def save_data(self, data_y, data_z):
        self.data = torch.nn.functional.one_hot(
                torch.cat( (data_y, data_z) ).to(self.cuda_device),
                self.alphabet_size
                ).float()
        del data_y
        del data_z

    def _get_scaling_matrix(self):
        mean_recip_est = self.data.mean(axis=0).reciprocal()
        mean_recip_est[mean_recip_est.isinf()] = 0
        return(torch.diag(mean_recip_est))



 


