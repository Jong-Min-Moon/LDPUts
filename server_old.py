from abc import ABC, ABCMeta, abstractmethod
from client import client
import utils
import torch
from scipy.stats import chi2
import numpy
import gc

class server(ABC):
    def __init__(self, privacy_level):
        

        self.privacy_level = privacy_level
        self.n_1 = torch.tensor(0)
        self.n_2 = torch.tensor(0)

    def load_private_data_multinomial_y(self, data_y, alphabet_size):
        self.n_1 = torch.tensor(utils.get_sample_size(data_y))
        self.alphabet_size = alphabet_size
        self.chisq_distribution = torch.distributions.chi2.Chi2(
            torch.tensor(self.alphabet_size - 1)
        )

 
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
        self.n = self.n_1 + self.n_2
        
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
       print(original_statistic)
       return(original_statistic)

    @abstractmethod
    def _get_statistic(self, perm):
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

    def get_sum_y(self, perm ):
        perm_toY_fromY, perm_toY_fromZ, _, _ = utils.split_perm(perm, self.n_1)
        return (self.data_y[perm_toY_fromY].sum(0).to(self.cuda_device_z ).add( self.data_z[perm_toY_fromZ].sum(0) )).to(torch.float)
    
    def get_sum_z(self, perm ):
        _, _, perm_toZ_fromY, perm_toZ_fromZ = utils.split_perm(perm, self.n_1)
        return (self.data_y[perm_toZ_fromY].sum(0).to(self.cuda_device_z ).add( self.data_z[perm_toZ_fromZ].sum(0) )).to(torch.float)
    
    def get_mean_y(self, perm):
        return self.get_sum_y(perm).div(self.n_1).to(torch.float)
    
    def get_mean_z(self, perm):
        return self.get_sum_z(perm).div(self.n_2).to(torch.float)

    def get_grand_mean(self):
        return(
            self.get_sum_y(torch.arange(self.n)).add(
                self.get_sum_z(torch.arange(self.n))
            ).div(self.n)
        ).to(torch.float)
  
        
        


class server_ell2(server):
    def _get_statistic(self, perm):
        perm_toY_fromY, perm_toY_fromZ, perm_toZ_fromY, perm_toZ_fromZ = utils.split_perm(perm, self.n_1) 
        y_row_sum = self.get_sum_y(perm)
        z_row_sum = self.get_sum_z(perm)
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
    def load_private_data_multinomial_z(self, data_z, alphabet_size ):
        super().load_private_data_multinomial_z(data_z, alphabet_size); 
        self.grand_mean = self.get_grand_mean()

        self.cov_est = torch.matmul(
            torch.transpose( self.data_y.sub(self.grand_mean),0,1 ),
            self.data_y.sub(self.grand_mean)
        #
        ).add(
        #
        torch.matmul(
            torch.transpose( self.data_z.sub(self.grand_mean),0,1 ),
            self.data_z.sub(self.grand_mean)
                )
        #
        ).div(
        #
        self.n-1
                ).to(torch.float)
    def release_p_value(self):
        test_stat = self.get_original_statistic()
        print(self.chisq_distribution.df)
        print(test_stat)     
        return(1 - self.chisq_distribution.cdf(test_stat))

    
 
       
    def _get_statistic(self, perm):
        proj_mu_hat_diff = torch.mv(
            self.get_proj_orth_one_space(),
            torch.sub(
                self.get_mean_y(perm),
                self.get_mean_z(perm)
            )
        )
        #print(proj_mu_hat_diff) # confirmed equality
        scaling_constant = 1/(1/ self.n_1 + 1/ self.n_2)
        #print(scaling_constant)
        print(self.cov_est)
        statistic = torch.dot(
            proj_mu_hat_diff,
            torch.solve(
                proj_mu_hat_diff.reshape(-1,1), 
                self.cov_est).solution.flatten()
        ).mul(scaling_constant)
        print(statistic)
        return(statistic)
    
    def get_proj_orth_one_space(self):
        matrix_iden = torch.eye(self.alphabet_size)
        one_one_t = torch.ones( torch.Size([self.alphabet_size, self.alphabet_size]) )
        one_one_t_over_d = one_one_t.div(self.alphabet_size)
        one_projector = matrix_iden.sub(one_one_t_over_d)
        return(one_projector.to(torch.float).to(self.cuda_device_z))

       
class server_multinomial_genrr(server_multinomial_bitflip):
    def _get_statistic(self, perm):
        mu_hat_diff_square = torch.sub(
            self.get_mean_y(perm),
            self.get_mean_z(perm)
            ).square()
        
        mean_recip_est = self.grand_mean.reciprocal()
        mean_recip_est[mean_recip_est.isinf()] = 0
        scaling_constant = 1/(1/ self.n_1 + 1/ self.n_2)

        statistic = mu_hat_diff_square.mul(
            mean_recip_est  
            ).mul(scaling_constant).sum()
        return(statistic)



 


