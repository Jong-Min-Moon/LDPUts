from abc import ABC, ABCMeta, abstractmethod
from client import client
import utils
import torch


import gc

class server(ABC):
    def __init__(self, privacy_level):
        

        self.privacy_level = privacy_level
        self.n_1 = torch.tensor(0)
        self.n_2 = torch.tensor(0)
        
    def load_private_data_multinomial(self, data_y, data_z, alphabet_size, device_y, device_z):
        self.alphabet_size = alphabet_size
        self.n_1 = torch.tensor(utils.get_sample_size(data_y))
        self.n_2 = torch.tensor(utils.get_sample_size(data_z))
        self.n = self.n_1 + self.n_2
        self.scaling_constant = 1/(1/ self.n_1 + 1/ self.n_2)
        self.chisq_distribution = torch.distributions.chi2.Chi2(
            torch.tensor(self.alphabet_size - 1)
        )
        self.cuda_device_y = device_y
        self.cuda_device_z = device_z

    def push_data_to_gpu(self):
        self.data_y = self.data_y.to(self.cuda_device_y)
        self.data_z = self.data_z.to(self.cuda_device_z)


    def release_p_value_permutation(self, n_permutation):
            
        original_statistic = self.get_original_statistic()
        permuted_statistic_vec = torch.empty(n_permutation).to(self.cuda_device_y)
       
        for i in range(n_permutation):   
            permuted_statistic_vec[i] = self._get_statistic( torch.randperm(self.n_1 + self.n_2) )
      
        return(self.get_p_value_proxy(permuted_statistic_vec, original_statistic), float(original_statistic))
    
    def get_original_statistic(self):
       original_statistic = self._get_statistic( torch.arange(self.n_1 + self.n_2) )
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
        return(float(p_value_proxy))
    
    def delete_data(self):
        del self.data_y
        
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

  
        
        


class server_ell2(server):
    def load_private_data_multinomial(self, data_y, data_z, alphabet_size, device_y, device_z):
        super().load_private_data_multinomial(data_y, data_z, alphabet_size, device_y, device_z)
        if utils.get_dimension(data_y) == 1:
            self.data_y = torch.nn.functional.one_hot( data_y , self.alphabet_size).double()
        if utils.get_dimension(data_z) == 1:
            self.data_z = torch.nn.functional.one_hot( data_z , self.alphabet_size).double()
        else:
            self.data_y = data_y.double()
            self.data_z = data_z.double()
        self.data_y_square_colsum = self.data_y.square().sum(1)
        self.data_z_square_colsum = self.data_z.square().sum(1)

        self.push_data_to_gpu()

    import torch

    def _get_statistic(self, perm):
        # Split permutation
        perm_toY_fromY, perm_toY_fromZ, perm_toZ_fromY, perm_toZ_fromZ = utils.split_perm(perm, self.n_1)
        
        # Get row sums
        y_row_sum = self.get_sum_y(perm).to(torch.float64)
        z_row_sum = self.get_sum_z(perm).to(torch.float64)
        
        # Calculate y_sqrsum and z_sqrsum
        y_sqrsum = (self.data_y_square_colsum[perm_toY_fromY].sum().to(self.cuda_device_z).to(torch.float64)
                    + self.data_z_square_colsum[perm_toY_fromZ].sum().to(torch.float64))
        
        z_sqrsum = (self.data_y_square_colsum[perm_toZ_fromY].sum().to(self.cuda_device_z).to(torch.float64)
                    + self.data_z_square_colsum[perm_toZ_fromZ].sum().to(torch.float64))
        
        # Calculate dot products
        one_Phi_one = y_row_sum.dot(y_row_sum).to(torch.float64)
        one_Psi_one = z_row_sum.dot(z_row_sum).to(torch.float64)
        cross = y_row_sum.dot(z_row_sum).to(torch.float64)
        
        # Calculate tilde values
        one_Phi_tilde_one = one_Phi_one - y_sqrsum
        one_Psi_tilde_one = one_Psi_one - z_sqrsum
        
        # Convert n_1 and n_2 to float64
        n_1 = self.n_1.to(torch.float64)
        n_2 = self.n_2.to(torch.float64)
        
        # y only part: log calculation in case of large n1
        sign_y = torch.sign(one_Phi_tilde_one)
        log_abs_u_y = (torch.log(torch.abs(one_Phi_tilde_one))
                    - torch.log(n_1)
                    - torch.log(n_1 - 1))
        u_y = sign_y * torch.exp(log_abs_u_y)
        
        # z only part: log calculation in case of large n2
        sign_z = torch.sign(one_Psi_tilde_one)
        log_abs_u_z = (torch.log(torch.abs(one_Psi_tilde_one))
                    - torch.log(n_2)
                    - torch.log(n_2 - 1))
        u_z = sign_z * torch.exp(log_abs_u_z)
        
        # cross part
        sign_cross = torch.sign(cross)
        log_abs_cross = (torch.log(torch.abs(cross))
                        + torch.log(torch.tensor(2.0, device=cross.device, dtype=torch.float64))
                        - torch.log(n_1)
                        - torch.log(n_2))
        u_cross = sign_cross * torch.exp(log_abs_cross)
        
        # Calculate the final statistic
        statistic = u_y + u_z - u_cross
        
        return statistic



class server_multinomial_genrr(server):
    def load_private_data_multinomial(self, data_y, data_z, alphabet_size, device_y, device_z):
        super().load_private_data_multinomial(data_y, data_z, alphabet_size, device_y, device_z);
        self.data_y = torch.nn.functional.one_hot( data_y , self.alphabet_size).float()
        self.data_z = torch.nn.functional.one_hot( data_z , self.alphabet_size).float()


        self.mean_recip_est = self.get_grand_mean().reciprocal().to(self.cuda_device_y)
        self.mean_recip_est[self.mean_recip_est.isinf()] = 0

        self.push_data_to_gpu()

    def _get_statistic(self, perm):
        mu_hat_diff_square = self.get_mean_diff(perm).square()
        self.grand_mean = self.get_grand_mean().to(self.cuda_device_y)
        mean_recip_est = self.grand_mean.reciprocal()
        mean_recip_est[mean_recip_est.isinf()] = 0

        statistic = mu_hat_diff_square.mul(
            mean_recip_est  
            ).mul(self.scaling_constant).sum()
        return(statistic)

    def release_p_value(self):
        test_stat = self.get_original_statistic()
        print(self.chisq_distribution.df)
        print(test_stat)     
        return(1 - self.chisq_distribution.cdf(test_stat))

    def release_p_value_permutation(self, n_permutation):
        mu_hat_diff_mat = torch.empty([self.alphabet_size, (n_permutation+1) ]).to(self.cuda_device_y)
       
        for i in range(n_permutation):
            mu_hat_diff_mat[:,i] = self.get_mean_diff( torch.randperm(self.n_1 + self.n_2) )
        mu_hat_diff_mat[:,n_permutation] =  self.get_mean_diff( torch.arange(self.n) )
        self.delete_data

        mu_hat_diff_square_mat = mu_hat_diff_mat.square()
        permuted_statistic_vec = torch.mul(mu_hat_diff_square_mat , self.mean_recip_est.unsqueeze(1)).sum(dim=0).mul(self.scaling_constant)        
        return(
            self.get_p_value_proxy(
                permuted_statistic_vec[:n_permutation],
                permuted_statistic_vec[n_permutation]
                ),
                float(permuted_statistic_vec[n_permutation])
        )

    def get_mean_diff(self, perm):
        
        mean_y = self.get_mean_y(perm)
        mean_y = mean_y.to(self.cuda_device_y)

        mean_z = self.get_mean_z(perm)
        mean_z = mean_z.to(self.cuda_device_y)
      
        mu_hat_diff =  torch.sub(mean_y, mean_z)
        return(mu_hat_diff)

    def get_grand_mean(self):
        perm_toY_fromY, perm_toY_fromZ, _, _ = utils.split_perm(torch.arange(self.n), self.n_1)
        sum_y = self.data_y[perm_toY_fromY].sum(0).add( self.data_z[perm_toY_fromZ].sum(0) ).to(torch.float)
        _, _, perm_toZ_fromY, perm_toZ_fromZ = utils.split_perm(torch.arange(self.n), self.n_1)   
        sum_z = self.data_y[perm_toZ_fromY].sum(0).add( self.data_z[perm_toZ_fromZ].sum(0) ).to(torch.float)
        grand_mean = torch.add(sum_z, sum_y).div(self.n).to(torch.float)
        return(grand_mean)

class server_multinomial_bitflip(server_multinomial_genrr):
    def load_private_data_multinomial(self, data_y, data_z, alphabet_size, device_y, device_z):
        self.alphabet_size = alphabet_size
        self.n_1 = torch.tensor(utils.get_sample_size(data_y))
        self.n_2 = torch.tensor(utils.get_sample_size(data_z))
        self.n = self.n_1 + self.n_2
        self.scaling_constant = 1/(1/ self.n_1 + 1/ self.n_2)
        self.chisq_distribution = torch.distributions.chi2.Chi2(
            torch.tensor(self.alphabet_size - 1)
        )
        self.data_y = data_y
        self.data_z = data_z
        self.cuda_device_y = device_y
        self.cuda_device_z = device_z
        self.get_cov_est()


        self.push_data_to_gpu()
        self.proj = self.get_proj_orth_one_space()

    def get_cov_est(self):        
        self.grand_mean = self.get_grand_mean()
        self.cov_est = torch.matmul(
            torch.transpose( self.data_y.sub(self.grand_mean),0,1 ),
            self.data_y.sub(self.grand_mean)
        )
        cov_est_z = torch.matmul(
            torch.transpose( self.data_z.sub(self.grand_mean),0,1 ),
            self.data_z.sub(self.grand_mean )
                )
        self.cov_est_L =  torch.cholesky(
            self.cov_est.add(cov_est_z).div(self.n-1).to(torch.float)
            )
        self.cov_est_L = self.cov_est_L.to(self.cuda_device_y)
    def _get_statistic(self, perm):
        
        proj_mu_hat_diff = torch.mv(
            self.proj,
            self.get_mean_diff(perm)
        )
        statistic = torch.dot(
            proj_mu_hat_diff,
            torch.cholesky_solve(
                proj_mu_hat_diff.reshape(-1,1), 
                self.cov_est_L).flatten()
        ).mul(self.scaling_constant)
        return(statistic)

    def get_proj_orth_one_space(self):
        matrix_iden = torch.eye(self.alphabet_size)
        one_one_t = torch.ones( torch.Size([self.alphabet_size, self.alphabet_size]) )
        one_one_t_over_d = one_one_t.div(self.alphabet_size)
        one_projector = matrix_iden.sub(one_one_t_over_d)
        one_projector = one_projector.to(torch.float).to(self.cuda_device_y)
        return(one_projector)
        
    def release_p_value_permutation(self, n_permutation):
        mu_hat_diff_mat = torch.empty([self.alphabet_size, (n_permutation+1) ]).to(self.cuda_device_y)
       
        for i in range(n_permutation):
            mu_hat_diff_mat[:,i] = self.get_mean_diff( torch.randperm(self.n_1 + self.n_2) )
        mu_hat_diff_mat[:,n_permutation] =  self.get_mean_diff( torch.arange(self.n) )
        self.delete_data

        self.cov_est = self.cov_est.to(self.cuda_device_y)
        proj_mu_hat_diff_mat = torch.mm(self.proj, mu_hat_diff_mat)
        Sigma_inv_proj_mu_hat_diff_mat = torch.cholesky_solve(proj_mu_hat_diff_mat, self.cov_est_L)
        permuted_statistic_vec = torch.mul(proj_mu_hat_diff_mat, Sigma_inv_proj_mu_hat_diff_mat).sum(dim=0).mul(self.scaling_constant)
        return(
            self.get_p_value_proxy(
                permuted_statistic_vec[:n_permutation],
                permuted_statistic_vec[n_permutation]
                ),
                float(permuted_statistic_vec[n_permutation])
        )
    

       




 


