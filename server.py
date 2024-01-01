from abc import ABC, ABCMeta, abstractmethod
from client import client
import utils
import torch
from scipy.stats import chi2
import numpy

class server(ABC):
    def __init__(self, cuda_device, privacy_level):
        self.cuda_device = cuda_device
        self.privacy_level = privacy_level

    def load_private_data_multinomial(self, data_y, data_z, alphabet_size):
        self.n_1 = torch.tensor(utils.get_sample_size(data_y)).to(self.cuda_device)
        self.n_2 = torch.tensor(utils.get_sample_size(data_z)).to(self.cuda_device)
        self.alphabet_size = alphabet_size.to(self.cuda_device)
        self.chisq_distribution = chi2(self.alphabet_size - 1)

        dim_1 = utils.get_dimension(data_y)
        dim_2 = utils.get_dimension(data_z)       
        if dim_1 != dim_2:
            raise Exception("different data dimensions")
        else:
            self.save_data(data_y, data_z)
    
    @abstractmethod
    def save_data(self, data_y, data_z):
        pass

    def release_p_value_permutation(self, n_permutation):       
        original_statistic = self.get_original_statistic()
        permuted_statistic_vec = torch.empty(n_permutation).to(self.cuda_device)
       
        for i in range(n_permutation):
            permutation = torch.randperm(self.n_1 + self.n_2)
            permuted_statistic_vec[i] = self._calculate_statistic(
                permutation[torch.arange(self.n_1)],
                permutation[torch.arange(self.n_1, self.n_1 + self.n_2)]
            )
      
        return(self.get_p_value_proxy(permuted_statistic_vec, original_statistic))
    
    def get_original_statistic(self):
       original_statistic = self._get_statistic(
           torch.arange(self.n_1),
           torch.arange(self.n_1, self.n_1 + self.n_2)
           )
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
    
class server_LapU(server):
    def save_data(self, data_y, data_z):
        self.data = torch.vstack( (data_y, data_z) ).to(self.cuda_device)           

    def _get_statistic(self, idx_1, idx_2):
        data_y = self.data[idx_1]
        data_z = self.data[idx_2]

        y_row_sum = torch.sum(data_y, axis = 0)
        z_row_sum = torch.sum(data_z, axis = 0)

        one_Phi_one = torch.inner(y_row_sum, y_row_sum)
        one_Psi_one = torch.inner(z_row_sum, z_row_sum)

        tr_Phi = torch.sum(torch.square(data_y))
        tr_Psi = torch.sum(torch.square(data_z))

        one_Phi_tilde_one = one_Phi_one - tr_Phi
        one_Psi_tilde_one = one_Psi_one - tr_Psi


        # y only part. log calculation in case of large n1
        sign_y = torch.sign(one_Phi_tilde_one)
        abs_u_y = torch.exp(torch.log(torch.abs(one_Phi_tilde_one)) - torch.log(self.n_1) - torch.log(self.n_1 - 1) )
        u_y = sign_y * abs_u_y


        # z only part. log calculation in case of large n2
        sign_z = torch.sign(one_Psi_tilde_one)

        abs_u_z = torch.exp(torch.log(torch.abs(one_Psi_tilde_one)) - torch.log(self.n_2) - torch.log(self.n_2- 1) )
        u_z = sign_z * abs_u_z

        # cross part
        cross = torch.inner(y_row_sum, z_row_sum)
        sign_cross = torch.sign(cross)
        abs_cross = torch.exp(torch.log(torch.abs(cross)) +torch.log(torch.tensor(2))- torch.log(self.n_1) - torch.log(nself.n_2) )
        u_cross = sign_cross * abs_cross

        return(u_y + u_z - u_cross)
<<<<<<< HEAD
    
class server_twosample_chi(server):
    def load_private_data(self, data_y, data_z):      
        n_1 = torch.tensor(utils.get_sample_size(data_y))
        n_2 = torch.tensor(utils.get_sample_size(data_z))
        alphabet_size_1 = utils.get_dimension(data_y)
        alphabet_size_2 = utils.get_dimension(data_z)
        if n_1 != n_2:
            raise Exception("sample size from each group must be the same)")
        elif alphabet_size_1 != alphabet_size_2:
            raise Exception("Alphabet sizes of each group must be the same)")
        #elif (self.not_multinomial(data_y)) or (self.not_multinomial(data_z)):
        #    raise Exception("only accepts multinomial data (torch.int64 or torch.long)")
            
        self.data_y = data_y
        self.data_z = data_z
        self.sample_size = n_1.to(self.cuda_device)
        self.alphabet_size = alphabet_size_1
        self.chisq_distribution = chi2(self.alphabet_size - 1) 

    def release_p_value(self):
        test_stat = self._calculate_statistic(self.data_y, self.data_z).cpu().numpy().item()      
        return(self.chisq_distribution.sf(test_stat))

    def not_multinomial(self, data):
        if data.dtype != torch.int64:
            return(True)
        elif data.dtype != torch.long:
            return(True)
        else:
            return(False)
        
class server_twosample_genRR(server_twosample_chi):
    def load_private_data(self, data_y, data_z, alphabet_size):      
        n_1 = torch.tensor(utils.get_sample_size(data_y))
        n_2 = torch.tensor(utils.get_sample_size(data_z))
            
        self.data_y = data_y
        self.data_z = data_z
        self.sample_size = n_1.to(self.cuda_device)
        self.alphabet_size = alphabet_size
        self.chisq_distribution = chi2(self.alphabet_size - 1) 

    def release_p_value_permutation(self, n_permutation):
        n_1 = utils.get_sample_size(self.data_y)
        n_2 = utils.get_sample_size(self.data_z)
        n = n_1 + n_2
        tst_data_combined = torch.cat((self.data_y, self.data_z))
       
        stat_original = self._calculate_statistic(self.data_y, self.data_z) #original statistic
        #print(f"original u-statistic:{u_stat_original}")
        
        #permutation procedure
        stat_permuted = torch.empty(n_permutation).to(self.cuda_device)
        
        for i in range(n_permutation):
            permutation = torch.randperm(n)
            perm_stat_now = self._calculate_statistic(
                tst_data_combined[permutation][:n_1],
                tst_data_combined[permutation][n_1:]
            ).to(self.cuda_device)
            stat_permuted[i] = perm_stat_now

        #print(u_stat_permuted)      
        p_value_proxy = (1 +
                         torch.sum(
                             torch.gt(input = stat_permuted, other = stat_original)
                         )
                        ) / (n_permutation + 1)
      
        return(p_value_proxy)
    def _calculate_statistic(self, data_y, data_z):
        # n_1 = torch.tensor(utils.get_sample_size(data_y))
        # n_2 = torch.tensor(utils.get_sample_size(data_z))
        
        #load data
        n_1 = self.sample_size
        n_2 = n_1
        Y_count = data_y.bincount(minlength=self.alphabet_size)
        Z_count = data_z.bincount(minlength=self.alphabet_size)
        
        #calculation
        total_count = Y_count + Z_count
        total_count_nonzero = total_count[total_count>0]
        Y_count_nonzero = Y_count[total_count>0]
        Z_count_nonzero = Z_count[total_count>0]
        count_diff_square = torch.sub(
            Y_count_nonzero.mul(n_2),
            Z_count_nonzero.mul(n_1)
        ).square()
        count_diff_square_normalized =  count_diff_square.divide(
                total_count_nonzero.mul(n_1).mul(n_2)
            )
        test_statistic = count_diff_square_normalized.sum()
        return(test_statistic)

 
class server_twosample_bitflip(server_twosample_chi):  
    def _calculate_statistic(self, data_y, data_z):
        mean_y = data_y.mean(axis=0).view([self.alphabet_size,1])
        mean_z = data_z.mean(axis=0).view([self.alphabet_size,1])
        cov_sum = torch.cov(data_y.T).add(torch.cov(data_z.T))
        test_statistic = self._calculate_bitflip_statistic(mean_y, mean_z, cov_sum, self.alphabet_size, self.sample_size)
        return(test_statistic)
        
    def _calculate_bitflip_statistic(self, mean_1, mean_2, cov_sum, alphabet_size, n):
        one_projector = utils.projection_orth_one(alphabet_size).to(self.cuda_device)
        mean_diff = mean_1.sub(mean_2)
=======



class server_multinomial_bitflip(server):
    def save_data(self, data_y, data_z):
        self.data = torch.vstack( (data_y, data_z) ).to(self.cuda_device)

    def _get_statistic(self, idx_1, idx_2):
        data_y = self.data[idx_1]
        data_z = self.data[idx_2]

        mu_hat_y = data_y.mean(axis=0).view([self.alphabet_size,1]) #column vector
        mu_hat_z = data_z.mean(axis=0).view([self.alphabet_size,1]) #column vector
        mu_hat_diff = torch.sub(mu_hat_y, mu_hat_z)
        scaling_matrix = self._get_scaling_matrix()
        scaling_constant = torch.reciprocal( torch.add( self.n_1.reciprocal(), self.n_2.reciprocal() ) )
        statistic = mu_hat_diff.T.matmul(scaling_matrix).matmul(mu_hat_diff).mul(scaling_constant)
        return(statistic)

    def _get_scaling_matrix(self)
        cov_est = torch.cov(self.data.T)
        mat_proj = self.get_proj_orth_one_space()
>>>>>>> f117be6494ab04651dcf973c34fce5639a55bf1b
        if self.cuda_device.type== "cpu":
            prec_mat_est =  torch.tensor(numpy.linalg.inv(cov_est.numpy())) 
        else:
            prec_mat_est =  torch.linalg.inverse(cov_est)
        return(
            mat_proj.matmul(prec_mat_est).matmul(mat_proj)
        )
    
    def get_proj_orth_one_space(self):
        matrix_iden = torch.eye(self.alphabet_size)
        one_one_t = torch.ones( torch.Size([self.alphabet_size, self.alphabet_size]) )
        one_one_t_over_d = one_one_t.div(self.alphabet_size)
        one_projector = matrix_iden.sub(one_one_t_over_d)
        return(one_projector)


        
 class server_multinomial_genRR(server_multinomial_bitflip):
    def save_data(self, data_y, data_z):
            self.data = torch.cat( (data_y, data_z) ).to(self.cuda_device)
            self.data = torch.nn.functional.one_hot(self.data, self.alphabet_size)

    def _get_scaling_matrix(self)
        mean_recip_est = self.data.mean(axis=0).reciprocal()
        mean_recip_est[mean_recip_est.isinf()] = 0
        return(torch.diag(mean_recip_est))



 


