from abc import ABCMeta, abstractmethod
from client import client
import torch

class server(client):
    def release_p_value(self, data_y, data_z, n_permutation):
        return(self._permutation_test(data_y, data_z, n_permutation))

    def _permutation_test(self, data_y, data_z, n_permutation):
        n_1 = self._get_sample_size(data_y)
        n_2 = self._get_sample_size(data_z) 
        n = n_1 + n_2
        tst_data_combined = torch.vstack((data_y, data_z))
       
        u_stat_original = self._calculate_statistic(data_y, data_z) #original statistic
        #print(f"original u-statistic:{u_stat_original}")
        
        #permutation procedure
        u_stat_permuted = torch.empty(n_permutation).to(self.cuda_device)
        
        for i in range(n_permutation):
            permutation = torch.randperm(n)
            perm_stat_now = self._calculate_statistic(
                tst_data_combined[permutation][:n_1,:],
                tst_data_combined[permutation][n_1:,:]
            ).to(self.cuda_device)
            u_stat_permuted[i] = perm_stat_now

        #print(u_stat_permuted)      
        p_value_proxy = (1 +
                         torch.sum(
                             torch.gt(input = u_stat_permuted, other = u_stat_original)
                         )
                        ) / (n_permutation + 1)
      
        return(p_value_proxy)
    
    @abstractmethod   
    def _calculate_statistic(self, data_y, data_z):
        raise NotImplementedError()
    
class server_twosample_U(server):    
    def _calculate_statistic(self, data_y, data_z):
        n_1 = torch.tensor(self._get_sample_size(data_y))
        n_2 = torch.tensor(self._get_sample_size(data_z))
    
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
        abs_u_y = torch.exp(torch.log(torch.abs(one_Phi_tilde_one)) - torch.log(n_1) - torch.log(n_1 - 1) )
        u_y = sign_y * abs_u_y


        # z only part. log calculation in case of large n2
        sign_z = torch.sign(one_Psi_tilde_one)

        abs_u_z = torch.exp(torch.log(torch.abs(one_Psi_tilde_one)) - torch.log(n_2) - torch.log(n_2 - 1) )
        u_z = sign_z * abs_u_z

        # cross part
        cross = torch.inner(y_row_sum, z_row_sum)
        sign_cross = torch.sign(cross)
        abs_cross = torch.exp(torch.log(torch.abs(cross)) +torch.log(torch.tensor(2))- torch.log(n_1) - torch.log(n_2) )
        u_cross = sign_cross * abs_cross

        return(u_y + u_z - u_cross)