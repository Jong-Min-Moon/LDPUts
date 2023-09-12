from abc import ABCMeta, abstractmethod
class server(client):

    def permutation_test(self, data_y, data_z, n_permutation):
        n_1 = self._get_sample_size(data_y)
        n_2 = self._get_sample_size(data_z) 
        n = n_1 + n_2
        tst_data_combined = torch.vstack((data_y, data_z))
       
        u_stat_original = self.calculate_statistic(data_y, data_z) #original statistic
        #print(f"original u-statistic:{u_stat_original}")
        
        #permutation procedure
        u_stat_permuted = torch.empty(n_permutation).to(self.cuda_device)
        
        for i in range(n_permutation):
            permutation = torch.randperm(n)
            perm_stat_now = self.calculate_statistic(
                tst_data_combined[permutation][:n_1,:],
                tst_data_combined[permutation][n_1:,:]
            ).to(self.cuda_device)
            u_stat_permuted[i] = perm_stat_now

               
        p_value_proxy = (1 +
                         torch.sum(
                             torch.gt(input = u_stat_permuted, other = u_stat_original)
                         )
                        ) / (n_permutation + 1)
      
        return(p_value_proxy)
    
    @abstractmethod   
    def calculate_statistic(self, data_y, data_z):
        raise NotImplementedError()