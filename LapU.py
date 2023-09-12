class LapU:
    def __init__(self, cuda_device):
        self.cuda_device = cuda_device


    def privatize(self, data_mutinomial, alphabet_size, privacy_level):
        data_onehot = self.transform_onehot(data_mutinomial, alphabet_size)


        d = self.kappa ** tst_data_y.size(dim = 1)
        theta = d**(1/2)
        tst_data_y_multi = self.h_bin(tst_data_y, self.kappa)
        tst_data_y_oneHot = self.transform_onehot(tst_data_y_multi, d)
        tst_data_z_multi = self.h_bin(tst_data_z, self.kappa) 
        tst_data_z_oneHot = self.transform_onehot(tst_data_z_multi, d)
        dataCombined = torch.cat([tst_data_y_oneHot, tst_data_z_oneHot], dim = 0)
        tst_data_priv = self.LapU(dataCombined, alpha, 2, theta)
        return(tst_data_priv)
    

    def LapU(self, oneHot, alpha, c, theta):
        p = torch.exp(torch.tensor(
            - alpha / (c * theta)
            )).to(self.cuda_device)
        laplaceSize = oneHot.size()
        laplaceNoise = self.generate_disc_laplace(p, laplaceSize)
        LDPView = torch.tensor(theta) * oneHot + laplaceNoise
        return(LDPView)
    
    def transform_onehot(data_multinomial, alphabet_size)
        return(torch.nn.functional.one_hot(data_multinomial, alphabet_size))
