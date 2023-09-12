class LapU:
    def __init__(self, cuda_device):
        self.cuda_device = cuda_device


    def privatize(self, data_mutinomial, alphabet_size, privacy_level):
        data_onehot = self.transform_onehot(data_mutinomial, alphabet_size)
        laplace_scale = torch.tensor(2*alphabet_size).sqrt().mul(2).div(privacy_level) #sigma_alpha in the paper
        sample_size = self.get_sample_size(data_mutinomial)
        laplace_noise = self.generate_random_noise(sample_size, alphabet_size, privacy_level)
        return(
            torch.add(
                data_onehot.mul(sqrt(alphabet_size)),
                laplace_noise.mul(laplace_scale)
            )  
        )
    
    

    def generate_random_noise(self, sample_size, alphabet_size, privacy_level):

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

    def get_sample_size(self, data):
        if data.dim() == 1:
            return( data.size(dim = 0) )
        elif data.dim() == 2:
            return( data.size(dim = 1) )
        else:
            return # we only use up to 2-dimensional tensor, i.e. matrix