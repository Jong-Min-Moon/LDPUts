import torch

class LapU:
    def __init__(self, cuda_device):
        self.cuda_device = cuda_device
        self.__initialize_laplace_generator()


    def privatize(self, data_mutinomial, alphabet_size, privacy_level):
        sample_size = self.__get_sample_size(data_mutinomial)
        data_onehot_scaled = self.__transform_onehot(data_mutinomial, alphabet_size).mul(alphabet_size**(1/2)) # scaled by \sqrt(k)
        noise = self.__generate_noise(alphabet_size, privacy_level, sample_size)
        return(torch.add(data_onehot_scaled,noise))
           
    def __generate_noise(self, alphabet_size, privacy_level, sample_size):
        laplace_scale = torch.tensor(2*alphabet_size).sqrt().mul(2).div(privacy_level).to(self.cuda_device) #sigma_alpha in the paper
        laplace_noise = self.unit_laplace_generator.sample(sample_shape = torch.Size([sample_size, alphabet_size])).to(self.cuda_device)
        return(laplace_noise.mul(laplace_scale))

    def __transform_onehot(self, data_multinomial, alphabet_size):
        return(torch.nn.functional.one_hot(data_multinomial, alphabet_size))

    def __get_sample_size(self, data):
        if data.dim() == 1:
            return( data.size(dim = 0) )
        elif data.dim() == 2:
            return( data.size(dim = 1) )
        else:
            return # we only use up to 2-dimensional tensor, i.e. matrix
        
    def __initialize_laplace_generator(self):
        self.unit_laplace_generator = torch.distributions.laplace.Laplace(
            torch.tensor(0.0).to(self.cuda_device),
            torch.tensor(2**(-1/2)).to(self.cuda_device)
        )

