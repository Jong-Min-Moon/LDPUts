class genRR()
    def __init__(self, cuda_device):
        self.cuda_device = cuda_device

    def release(self, data, privacy_level, alphabet_size):
       p = 1 / (torch.tensor(privacy_level).exp() + alphabet_size - 1)
       p = p.repeat_interleave(alphabet_size)
       p[data] = p[data] * torch.tensor(privacy_level).exp()
       priv <- torch.multinomial(p, 1)
       return(which(!priv == 0))
