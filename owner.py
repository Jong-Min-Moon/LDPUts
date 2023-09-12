class owner:
    def __init__(self, cuda_device, channel):
        self.cuda_device = cuda_device
        self.privacy_channel = channel
        self.transformer = data_transformer()
    
    def import_disc(self, discdata)
        self.discdata = discdata

    def release(self)
        return(
            self.privacy_channel.privatize(self.discdata)
        )

    

