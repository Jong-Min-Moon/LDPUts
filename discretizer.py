class discretizer:
    def __init__(self, cuda_device):
        self.cuda_device = cuda_device

       def h_bin(self, data, kappa): 
        ''' Only for continuous data
        input arguments
            data: torch tensor of continuous data
            kappa: number of bin in each dimension
        output
            torch tensor of multivariate data
        '''
               
        # create designated number of intervals
        d = self.get_dimension(data)
     
        # 1. for each dimension, turn the continuous data into interval
        # each row now indicates a hypercube in [0,1]^d
        # the more the data is closer to 1, the larger the interval index.
        dataBinIndex = self.transform_bin_index(data = data, nIntervals = kappa)
        
        # 2. for each datapoint(row),
        #    turn the hypercube data into a multivariate data of (1, 2, ..., kappa^d)
        #    each row now becomes an integer.
        dataMultivariate = self.TransformMultivariate(dataBinIndex, kappa)
        
        return(dataMultivariate)
    
          
    def transform_onehot(dataMultivariate, d):
        return(
            torch.nn.functional.one_hot(dataMultivariate, num_classes = d)
        )
    
 
    

       def transform_bin_index(self, data, nIntervals):
        ''' Only for continuous data.
        for each dimension, transform the data in [0,1] into the interval index
        first interval = [0, x], the others = (y z]
        
        input arguments
            data: torch tensor object on GPU
            nIntervals: integer
        output
            dataIndices: torch tensor, dimension same as the input
        '''
        # create designated number of intervals
        d = self.get_dimension(data)
        breaks = torch.linspace(start = 0, end = 1, steps = nIntervals + 1).to(self.cuda_device) #floatTensor
        dataIndices = torch.bucketize(data, breaks, right = False) # ( ] form.
        dataIndices = dataIndices.add(
            dataIndices.eq(0)
        ) #move 0 values from the bin number 0 to the bin number 1       
        return(dataIndices)    

    def TransformMultivariate(self, dataBinIndex, nBin):
        """Only for continuous and multivariate data ."""
        d = self.get_dimension(dataBinIndex)
        if d == 1:
            return(dataBinIndex.sub(1).reshape(-1,))
        else:
            exponent = torch.linspace(start = (d-1), end = 0, steps = d, dtype = torch.long)
            vector = torch.tensor(nBin).pow(exponent)
            return( torch.matmul( dataBinIndex.sub(1).to(torch.float), vector.to(torch.float).to(self.cuda_device) ).to(torch.long) )   
    
    