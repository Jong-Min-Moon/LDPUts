import torch
import utils

class validator:
    def get_bitflip_prob(self, privacy_level):
        half = torch.tensor(privacy_level/2)
        exp_half = half.exp()
        prob_same = exp_half.div(exp_half.add(1))
        prob_diff = 1-prob_same
        return(prob_same, prob_diff)
    
    def get_uniform_perturb_l2_dist_squared(self, alphabet_size, bump_size):
        p1, p2 = utils.get_uniform_perturb(alphabet_size, bump_size)
        dist_squared = p1.sub(p2).square().sum()
        return(dist_squared)
    
    def get_uniform_perturb_bitflip_elltwo_expectation(self, alphabet_size, bump_size, privacy_level):
        dist_squared = self.get_uniform_perturb_l2_dist_squared(alphabet_size, bump_size)
        prob_same, prob_diff = self.get_bitflip_prob(privacy_level)
        expectation = prob_same.square().mul(dist_squared)
        return(expectation)