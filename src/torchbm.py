import numpy as np
import torch
import utilities as utils
from utilities import gauge_adjust_couplings
from tqdm import tqdm

class BM():
    def __init__(self, N=100, n_c=20, gauge='zerosum', zero_field=False, init_couplings = True, curr_float = torch.float32):
        self.N = N
        self.n_c = n_c
        self.zero_field = zero_field
        #self.random_state = np.random.default_rng()
        self.layer_name = 'Potts_coupled'
        self.curr_float = curr_float

        self.fields = torch.zeros([self.N, self.n_c], dtype=self.curr_float)
        self.fields0 = torch.zeros([self.N, self.n_c], dtype=self.curr_float)
        self.couplings = torch.zeros([self.N, self.N, self.n_c, self.n_c], dtype=self.curr_float)
        self.couplings0 = torch.zeros([self.N, self.N, self.n_c, self.n_c], dtype=self.curr_float)
        
        self.gauge = gauge
        self.zero_field = zero_field

        #super(BM, self).__init__(n_layers=1, layers_size=[self.N], layers_nature=[
        #    self.nature + '_coupled'], layers_n_c=[self.n_c], layers_name=['layer'])

        #self.layer = layer.initLayer(N=self.N, nature=self.nature + '_coupled', position='visible',
        #                             n_c=self.n_c, random_state=self.random_state, zero_field=self.zero_field, gauge=self.gauge)
        if init_couplings:
            self.init_couplings(0.01)

    def init_couplings(self, amplitude):
        
        self.couplings = (amplitude *
                            torch.randn( (self.N, self.N, self.n_c, self.n_c), dtype=self.curr_float ) )
        
        self.couplings = gauge_adjust_couplings(self.couplings, gauge = self.gauge) ###MISSING
        
    def init_params_from_data(self, data, eps=1e-6, weights=None, value='data'):
        
        if data is None:
            self.fields = torch.zeros([self.N, self.n_c], dtype=torch.float32)
            self.fields0 = torch.zeros([self.N, self.n_c], dtype=torch.float32)
            self.couplings = torch.zeros(
                [self.N, self.N, self.n_c, self.n_c], dtype=torch.float32)
            self.couplings0 = torch.zeros(
                [self.N, self.N, self.n_c, self.n_c], dtype=torch.float32)

        moments = self.get_moments(data)
        self.fields = utils.invert_softmax(moments['fields'], eps=eps, gauge=self.gauge)
        self.fields0 = self.fields.detach().clone()
        self.couplings *= 0
        self.couplings0 *= 0
        
    def get_moments(self, data):
        moments = {}
        moments['fields'] = utils.average_C(data, n_c=self.n_c)
        moments['coupling'] = utils.covariance_C(data, n_c=self.n_c)
        return moments

    def dict_moments(self, data):
        moments = {}
        moments['fields'] = data[0]
        moments['coupling'] = data[1]
        return moments
            
    def compute_grad(self, data_pos, data_neg, l1=0, l2=0, value = 'data', value_neg = 'data'):
        if value == 'moments':
            moments_pos = self.dict_moments(data_pos)
        else:
            moments_pos = self.get_moments(data_pos)

        if value_neg == 'moments':
            moments_neg = self.dict_moments(data_neg)
        else:
            moments_neg = self.get_moments(data_neg)
        
        field_grad = moments_pos['fields'] - moments_neg['fields']
        coupling_grad = moments_pos['coupling'] - moments_neg['coupling']
        
        if l2 > 0:
            coupling_grad -= l2 * self.couplings
        if l1 > 0:
            coupling_grad -= l1 * torch.sign(self.couplings)
            
        return (field_grad, coupling_grad)


    def internal_gradients(self, data_pos, data_neg, l1=0, l2=0, data_0=None, weights=None, weights_neg=None, weights_0=None,
                           value='data', value_neg='data', value_0='input'):
        gradient = {}
        if value == 'moments':
            moments_pos = data_pos
        else:
            moments_pos = self.get_moments(
                data_pos,  value=value, weights=weights, beta=1)
        if value_neg == 'moments':
            moments_neg = data_neg
        else:
            moments_neg = self.get_moments(
                data_neg,  value=value_neg, weights=weights_neg, beta=1)
            
        if weights is not None:
            mean_weights = weights.mean()
        else:
            mean_weights = 1.

        if self.target0 == 'pos':
            print('wannabe')
            self._target_moments0 = moments_pos
            self._mean_weight0 = mean_weights
        else:
            print('wannabe2')
            self._target_moments0 = moments_neg
            self._mean_weight0 = 1.

        for k, key in enumerate(self.list_params):
            gradient[key] = mean_weights * self.factors[k] * \
                (moments_pos[k] - moments_neg[k])

        if l2 > 0:
            gradients['couplings'] -= l2 * self.couplings
        if l1 > 0:
            gradients['couplings'] -= l1 * torch.sign(self.couplings)
            
        return gradients  

    def compute_output(self, config, couplings, direction='up', out=None):
        #assert (self.n_c > 1) & (couplings.ndim == 4)  # output layer is Potts

        N_output_layer = couplings.shape[0]
        n_c_output_layer = couplings.shape[2]

        config, xshape = utils.reshape_in(config, xdim=1)

        out_dim = list(xshape[:-1]) + [N_output_layer, n_c_output_layer]
        out_ndim = 2

        if out is not None:
            if not list(out.shape) == out_dim:
                out = torch.zeros(out_dim, dtype=torch.float32)
            else:
                out *= 0
        else:
            out = torch.zeros(out_dim, dtype=torch.float32)

        out, _ = utils.reshape_in(out, xdim=out_ndim)
        out = utils.compute_output_Potts_C( config, couplings, n_c = self.n_c )

        return utils.reshape_out(out, xshape, xdim=1)

    def compute_fields_eff(self, x, beta=1):
        if x.ndim == 1:
            x = x[None, :]
        return self.compute_output(x, self.couplings) + self.fields[None]

    def energy(self, config, beta=1):
        if config.ndim == 1:
            config = config[None, :]
            
        fields = self.fields
        couplings = self.couplings
            
        return - utils.dot_Potts_C(config, fields, n_c = self.n_c) - 0.5 * utils.bilinear_form_Potts(config, couplings,n_c = self.n_c)

    def get_input(self, I, I0=None, beta=1):
        if I is None:
            
            I = np.zeros([1, self.N, self.n_c], dtype=curr_float)

        beta_not_one = (beta != 1)

        if beta_not_one:
            I = I * beta
            if I0 is not None:
                I = I + (1 - beta) * I0
        if I.ndim == 1:
            # ensure that the config data is a batch, at least of just one vector
            I = I[None, :]
        return I
        
    def sample_from_inputs(self, I, I0=None, beta=1, previous=(None, None), **kwargs):
        
        if I is None:
            if I0 is not None:
                I = (1 - beta) * I0
        else:
            I = self.get_input(I, I0=I0, beta=beta)
        (x, fields_eff) = previous
        if x is None:
            B = I.shape[0]
            x = torch.randint(0, high=self.n_c, size=[B, self.N], dtype=torch.int32)
        else:
            B = x.shape[0]
        if fields_eff is None:
            fields_eff = self.fields.unsqueeze(dim=0) + self.compute_output(x, self.couplings)
            
            ###TO MODIFY with the correct one
        if I is not None:
            x, fields_eff = utils.Potts_sampling( x, fields_eff, B, self.N, self.n_c, self.fields0.repeat( (fields_eff.shape[0],1,1) ), self.couplings, beta )
            
            #### here Jerome puts the function Potts Gibbs input C __ and adds I as input, but never used!!!!
        else:
            x, fields_eff = utils.Potts_sampling( x, fields_eff, B, self.N, self.n_c, self.fields0.repeat( (fields_eff.shape[0],1,1) ), self.couplings, beta )
            
            #### Potts_sampling here is equivalent to Pott_Gibbs_free_C
        return (x, fields_eff)
    
    def markov_step(self, config, beta=1):
        x, fields_eff = config
        (x, fields_eff) = self.sample_from_inputs( I=None, previous=(x, fields_eff), beta=beta )
        
        return (x, fields_eff)
    
    
    def fit(self, data, batch_size=100, nchains=100, learning_rate=None, extra_params=None, init='independent', optimizer='SGD', N_PT=1, 
            N_MC=1, n_iter=100, lr_decay=True, lr_final=None, decay_after=0.5, l1=0, l2=0, no_fields=False, batch_norm=False,
            update_betas=None, record_acceptance=None, epsilon=1e-6, verbose=1, record=[], record_interval=100, p=[1, 0, 0], pseudo_count=0, weights=None):


        #### In BM batch size should just contribute to the total number of iters. It does not involve real batchSGD as we compute M,C from the beginning.
        
        if N_PT != 1:
            raise ValueError('Only support N_PT = 1 for now' )
        if optimizer != 'SGD':
            raise ValueError('Only support SGD optimizer for now')
            
        self.nchains = nchains
        self.optimizer = optimizer
        self.record_swaps = False
        #self.batch_norm = batch_norm
        #self.layer.batch_norm = batch_norm

        self.n_iter = n_iter

        if learning_rate is None:
            learning_rate = 0.1

        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        if self.lr_decay:
            self.decay_after = decay_after
            self.start_decay = self.n_iter * self.decay_after
            if lr_final is None:
                self.lr_final = 1e-2 * self.learning_rate
            else:
                self.lr_final = lr_final
            self.decay_gamma = (float(self.lr_final) / float(self.learning_rate)
                                )**(1 / float(self.n_iter * (1 - self.decay_after)))

        #self.gradient = self.initialize_gradient_dictionary()

        if weights is not None:
            weights = torch.tensor(weights, dtype=torch.float32)

        mean = utils.average_C(data, n_c=self.n_c)
        covariance = utils.covariance_C(data, n_c=self.n_c)
        
        if pseudo_count > 0:
            p = data.shape[0] / float(data.shape[0] + pseudo_count)
            covariance = p**2 * covariance + p * \
                (1 - p) * (mean[None, :, None, :] * mean[:,
                                                                     None, :, None]) / self.n_c + (1 - p)**2 / self.n_c**2
            mean = p * mean + (1 - p) / self.n_c

        iter_per_epoch = data.shape[0] // batch_size
        self.init_params_from_data(data, eps=epsilon, value='data')

        self.N_PT = N_PT
        self.N_MC = N_MC

        self.l1 = l1
        self.l2 = l2
        self.no_fields = no_fields

        (self.fantasy_x, self.fantasy_fields_eff) = self.sample_from_inputs(
            torch.zeros([self.N_PT * self.nchains, self.N, self.n_c],dtype=torch.int16), beta=0)
        

        self.count_updates = 0
        if verbose:
            if weights is not None:
                lik = (self.energy(data) *
                       weights).sum() / weights.sum()
            else:
                lik = self.energy(data).mean()
            print('Iteration number 0, Energy: %.2f' % lik)

        result = {}
        if 'J' in record:
            result['J'] = []
        if 'F' in record:
            result['F'] = []

        count = 0

        for epoch in tqdm(range(1, n_iter + 1)):

            if self.lr_decay:
                if (epoch > self.start_decay):
                    self.learning_rate *= self.decay_gamma

            #print('Starting epoch %s' % (epoch))
            for _ in range(iter_per_epoch):
                self.minibatch_fit(mean, covariance)

                if (count % record_interval == 0):
                    if 'J' in record:
                        result['J'].append(self.couplings.detach().clone())
                    if 'F' in record:
                        result['F'].append(self.fields.detach().clone())

                count += 1

            if verbose:
                lik = self.energy(data).mean()
                print(f'Iteration number {epoch}, Energy: {lik} \n' )

        return result
            
            
    def minibatch_fit(self, mean, covariance):
            
        for _ in range(self.N_MC):
            (self.fantasy_x, self.fantasy_fields_eff) = self.markov_step(
                (self.fantasy_x, self.fantasy_fields_eff))
        X_neg = self.fantasy_x
        field_grad, coupling_grad = self.compute_grad((mean, covariance), X_neg, l1=self.l1, l2=self.l2, value='moments')
        
        self.couplings = self.couplings + self.learning_rate * coupling_grad
        self.fields = self.fields + self.learning_rate * field_grad
        
        self.couplings = gauge_adjust_couplings( self.couplings, gauge=self.gauge )
        self.couplings[torch.arange(self.N), torch.arange(self.N)] *= 0 ### set diagonal to zero
        
        self.fantasy_fields_eff = self.compute_output(self.fantasy_x, self.couplings) + self.fields[None]

    def gen_data(self, N_data = 1000, n_iter = 100, Nthermalize=100, Nstep=10, beta=1, reshape = True):

        config = self.sample_from_inputs( torch.zeros([N_data, self.N, self.n_c],dtype = torch.int32), beta=0)
        
        for _ in range(Nthermalize):
            config = self.markov_step(config, beta )

        new_samples = torch.zeros( size = (n_iter, N_data, self.N) ,dtype = torch.int32 )
        
        for i in tqdm(range(n_iter)):
            for _ in range(Nstep):
                config = self.markov_step( config, beta)

            new_samples[i] = config[0].detach().clone()

        if reshape:
            new_samples = new_samples.reshape(N_data*n_iter, -1)

        return new_samples

        
        
            
            
            
                

