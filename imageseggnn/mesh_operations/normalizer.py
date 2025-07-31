import numpy as np
import torch

class Normalizer_np():
    def __init__(self,params=[],method = '-11',dim=None):
        self.params = params
        self.method = method
        self.dim = dim
    def fit_normalize(self,data):
        assert type(data) == np.ndarray
        if len(self.params) ==0:
            if self.method == '-11':
                self.params = (np.max(data,axis=0),np.min(data,axis=self.dim))
            elif self.method == 'ms':
                self.params = (np.mean(data,axis=0),np.std(data,axis=self.dim))
        return self.fnormalize(data,self.params,self.method)

    def normalize(self, new_data):
        return self.fnormalize(new_data,self.params,self.method)
    def denormalize(self, new_data_norm):
        return self.fdenormalize(new_data_norm,self.params,self.method)
    def get_params(self):
        if self.method == 'ms':
            print('returning mean and std')
        if self.method == '-11':
            print('returning max and min')
        return self.params
    @staticmethod
    def fnormalize(data, params, method):
        if method == '-11':
            return (data-params[1])/(params[0]-params[1])*2-1
        if method == 'ms':
            return (data-params[0])/params[1]
        
    @staticmethod
    def fdenormalize(data_norm, params, method):
        if method == '-11':
            return (data_norm+1)/2*(params[0]-params[1])+params[1]
        if method == 'ms':
            return data_norm*params[1]+params[0]

class Normalizer_ts():
    def __init__(self,params=[],method = '-11',dim=None):
        self.params = params
        self.method = method
        self.dim = dim
    def fit_normalize(self,data):
        assert type(data) == torch.Tensor
        if len(self.params) ==0:
            if self.method == '-11' or self.method =='01':
                if self.dim == None:
                    self.params = (torch.max(data),torch.min(data))
                else:
                    self.params = (torch.max(data,dim=self.dim)[0],torch.min(data,dim=self.dim)[0])
            elif self.method == 'ms':
                if self.dim == None:
                    self.params = (torch.mean(data,dim=None),torch.std(data,dim=None))
                else:
                    self.params = (torch.mean(data,dim=self.dim),torch.std(data,dim=self.dim))

        return self.fnormalize(data,self.params,self.method)

    def normalize(self, new_data):
        return self.fnormalize(new_data,self.params,self.method)
    def denormalize(self, new_data_norm):
        return self.fdenormalize(new_data_norm,self.params,self.method)
    def get_params(self):
        if self.method == 'ms':
            print('returning mean and std')
        if self.method == '-11' or self.method == '01':
            print('returning max and min')
        return self.params
    @staticmethod
    def fnormalize(data, params, method):
        if method == '-11':
            return (data-params[1])/(params[0]-params[1])*2-1
        if method == 'ms':
            return (data-params[0])/params[1]
        if method == '01':
            return (data-params[1])/(params[0]-params[1])

        
    @staticmethod
    def fdenormalize(data_norm, params, method):
        if method == '-11':
            return (data_norm+1)/2*(params[0]-params[1])+params[1]
        if method == 'ms':
            return data_norm*params[1]+params[0]
        if method == '01':
            return (data_norm)/(params[0]-params[1])+params[1]

def get_data_range(dataset, data_label):
    data_all = []
    for i,ele in enumerate(dataset):
        # print(ele[data_label].shape)
        data_all.append(ele[data_label])
    data_all = torch.cat(data_all,dim=0)

    data_max = torch.max(data_all,dim= 0)[0]
    data_min = torch.min(data_all,dim= 0)[0]
    data_mean = torch.mean(data_all,dim= 0)
    data_std = torch.std(data_all,dim= 0)


    data_all_norm = (data_all- data_mean)/data_std
    #     data_max.append(torch.max(temp, dim=0)[0])
    #     data_min.append(torch.min(temp, dim=0)[0])
    #     data_mean.append(torch.mean(temp, dim=0))
    #     data_std.append(torch.std(temp, dim=0))
    # data_max = torch.stack(data_max)
    # data_min = torch.stack(data_min)
    # data_mean = torch.stack(data_mean)
    # data_std = torch.stack(data_std)
    return {'max':torch.max(data_max),
            'min':torch.min(data_min),
            'mean':torch.min(data_mean),
            'std':torch.min(data_std)}

if __name__ == "__main__":
    my_data = np.random.random((100,50))
    # my_normalizer = Normalizer(method = '-11')
    my_normalizer = Normalizer_np(method = 'ms')
    my_data_norm = my_normalizer.fit_normalize(my_data)
    print(my_data_norm.shape)
    my_data_rec = my_normalizer.denormalize(my_data_norm)
    print(np.max(abs(my_data-my_data_rec)))

    my_data = torch.rand(100,33)
    # my_normalizer = Normalizer(method = '-11')
    my_normalizer = Normalizer_ts(method = 'ms')
    my_data_norm = my_normalizer.fit_normalize(my_data)
    print(my_data_norm.shape)
    my_data_rec = my_normalizer.denormalize(my_data_norm)
    print(torch.max(torch.abs(my_data-my_data_rec)))