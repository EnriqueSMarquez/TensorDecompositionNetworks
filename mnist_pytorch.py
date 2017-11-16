import torch
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn

import torch.optim as optim

from tqdm import tqdm
import random
import cPickle
import sys
import os
import numpy as np
import math


def _generate_orthogonal_tt_cores(input_shape, output_shape, ranks):
    # Generate random orthogonalized tt-tensor.
    input_shape = np.array(input_shape)
    output_shape = np.array(output_shape)
    ranks = np.array(ranks)
    cores_arr_len = np.sum(input_shape * output_shape *
                           ranks[1:] * ranks[:-1])
    cores_arr = np.zeros(cores_arr_len).astype(float)
    cores_arr_idx = 0
    core_list = []
    rv = 1
    shapes = list()
    cores_indexes = list()

    for k in range(input_shape.shape[0]):
        shape = [ranks[k], input_shape[k], output_shape[k], ranks[k+1]]
        shapes.append((input_shape[k] * ranks[k + 1], ranks[k] * output_shape[k]))
        tall_shape = (np.prod(shape[:3]), shape[3])
        curr_core = np.dot(rv, np.random.normal(0, 1, (shape[0], np.prod(shape[1:]))))
        curr_core = curr_core.reshape(tall_shape)
        if k < input_shape.shape[0]-1:
            curr_core, rv = np.linalg.qr(curr_core)
        cores_arr[cores_arr_idx:cores_arr_idx+curr_core.size] = curr_core.flatten()
        cores_indexes.append((cores_arr_idx,cores_arr_idx+curr_core.size))
        cores_arr_idx += curr_core.size
    # TODO: use something reasonable instead of this dirty hack.
    glarot_style = (np.prod(input_shape) * np.prod(ranks))**(1.0 / input_shape.shape[0])
    cores_arr = (0.1 / glarot_style) * cores_arr
    # mean = np.mean(cores_arr)
    # std = np.std(cores_arr)
    # cores_arr -= mean
    # cores_arr /= (std + 1.e-7)

    return cores_arr,shapes,cores_indexes

class TTLayer(nn.Module):

    def __init__(self,tt_input_shape, tt_output_shape, tt_ranks):
        super(TTLayer, self).__init__()
        tt_input_shape = np.array(tt_input_shape)
        tt_output_shape = np.array(tt_output_shape)
        tt_ranks = np.array(tt_ranks)
        self.tt_input_shape = tt_input_shape
        self.tt_output_shape = tt_output_shape
        self.tt_ranks = tt_ranks
        self.num_dim = tt_input_shape.shape[0]
        local_cores_arr,cores_shape,cores_indexes = _generate_orthogonal_tt_cores(tt_input_shape,
                                                       tt_output_shape,
                                                       tt_ranks)
        cores_indexes.reverse()
        self.cores_indexes = cores_indexes
        self.cores_arr = nn.Parameter(torch.Tensor(torch.from_numpy(local_cores_arr).float()))
        # self.cores_arr = self.cores_arr
        num_units = np.prod(tt_output_shape)
        cores_shape.reverse()
        self.cores_shape = cores_shape
        self.bias = nn.Parameter(torch.Tensor(num_units).fill_(0))

    def forward(self, x):
        for i,(shape,indexes) in enumerate(zip(self.cores_shape,self.cores_indexes)):
            x = x.view(x.size(0),-1,shape[0])
            # curr_core = self.cores_arr[indexes[0]:indexes[1]].contiguous()
            # curr_core = curr_core.view(shape)
            # curr_core = curr_core.repeat(x.size(0),1,1)
            indices = torch.LongTensor(range(indexes[0],indexes[1])).cuda()
            x = torch.bmm(x,torch.index_select(self.cores_arr, 0, Variable(indices)).view(shape).repeat(x.size(0),1,1))
            x = x.view(x.size(0),-1,self.tt_output_shape[-1-i])
        x = x.view(x.size(0),-1)
        x += self.bias
        return x


class TT_Net(nn.Module):
    def __init__(self,tt_input_shape, tt_output_shape, tt_ranks):
        super(TT_Net, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(np.prod(tt_output_shape).astype(int), 800)
        self.fc2 = nn.Linear(800, 10)
        # self.fc3 = nn.Linear(256, 10)

        for m in self.modules():
            if isinstance(m,nn.Linear):
                n = m.in_features * m.out_features
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

        self.ttlayer = TTLayer(tt_input_shape, tt_output_shape, tt_ranks)
        

    def forward(self, x):
        # x = x.view(x.size(0), -1)
        x = self.ttlayer(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.relu(x)
        # x = self.fc3(x)
        return x

folder_to_save = './Run1/'
torch.backends.cudnn.enabled = True

nEpoch = 150
depth = 27
filters = 64
depth_per_block = np.floor(depth/3.).astype(int)
depth_res = np.mod(depth,3)
num_filters = filters*np.asarray([1,2,4])

if not os.path.isdir(folder_to_save):
    os.mkdir(folder_to_save)

trainloader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./datasets', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=500, shuffle=True)
testloader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./datasets', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=500, shuffle=True)


tt_input_shape = [4, 7, 4, 7]
tt_output_shape = [5, 5, 8, 4]
tt_ranks = [1, 3, 3, 3, 1]
model = TT_Net(tt_input_shape, tt_output_shape, tt_ranks).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),weight_decay=0)
test_accuracy = list()
for epoch in range(nEpoch):  # loop over the dataset multiple times
    print('EPOCH %d')%(epoch)
    running_loss = 0.0
    trainloader = tqdm(trainloader)
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss = 0.99 * running_loss + 0.01 * loss.data[0]
        trainloader.set_postfix(loss=running_loss)
    correct = 0.
    total = 0.
    for data in testloader:
        images, labels = data
        labels = labels.cuda()

        outputs = model(Variable(images).cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    test_accuracy.append(correct/total)
    print('TEST ACCURACY : %.3f')%(test_accuracy[-1])
with open(folder_to_save + 'history_'+str(depth)+'_'+str(filters)+'_'+'.txt','w') as fp:
    cPickle.dump(test_accuracy,fp)
print('**** Finished Training ****')