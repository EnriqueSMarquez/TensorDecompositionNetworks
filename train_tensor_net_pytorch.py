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
        self.bias = nn.Parameter(torch.Tensor(num_units).zero_())

    def forward(self, x):
        for i,(shape,indexes) in enumerate(zip(self.cores_shape,self.cores_indexes)):
            x = x.view(x.size(0),-1,shape[0])
            # curr_core = self.cores_arr[indexes[0]:indexes[1]].contiguous()
            # curr_core = curr_core.view(shape)
            # curr_core = curr_core.repeat(x.size(0),1,1)
            indices = torch.LongTensor(range(indexes[0],indexes[1])).cuda()
            x = torch.bmm(x,torch.index_select(self.cores_arr, 0, Variable(indices)).view(shape).repeat(x.size(0),1,1))
            # x = x.view(x.size(0),-1,self.tt_output_shape[-1-i])
        x = x.view(x.size(0),-1)
        x += self.bias
        return x


class TT_Net(nn.Module):
    def __init__(self,block,tt_input_shape, tt_output_shape, tt_ranks):
        super(TT_Net, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3,num_filters[0],3,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.block1 = block(num_filters[0],depth_per_block-1)
        self.conv3 = nn.Conv2d(num_filters[0],num_filters[1],3,padding=1)
        # self.block2 = blocks[1]
        # self.conv4 = nn.Conv2d(num_filters[1],num_filters[1],3,padding=1)
        if depth_res > 1:
            self.block2 = block(num_filters[1],depth_per_block)
        else:
            self.block2 = block(num_filters[1],depth_per_block-1)
        self.conv5 = nn.Conv2d(num_filters[1],num_filters[2],3,padding=1)
        # self.block3 = blocks[2]
        # self.conv6 = nn.Conv2d(num_filters[2],num_filters[2],3,padding=1)
        if depth_res >= 1:
            self.block3 = block(num_filters[2],depth_per_block)
        else:
            self.block3 = block(num_filters[2],depth_per_block-1)
        # self.fc1 = nn.Linear(np.prod(tt_output_shape).astype(int), 512)
        self.fc1 = nn.Linear(3*32*32, 512)
        # self.fc1 = nn.Linear(num_filters[-1] * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                n = m.in_features * m.out_features
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m,nn.Sequential):
                for conv in m:
                    if isinstance(conv,nn.Conv2d):
                        n = conv.kernel_size[0] * conv.kernel_size[1] * conv.out_channels
                        conv.weight.data.normal_(0, math.sqrt(2. / n))
                        conv.bias.data.zero_()

        self.ttlayer = TTLayer(tt_input_shape, tt_output_shape, tt_ranks)
        

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.relu(x)
        # # x = self.block1(x)
        # x = self.pool(x)

        # x = self.conv3(x)
        # x = self.relu(x)
        # # x = self.block2(x)
        # x = self.pool(x)

        # x = self.conv5(x)
        # x = self.relu(x)
        # # x = self.block3(x)
        # x = self.pool(x)

        x = x.view(x.size(0), -1)
        # x = self.ttlayer(x)
        # x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def make_block(num_filters,nb_layers):
    layers = []
    for i in range(nb_layers):
        conv2d = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        # layers += [conv2d]
        layers += [conv2d, nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)

folder_to_save = './Run1/'
torch.backends.cudnn.enabled = True

nEpoch = 150
depth = 9
filters = 256
depth_per_block = np.floor(depth/3.).astype(int)
depth_res = np.mod(depth,3)
num_filters = filters*np.asarray([1,2,4])

if not os.path.isdir(folder_to_save):
    os.mkdir(folder_to_save)

transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(root='./datasets', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=1)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



tt_input_shape = [4,4,4,4,4,3]
tt_output_shape = [5,5,5,5,5,5]
tt_ranks = [1,3,3,3,3,3,1]
model = TT_Net(make_block,tt_input_shape, tt_output_shape, tt_ranks).cuda()

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