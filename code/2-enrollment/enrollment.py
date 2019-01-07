import torch
import torchvision
import torchvision.transforms as transforms
import os
from scipy.io.wavfile import read
import scipy.io.wavfile as wav
import subprocess as sp
import numpy as np
import argparse
import random
import os
import sys
import torch.nn.init as init
from random import shuffle
import speechpy
import datetime
sys.path.insert(0, '../InputPipeline')
from DataProviderEnrollment import AudioDataset, CMVN, Feature_Cube, ToOutput
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import time
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import Normalizer

# Useful function for arguments.
def str2bool(v):
    return v.lower() in ("yes", "true")


parser = argparse.ArgumentParser(description='Creating background model in development phase')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
parser.add_argument('--basenet', default=None, help='pretrained base model')
parser.add_argument('--load_weights', default='../1-development/weights/net_final.pth', type=str, help='Load weights')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--batch_per_log', default=10, type=int, help='Print the log at what number of batches?')
parser.add_argument('--enrollment_path',
                    default=os.path.expanduser('~/Downloads/voxceleb_data/voxceleb1_enrollment.txt'),
                    help='The file names for enrollment phase')
parser.add_argument('--save_folder', default='model', help='Location to save models')
parser.add_argument('--audio_dir', default=os.path.expanduser('~/Downloads/voxceleb_data/voxceleb1_audio'),
                    help='Location of sound files')
args = parser.parse_args()

# Checking the appropriate folder for saving
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

##################################
####### Initiate the dataset #####
##################################
enrollment_data = AudioDataset(files_path=args.enrollment_path,
                                audio_dir=args.audio_dir,
                                transform=transforms.Compose([CMVN(), Feature_Cube((80, 40, 20)), ToOutput()]))

dataloader = torch.utils.data.DataLoader(enrollment_data, batch_size=args.batch_size,
                                          shuffle=False, num_workers=args.num_workers)

# Ex: get some random training images (not used!)
dataiter = iter(dataloader)
images, labels = dataiter.next()

#############
### Model ###
#############
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv11 = nn.Conv3d(1, 16, (4, 9, 9), stride=(1, 2, 1))
        self.conv11_bn = nn.BatchNorm3d(16)
        self.conv12 = nn.Conv3d(16, 16, (4, 9, 9), stride=(1, 1, 1))
        self.conv12_bn = nn.BatchNorm3d(16)
        self.conv21 = nn.Conv3d(16, 32, (3, 7, 7), stride=(1, 1, 1))
        self.conv21_bn = nn.BatchNorm3d(32)
        self.conv22 = nn.Conv3d(32, 32, (3, 7, 7), stride=(1, 1, 1))
        self.conv22_bn = nn.BatchNorm3d(32)
        self.conv31 = nn.Conv3d(32, 64, (3, 5, 5), stride=(1, 1, 1))
        self.conv31_bn = nn.BatchNorm3d(64)
        self.conv32 = nn.Conv3d(64, 64, (3, 5, 5), stride=(1, 1, 1))
        self.conv32_bn = nn.BatchNorm3d(64)
        self.conv41 = nn.Conv3d(64, 128, (3, 3, 3), stride=(1, 1, 1))
        self.conv41_bn = nn.BatchNorm3d(128)
        # self.conv42 = nn.Conv3d(128, 512, (4, 6, 2), stride=(1, 1, 1))
        # self.conv42_bn = nn.BatchNorm3d(512)

        # Fully-connected
        self.fc1 = nn.Linear(128 * 4 * 6 * 2, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 1211)

    def forward(self, x):
        x = F.relu(self.conv11_bn(self.conv11(x)))
        x = F.relu(self.conv12_bn(self.conv12(x)))
        x = F.relu(self.conv21_bn(self.conv21(x)))
        x = F.relu(self.conv22_bn(self.conv22(x)))
        x = F.relu(self.conv31_bn(self.conv31(x)))
        x = F.relu(self.conv32_bn(self.conv32(x)))
        x = F.relu(self.conv41_bn(self.conv41(x)))
        # x = F.relu(self.conv2_bn(self.conv42(x))

        x = x.view(-1, 128 * 4 * 6 * 2)
        x = self.fc1_bn(self.fc1(x))
        # x = self.fc2(x)
        # x = self.fc3(x)
        return x


# Call the net
net = Net()

############
### Cuda ###
############

# Multi GPU calling
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net)

# Put operations on GPU
if args.cuda and torch.cuda.is_available():
    net.cuda()

#################
### Optimizer ###
#################
criterion = nn.CrossEntropyLoss()

############
### load ###
############

weights = torch.load(os.path.join(args.load_weights))
print('Loading base network...')
# We only load the 'state_dict' related parameters which are the weights.
# The reason is that we do not want to resume, we just want the pretrained weights!
net.load_state_dict(weights['state_dict'])

######################
### Training loop ####
######################
num_batches = len(dataloader)

running_loss = 0.0
running_accuracy = 0.0
num_enrollment = 1
output_numpy = np.zeros(shape=[num_enrollment,40,128],dtype=np.float32)
model = np.zeros(shape=[40,128],dtype=np.float32)

for i in range(num_enrollment):
    for iteration, data in enumerate(dataloader, 1):

        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if args.cuda:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)


        # forward + backward + optimize
        t0 = time.time()
        outputs = net(inputs)
        output_numpy[i] = outputs.cpu().data.numpy()

        # Loss
        loss = criterion(outputs, labels)

        # Prediction
        _, predictions = torch.max(outputs, dim=-1)
        # pred == y returns a ByteTensor, which has only an 8-bit range. Hence, after a particular batch-size, the sum may overflow
        # and hence shoot the wrong results.
        correct_count = (predictions == labels).double().sum().data[0]
        accuracy = float(correct_count) / args.batch_size

        # forward, backward & optimization
        t1 = time.time()
        duration_estimate = t1 - t0

        # print statistics
        running_loss += loss.data[0]
        running_accuracy += accuracy
        if iteration % args.batch_per_log == 0:
            print('Estimated time for each batch: {:.4f} sec.\n'.format(duration_estimate), end=' ')
            print(('epoch {:2d} ' + '|| batch {:2d} of {:2d} ||' + ' Loss: {:.4f} ||' + ' Batch-Accuracy: {:.4f} ||\n').format(
                epoch + 1, iteration, num_batches, running_loss / args.batch_per_log, accuracy), end=' ')
            running_loss = 0.0

        # Print the averaged accuracy for epoch
        print('The averaged accuracy for each epoch: {:.4f}.\n'.format(100.0 * running_accuracy / num_batches), end=' ')


for i in range(output_numpy.shape[0]):
    enrollment_model = output_numpy[i]
    enrollment_model = Normalizer(norm='l2').fit_transform(enrollment_model)
    model += enrollment_model
model = model / float(num_enrollment)
#model = outputs.cpu().data.numpy()
np.save(os.path.join(args.save_folder,'model.npy'),model)

