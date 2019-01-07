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
from DataProviderDevelopment import AudioDataset, CMVN, Feature_Cube, ToOutput
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import time
from torch.optim.lr_scheduler import StepLR
# import loader_fn


# Useful function for arguments.
def str2bool(v):
    return v.lower() in ("yes", "true")


parser = argparse.ArgumentParser(description='Creating background model in development phase')

#################
# Dataset Flags #
#################
parser.add_argument('--development_path',
                    default=os.path.expanduser('~/Downloads/voxceleb_data/voxceleb1_development.txt'),
                    help='The file names for development phase')
parser.add_argument('--audio_dir', default=os.path.expanduser('~/Downloads/voxceleb_data/voxceleb1_audio'),
                    help='Location of sound files')

#############################
# Finetuning & Resume Flags #
#############################
parser.add_argument('--resume', default=None, type=str, help="Resume from checkpoint. ex: os.path.expanduser('~/weights/net_final.pth')")
parser.add_argument('--fine_tuning', default=None, type=str,
                    help="Fine_tuning from checkpoint ex: os.path.expanduser('~/weights/net_epoch_10.pth')")
parser.add_argument('--trainable_layers', default=None, type=list,
                    help="Trainable layer and the other layers will be freezed. If it is None, all layers will be trainable ex:['fc1', 'fc2']")
parser.add_argument('--exlude_layer_from_checkpoint', default=None, type=list,
                    help="Layers to be excluded from checkpoint ex: ['fc1','fc2']")

######################
# Optimization Flags #
######################

parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--epochs_per_lr_drop', default=450, type=float,
                    help='number of epochs for which the learning rate drops')

##################
# Training Flags #
##################
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
parser.add_argument('--num_epoch', default=450, type=int, help='Number of training iterations')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--save_folder', default=os.path.expanduser('~/weights'), help='Location to save checkpoint models')
parser.add_argument('--epochs_per_save', default=10, type=int,
                    help='number of epochs for which the model will be saved')
parser.add_argument('--batch_per_log', default=10, type=int, help='Print the log at what number of batches?')

# Add all arguments to parser
args = parser.parse_args()

# Checking the appropriate folder for saving
if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

################################
### Initialization functions ###
################################

def xavier(param):
    init.xavier_uniform(param)


# Initializer function
def weights_init(m):
    """
    Different type of initialization have been used for conv and fc layers.
    :param m: layer
    :return: Initialized layer. Return occurs in-place.
    """
    if isinstance(m, nn.Conv3d):
        xavier(m.weight.data)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0]  # number of rows
        fan_in = size[1]  # number of columns
        variance = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)

#######################
### Save & function ###
#######################

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    print('saving model ...')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

##################################
####### Initiate the dataset #####
##################################
development_data = AudioDataset(files_path=args.development_path,
                                audio_dir=args.audio_dir,
                                transform=transforms.Compose([CMVN(), Feature_Cube((80, 40, 20)), ToOutput()]))


trainloader = torch.utils.data.DataLoader(development_data, batch_size=args.batch_size,
                                          shuffle=True,num_workers=0,pin_memory=False)

# Ex: get some random training images (not used!)
dataiter = iter(trainloader)
item = dataiter.next()
images, labels = item


#############
### Model ###
#############
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        ################
        ### Method 1 ###
        ################
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

        # Fully-connected
        self.fc1 = nn.Linear(128 * 4 * 6 * 2, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 1211)


        # ################
        # ### Method 2 ###
        # ################
        # self.cnn = nn.Sequential(
        #      nn.Conv3d(1, 16, (4, 9, 9), stride=(1, 2, 1)),
        #      nn.BatchNorm3d(16),
        #      nn.ReLU(),
        #      nn.Conv3d(16, 16, (4, 9, 9), stride=(1, 1, 1)),
        #      nn.BatchNorm3d(16),
        #      nn.ReLU(),
        #      nn.Conv3d(16, 32, (3, 7, 7), stride=(1, 1, 1)),
        #      nn.BatchNorm3d(32),
        #      nn.ReLU(),
        #      nn.Conv3d(32, 32, (3, 7, 7), stride=(1, 1, 1)),
        #      nn.BatchNorm3d(32),
        #      nn.ReLU(),
        #      nn.Conv3d(32, 64, (3, 5, 5), stride=(1, 1, 1)),
        #      nn.BatchNorm3d(64),
        #      nn.ReLU(),
        #      nn.Conv3d(64, 64, (3, 5, 5), stride=(1, 1, 1)),
        #      nn.BatchNorm3d(64),
        #      nn.ReLU(),
        #      nn.Conv3d(64, 128, (3, 3, 3), stride=(1, 1, 1)),
        #      nn.BatchNorm3d(128),
        #      nn.ReLU(),
        # )
        #
        # self.fc = nn.Sequential(
        #      nn.Linear(128 * 4 * 6 * 2, 512),
        #      nn.BatchNorm1d(512),
        #      nn.ReLU(),
        #      nn.Linear(512, 1211),
        # )

    def forward(self, x):
        # Method-1
        x = F.relu(self.conv11_bn(self.conv11(x)))
        x = F.relu(self.conv12_bn(self.conv12(x)))
        x = F.relu(self.conv21_bn(self.conv21(x)))
        x = F.relu(self.conv22_bn(self.conv22(x)))
        x = F.relu(self.conv31_bn(self.conv31(x)))
        x = F.relu(self.conv32_bn(self.conv32(x)))
        x = F.relu(self.conv41_bn(self.conv41(x)))
        x = x.view(-1, 128 * 4 * 6 * 2)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = self.fc2(x)

        # # Method Sequential
        # x = self.cnn(x)
        # x = x.view(-1, 128 * 4 * 6 * 2)
        # x = self.fc(x)

        return x


# Call the net
model = Net()

############
### Cuda ###
############

# Multi GPU calling
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

# Put operations on GPU
if args.cuda and torch.cuda.is_available():
    model.cuda()


#################
### Optimizer ###
#################

# Metric for evaluation
criterion = nn.CrossEntropyLoss()

# Get the trainable variable list.
model_dict = model.state_dict()
if args.trainable_layers is not None:
    keys = [k for k, v in model_dict.items()]
    trainable_variable_list = []
    for key in keys:
        for layer_name in args.trainable_layers:
            if layer_name in key:
                trainable_variable_list.append(key)

    # Define optimizer with ignoring variables.
    # The variables that are not trainable, get the learning rate of zero!
    parameters_indicator = []
    for name, param in model.named_parameters():
        if name in trainable_variable_list:
            parameters_indicator.append({'params': param})
        else:
            parameters_indicator.append({'params': param, 'lr': 0.00001})

    # Define optimizer with putting the learning rate of some variables to zero!
    # The variables that we want to freeze them.
    optimizer = optim.SGD(parameters_indicator, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

else:
    # If args.trainable_layers == None, all layers set to be trainable.
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

# Learning rate policy
scheduler = StepLR(optimizer, step_size=args.epochs_per_lr_drop, gamma=args.gamma)

#########################
### Resume & Finetune ###
#########################

if not args.resume and not args.fine_tuning:
    print('Initializing weights...')
    # initialize newly added layers' weights with xavier method
    model.apply(weights_init)

else:
    # Proper assertion (we have to either start from a pretrained model or resume training)
    assert args.resume is None or args.fine_tuning is None, 'You want to resume or fine-tuning from a pretrained model\nboth flags cannot be true!?'
    if args.resume:
        if os.path.isfile(args.resume):
            print('Resume the training ...')
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch_from_resume = checkpoint['epoch']
            best_accuracy = checkpoint['best_accuracy']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    elif args.fine_tuning:
        '''
        Finetuning from pretrained weights.

           * We do not want to resume, we just want the pretrained weights!
           * For this part we filtering out the keys that are available in pretrained weights and not the model at hand
           * we also filterout the keys that are not supposed to be loaded from checkpoint(excluded).
        '''

        # We only load the 'state_dict' related parameters which are the weights.
        print('Loading base network...')
        pretrained_dict = torch.load(os.path.join(args.save_folder, args.fine_tuning))['state_dict']
        model_dict = model.state_dict()

        # This part is for filtering out the keys that are available in pretrained weights and not the model at hand
        # & filtering out the keys that are not supposed to be loaded from checkpoint.
        # Get all the keys for defined layers to be excluded from checkpoint
        if args.exlude_layer_from_checkpoint is not None:
            keys = [k for k, v in model_dict.items()]
            exclude_model_dict = []
            for key in keys:
                for layer_name in args.exlude_layer_from_checkpoint:
                    if layer_name in key:
                        exclude_model_dict.append(key)

            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in exclude_model_dict}
        else:
            # This part is for just filtering out the keys that are available in pretrained weights and not the model
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)

        # 3. load the new state dict
        model.load_state_dict(model_dict)

######################
### Training loop ####
######################
best_accuracy = 0.0
num_batches = len(trainloader)

# Start epochs from resume
if args.resume:
    start = start_epoch_from_resume
else:
    start = 0
for epoch in range(start, args.num_epoch):  # loop over the dataset multiple times

    # Step the lr scheduler each epoch!
    scheduler.step()

    # Running loss would be initiated for each iteration.
    running_loss = 0.0
    running_accuracy = 0.0
    for iteration, data in enumerate(trainloader, 1):

        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if args.cuda:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        t0 = time.time()
        outputs = model(inputs)

        # Loss
        loss = criterion(outputs, labels)

        # Prediction
        _, predictions = torch.max(outputs, dim=-1)
        # pred == y returns a ByteTensor, which has only an 8-bit range. Hence, after a particular batch-size, the sum may overflow
        # and hence shoot the wrong results.
        correct_count = (predictions == labels).double().sum().data[0]
        accuracy = float(correct_count) / args.batch_size

        # best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy

        # forward, backward & optimization
        loss.backward()
        optimizer.step()
        t1 = time.time()
        duration_estimate = t1 - t0

        # print statistics
        running_loss += loss.data[0]
        running_accuracy += accuracy
        if iteration % args.batch_per_log == 0:
            print('Estimated time for each batch: {:.4f} sec.\n'.format(duration_estimate), end=' ')
            print((
                'epoch {:2d} ' + '|| batch {:2d} of {:2d} ||' + ' Loss: {:.4f} ||' + ' Batch-Accuracy: {:.4f} ||\n').format(
                epoch + 1, iteration, num_batches, running_loss / args.batch_per_log, accuracy), end=' ')
            running_loss = 0.0

    # Print the averaged accuracy for epoch
    print('The averaged accuracy for each epoch: {:.4f}.\n'.format(100.0 * running_accuracy / num_batches), end=' ')
    if int(epoch + 1) % args.epochs_per_save == 0:
        # Save the model after some epochs.
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_accuracy': best_accuracy,
            'optimizer': optimizer.state_dict(),
        }, is_best=accuracy == best_accuracy, filename=os.path.join(args.save_folder, 'net_epoch_' +
                                                                    str(epoch + 1)) + '.pth')

# Save the final model at the end of training.
save_checkpoint({
    'epoch': epoch + 1,
    'state_dict': model.state_dict(),
    'best_accuracy': best_accuracy,
    'optimizer': optimizer.state_dict(),
}, is_best=accuracy == best_accuracy, filename=os.path.join(args.save_folder, 'net_final') + '.pth')
print('Finished Training')
