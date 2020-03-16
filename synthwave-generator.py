"""
Given an input wav file containing synthwave audio, this script
trains a encoder decoder architecture base LSTM to use the decoder
part as a music generator
"""
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav_ops
import time
from tqdm import tqdm
import os
import pickle
import shutil
# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

###### Parameters Definition ######
# training params
num_epochs = 1
training_size = 0.95
batch_size = 256
batch_save_frequency = 0.01
total_batches_to_save = 5000

# data params
num_input_steps = 44100//20
num_output_steps = num_input_steps//2
num_channels = 2 # check by reading the file separately in python

do_data_prep = False
do_model_training = True
do_model_prediction = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'

np.random.seed(42)

############ Data Prep ############
if(do_data_prep):
    # read the wav file
    sample_rate, data = wav_ops.read('wav_file.wav')
    if(os.path.exists('data')):
        # clean out the directory
        shutil.rmtree('data')
    # make directory again/first time
    os.mkdir('data')
    batches_saved = 0
    # start writing mini batches into files which can be read up while training
    for i in tqdm(range((data.shape[0] - num_input_steps - num_output_steps)//batch_size)):
        # only write every 10th batch
        if(np.random.random() > batch_save_frequency):
            continue
        if(batches_saved > total_batches_to_save):
            break
        start_idx = i * batch_size
        end_idx = start_idx + num_input_steps + num_output_steps
        X = data[start_idx:end_idx - num_output_steps]
        y = data[start_idx + num_input_steps:end_idx]
        # convert the data from num_points X num_channels to
        # num_points X timestamp X num_channels
        data_strides = data.strides
        X = np.lib.stride_tricks.as_strided(X,
               shape=(batch_size, num_input_steps, num_channels),
               strides=(data_strides[0], data_strides[0], data_strides[1]),
               writeable=False)
        y = np.lib.stride_tricks.as_strided(y,
               shape=(batch_size, num_output_steps, num_channels),
               strides=(data_strides[0], data_strides[0], data_strides[1]),
               writeable=False)
        # normalize data to be between -1 and 1
        X = X/np.iinfo(data.dtype).max

        # normalize the targets between -1 and 1
        y = y/np.iinfo(data.dtype).max

        # save to disk
        with open('data/batch_{:010d}'.format(i), 'wb') as f:
            pickle.dump((X, y, sample_rate, np.iinfo(data.dtype).max), f)

        batches_saved += 1
        if((batches_saved+1)%100 == 0):
            print('Saved {:d} batches'.format(batches_saved+1))

######## Model Definition #########
if(do_model_training or do_model_prediction):
    # define separate classes for encoder, decoder and seq2seq model
    class Encoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, n_layers):
            super(Encoder, self).__init__()
            # add a simple RNN cell
            self._rnn1 = nn.RNN(input_size=input_dim, hidden_size=hidden_dim,
                               num_layers=n_layers, batch_first=True)

        def forward(self, input_data):
            output, hidden = self._rnn1(input_data)
            return output, hidden

    class Decoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, n_layers, num_channels):
            super(Decoder, self).__init__()
            # add a simple RNN cell
            self._rnn1 = nn.RNN(input_size=input_dim, hidden_size=hidden_dim,
                               num_layers=n_layers, batch_first=True)
            self.fc1 = nn.Linear(in_features=hidden_dim, out_features=num_channels)

        def forward(self, input_data, hidden_state):
            # for first pass, use the input data into rnn
            output, hidden_state = self._rnn1(input_data, hidden_state)
            # apply the fully connected layer
            output = self.fc1(output)
            return output, hidden_state

    class Seq2Seq(nn.Module):
        # pass the encoder and decoder here
        def __init__(self, encoder, decoder):
            super(Seq2Seq, self).__init__()
            # initialize the encoder and decoder
            self._encoder = encoder
            self._decoder = decoder

        def forward(self, input_data, num_channels=2, num_timesteps=1):
            # do a forward pass through encoder and decoder
            # final output
            output_final = torch.zeros(input_data.size(0), num_timesteps, 
                                       num_channels, device=device)
            # for training
            output, hidden_state = self._encoder(input_data)
            # first instance of output should be zeros
            output = torch.zeros(input_data.size(0), 1, num_channels,
                                 device=device)
            # for num of time steps, keep running the decoder
            for i in range(num_timesteps):
                output, hidden_state = self._decoder(output, hidden_state)
                output_final[:, i, :] = output.view(-1, num_channels)
            # return only the output
            return output_final

######### Model Training ##########
if(do_model_training):
    # define the models
    hidden_dim = num_channels*2
    encoder = Encoder(input_dim=num_channels, hidden_dim=hidden_dim, n_layers=1).to(device)
    decoder = Decoder(input_dim=num_channels, hidden_dim=hidden_dim, n_layers=1, 
                      num_channels=num_channels).to(device)
    model = Seq2Seq(encoder, decoder).to(device)

    # define optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # define loss function
    loss_fn = nn.MSELoss()

    # get all the files available in the data directory
    if(not os.path.exists('data')):
        print('data directory is not available')
        sys.exit()
    else:
        for _, _, file_list in os.walk('data'):
            pass
    # perform training
    for epoch_num in range(num_epochs):
        epoch_training_loss = 0
        epoch_validation_loss = 0
        # shuffle the file list to remove any associated bias
        np.random.shuffle(file_list)
        for i in tqdm(range(len(file_list))):
            # determine whether the index is for train or validation
            training_index = True
            if(i > len(file_list) * training_size):
                training_index = False
            # set the gradients to zero before start of every batch
            optimizer.zero_grad()
            # read the batch for training
            with open(os.path.join('data', file_list[i]), 'rb') as f:
                X, y, _, _ = pickle.load(f)
            # calculate indices for the training sample
            X = torch.from_numpy(X).float().to(device)
            y = torch.from_numpy(y).float().to(device)
            # do forward pass
            y_pred = model(X, num_channels, num_output_steps)
            # calculate loss
            loss = loss_fn(y_pred, y)
            if(training_index):
                # calculate gradients
                loss.backward()
                # step the optimizer
                optimizer.step()
                # update total epoch loss
                epoch_training_loss += loss.item()
            else:
                epoch_validation_loss += loss.item()
        
        epoch_training_loss /= len(file_list) * training_size
        epoch_training_loss /= len(file_list) * (1 - training_size)

        print(str('Epoch {:0' + str(int(np.log10(num_epochs))) + 'd} | Training Loss : {:.5f} | Validation Loss : {:.5f}')\
              .format(epoch_num, epoch_training_loss, epoch_validation_loss))

        # save the model
        torch.save(model.state_dict(), 'models/seq2seq_model_epoch_{:05d}'.format(epoch_num))
