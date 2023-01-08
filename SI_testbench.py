import numpy as np
from torch import cuda, no_grad, Tensor, save, load, flatten
from torch.nn import Module, Linear, ReLU, Conv2d, BatchNorm2d, MaxPool2d, Dropout, ELU
from torch.nn import CrossEntropyLoss, Softmax, Conv1d, BatchNorm1d
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from typing import Tuple
from sklearn.metrics import accuracy_score
from copy import deepcopy
import sys
import os
import librosa
import time



class RawNet(Module):

    def __init__(self,
                 conv_1_in_dim: int,
                 conv_1_out_dim: int,
                 num_norm_features_1: int,
                 mp_1_kernel_size: Tuple[int, int],
                 conv_2_in_dim: int,
                 conv_2_out_dim: int,
                 num_norm_features_2: int,
                 mp_2_kernel_size: Tuple[int, int],
                 conv_3_in_dim: int,
                 conv_3_out_dim: int,
                 num_norm_features_3: int,
                 mp_3_kernel_size: Tuple[int, int],

                 linear_1_input_dim: int,
                 linear_1_output_dim: int,
                 linear_2_input_dim: int,
                 linear_2_output_dim: int,
                 linear_3_input_dim: int,
                 linear_3_output_dim: int,

                 zero_padding: int,
                 kernel_size: int,
                 dropout: float) \
            -> None:
        super().__init__()
        
        
        self.encoder_conv_layer_1 = Conv1d(in_channels=1, out_channels=40, kernel_size=10, stride=5, padding=3)
        self.encoder_batch_normalization_1 = BatchNorm1d(40)
        
        self.encoder_conv_layer_2 = Conv1d(in_channels=40, out_channels=40, kernel_size=8, stride=4, padding=2)
        self.encoder_batch_normalization_2 = BatchNorm1d(40)
        
        self.encoder_conv_layer_3 = Conv1d(in_channels=40, out_channels=40, kernel_size=4, stride=2, padding=1)
        self.encoder_batch_normalization_3 = BatchNorm1d(40)
        
        self.encoder_conv_layer_4 = Conv1d(in_channels=40, out_channels=40, kernel_size=4, stride=2, padding=1)
        self.encoder_batch_normalization_4 = BatchNorm1d(40)
        
        self.encoder_conv_layer_5 = Conv1d(in_channels=40, out_channels=40, kernel_size=4, stride=2, padding=1)
        self.encoder_batch_normalization_5 = BatchNorm1d(40)
        
        #################################################################
        
        self.conv_layer_1 = Conv2d(in_channels=conv_1_in_dim,
                                   out_channels=conv_1_out_dim,
                                   kernel_size=kernel_size,
                                   padding=zero_padding)

        self.batch_normalization_1 = BatchNorm2d(num_norm_features_1)
        self.maxpooling_1 = MaxPool2d(
            mp_1_kernel_size)  # Default value of stride is kernel_size

        self.conv_layer_2 = Conv2d(in_channels=conv_2_in_dim,
                                   out_channels=conv_2_out_dim,
                                   kernel_size=kernel_size,
                                   padding=zero_padding)

        self.batch_normalization_2 = BatchNorm2d(num_norm_features_2)
        self.maxpooling_2 = MaxPool2d(
            mp_2_kernel_size)  # Default value of stride is kernel_size

        self.conv_layer_3 = Conv2d(in_channels=conv_3_in_dim,
                                   out_channels=conv_3_out_dim,
                                   kernel_size=kernel_size,
                                   padding=zero_padding)

        self.batch_normalization_3 = BatchNorm2d(num_norm_features_3)
        self.maxpooling_3 = MaxPool2d(
            mp_3_kernel_size)  # Default value of stride is kernel_size


        #################################################################

        self.linear_layer_1 = Linear(in_features=linear_1_input_dim,
                                     out_features=linear_1_output_dim)

        self.linear_layer_2 = Linear(in_features=linear_2_input_dim,
                                     out_features=linear_2_output_dim)

        self.linear_layer_3 = Linear(in_features=linear_3_input_dim,
                                     out_features=linear_3_output_dim)

        self.non_linearity_relu = ReLU()
        self.non_linearity_elu = ELU()
        self.dropout = Dropout(dropout)

    def forward(self, X: Tensor) -> Tensor:
        # Make the batches of size [batch_size, audio_window_length] into size
        # [batch_size, 1, audio_window_length] by adding a dummy dimension
        X = X.unsqueeze(1)
        
        # The encoder convolutional layers
        X = self.non_linearity_relu(self.encoder_batch_normalization_1(self.encoder_conv_layer_1(X)))
        X = self.non_linearity_relu(self.encoder_batch_normalization_2(self.encoder_conv_layer_2(X)))
        X = self.non_linearity_relu(self.encoder_batch_normalization_3(self.encoder_conv_layer_3(X)))
        X = self.non_linearity_relu(self.encoder_batch_normalization_4(self.encoder_conv_layer_4(X)))
        X = self.non_linearity_relu(self.encoder_batch_normalization_5(self.encoder_conv_layer_5(X)))
        
        # Make the batches of size [batch_size, num_features, num_frames] into size
        # [batch_size, 1, num_frames, num_features] by switching the last two dimensions
        # and by adding a dummy dimension
        X = X.permute(0, 2, 1)
        X = X.unsqueeze(1)
        
        # The convolutional layers
        X = self.non_linearity_relu(self.batch_normalization_1(self.conv_layer_1(X)))
        X = self.dropout(self.maxpooling_1(X))
        
        X = self.non_linearity_relu(self.batch_normalization_2(self.conv_layer_2(X)))
        X = self.dropout(self.maxpooling_2(X))
        X = self.non_linearity_relu(self.batch_normalization_3(self.conv_layer_3(X)))
        X = self.dropout(self.maxpooling_3(X))

        # Flatten before the linear layers
        X = flatten(X, start_dim=1, end_dim=-1)

        # The linear layers
        X = self.non_linearity_elu(self.linear_layer_1(X))
        X = self.dropout(X)
        X = self.non_linearity_elu(self.linear_layer_2(X))
        X = self.dropout(X)
        X = self.non_linearity_elu(self.linear_layer_3(X))

        # Return the output
        return X





class Net(Module):

    def __init__(self,
                 conv_1_in_dim: int,
                 conv_1_out_dim: int,
                 num_norm_features_1: int,
                 mp_1_kernel_size: Tuple[int, int],
                 conv_2_in_dim: int,
                 conv_2_out_dim: int,
                 num_norm_features_2: int,
                 mp_2_kernel_size: Tuple[int, int],
                 conv_3_in_dim: int,
                 conv_3_out_dim: int,
                 num_norm_features_3: int,
                 mp_3_kernel_size: Tuple[int, int],

                 linear_1_input_dim: int,
                 linear_1_output_dim: int,
                 linear_2_input_dim: int,
                 linear_2_output_dim: int,
                 linear_3_input_dim: int,
                 linear_3_output_dim: int,

                 zero_padding: int,
                 kernel_size: int,
                 dropout: float) \
            -> None:
        super().__init__()

        self.conv_layer_1 = Conv2d(in_channels=conv_1_in_dim,
                                   out_channels=conv_1_out_dim,
                                   kernel_size=kernel_size,
                                   padding=zero_padding)

        self.batch_normalization_1 = BatchNorm2d(num_norm_features_1)
        self.maxpooling_1 = MaxPool2d(
            mp_1_kernel_size)  # Default value of stride is kernel_size

        self.conv_layer_2 = Conv2d(in_channels=conv_2_in_dim,
                                   out_channels=conv_2_out_dim,
                                   kernel_size=kernel_size,
                                   padding=zero_padding)

        self.batch_normalization_2 = BatchNorm2d(num_norm_features_2)
        self.maxpooling_2 = MaxPool2d(
            mp_2_kernel_size)  # Default value of stride is kernel_size

        self.conv_layer_3 = Conv2d(in_channels=conv_3_in_dim,
                                   out_channels=conv_3_out_dim,
                                   kernel_size=kernel_size,
                                   padding=zero_padding)

        self.batch_normalization_3 = BatchNorm2d(num_norm_features_3)
        self.maxpooling_3 = MaxPool2d(
            mp_3_kernel_size)  # Default value of stride is kernel_size


        #################################################################

        self.linear_layer_1 = Linear(in_features=linear_1_input_dim,
                                     out_features=linear_1_output_dim)

        self.linear_layer_2 = Linear(in_features=linear_2_input_dim,
                                     out_features=linear_2_output_dim)

        self.linear_layer_3 = Linear(in_features=linear_3_input_dim,
                                     out_features=linear_3_output_dim)

        self.non_linearity_relu = ReLU()
        self.non_linearity_elu = ELU()
        self.dropout = Dropout(dropout)

    def forward(self, X: Tensor) -> Tensor:
        # Make the batches of size [batch_size, num_frames, num_features] into size
        # [batch_size, 1, num_frames, num_features] by adding a dummy dimension
        X = X.unsqueeze(1)

        # The convolutional layers
        X = self.non_linearity_relu(
            self.batch_normalization_1(self.conv_layer_1(X)))
        X = self.dropout(self.maxpooling_1(X))
        X = self.non_linearity_relu(
            self.batch_normalization_2(self.conv_layer_2(X)))
        X = self.dropout(self.maxpooling_2(X))
        X = self.non_linearity_relu(
            self.batch_normalization_3(self.conv_layer_3(X)))
        X = self.dropout(self.maxpooling_3(X))

        # Flatten before the linear layers
        X = flatten(X, start_dim=1, end_dim=-1)

        # The linear layers
        X = self.non_linearity_elu(self.linear_layer_1(X))
        X = self.dropout(X)
        X = self.non_linearity_elu(self.linear_layer_2(X))
        X = self.dropout(X)
        X = self.non_linearity_elu(self.linear_layer_3(X))

        # Return the output
        return X







class Dataset_raw_audio(Dataset):

    def __init__(self, train_val_test='train', train_test_ratio=0.8,
                 train_val_ratio=0.75, random_seed=22, noise_level=None,
                 file_dir='./train-clean-100', data_sampling_rate=1.00):
        super().__init__()

        f = open("SPEAKERS.TXT", mode = 'r')

        id_list = []
        for row in f.readlines():
            if 'train-clean-100' in row and float(row[30:35]) >= 20.00:
                id_list.append(int(row[0:4]))
                
        id_list = sorted(id_list, key=lambda x: int(x))

        # Find out our FLAC files in the given directory
        paths_flac = []
        id_list[:] = [str(i) for i in id_list]
        first = True
        try:
            for root, dirs, files in os.walk(file_dir):
                if first:
                    dirs[:] = [d for d in dirs if d in id_list]
                    first = False
                if len(files) > 0:
                    for file in files:
                        paths_flac.append(os.path.join(root, file))
        except FileNotFoundError:
            sys.exit(f'Given .flac file directory {file_dir} does not exist!')
        
        self.id_list = id_list

        # Clean the list if there are other files than .wav files
        flac_path_names = [filename for filename in paths_flac if
                          filename.endswith('.flac')]
        flac_path_names = np.array(
            sorted(flac_path_names, key=lambda x: int(x.split('\\')[1])))
        del paths_flac
        
        if data_sampling_rate < 1.00:
            # We randomly select a subset of the data
            np.random.seed(3*random_seed)
            num_sampled = int(data_sampling_rate * len(flac_path_names))
            flac_path_names = np.random.choice(flac_path_names, num_sampled, replace=False)

        # Split our data into a train, validation, and test set
        np.random.seed(random_seed)
        mask_traintest_split = np.random.rand(
            len(flac_path_names)) <= train_test_ratio
        trainval_files = flac_path_names[mask_traintest_split]
        test_files = flac_path_names[~mask_traintest_split]
        np.random.seed(
            2 * random_seed)  # We use a different random seed for splitting trainval_files
        mask_trainval_split = np.random.rand(
            len(trainval_files)) <= train_val_ratio
        train_files = trainval_files[mask_trainval_split]
        val_files = trainval_files[~mask_trainval_split]

        if train_val_test == 'train':
            file_names = train_files
        elif train_val_test == 'validation':
            file_names = val_files
        else:
            file_names = test_files
        self.filenames = file_names
        self.train_val_test = train_val_test
        self.noise_level = noise_level

    def __getitem__(self, index):
        # we take a random part of the audio file of length audio_window_length
        x, fs = librosa.core.load(self.filenames[index], sr=None)
        if self.noise_level is not None:
            if self.noise_level == 'low':
                target_snr_db = 20
            elif self.noise_level == 'medium':
                target_snr_db = 10
            else:
                target_snr_db = 0
            sig_avg_db = 10 * np.log10(np.mean(x ** 2))
            noise_avg_db = sig_avg_db - target_snr_db
            noise_avg_watts = 10 ** (noise_avg_db / 10)
            mean_noise = 0
            if self.train_val_test == 'test':
                np.random.seed(32)
            noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x))
            x += noise
        
        # set audio_window_length to 1.4 * 16000 to include every flac-file
        audio_window_length = int(1.4 * 16000)
        if self.train_val_test == 'test':
            np.random.seed(12)
        part_index = np.random.randint(len(x) - audio_window_length + 1)
        feat = x[part_index:(part_index + audio_window_length)]
        
        label = self.filenames[index].split('\\')[1]
        label_index = self.id_list.index(label)

        return feat, label_index

    def __len__(self) -> int:
        return len(self.filenames)






class Dataset_logmel(Dataset):

    def __init__(self, train_val_test='train', train_test_ratio=0.8,
                 train_val_ratio=0.75, random_seed=22, noise_level=None,
                 file_dir='./train-clean-100', data_sampling_rate=1.00):
        super().__init__()
        self.longest_sample_length = 141

        f = open("SPEAKERS.TXT", mode = 'r')


        id_list = []
        for row in f.readlines():
            if 'train-clean-100' in row and float(row[30:35]) >= 20.00:
                id_list.append(int(row[0:4]))
        
        id_list = sorted(id_list, key=lambda x: int(x))

        # Find out our WAV files in the given directory
        paths_flac = []
        id_list[:] = [str(i) for i in id_list]
        first = True
        try:
            for root, dirs, files in os.walk(file_dir):
                if first:
                    dirs[:] = [d for d in dirs if d in id_list]
                    first = False
                if len(files) > 0:
                    for file in files:
                        paths_flac.append(os.path.join(root, file))
        except FileNotFoundError:
            sys.exit(f'Given .flac file directory {file_dir} does not exist!')
        
        self.id_list = id_list

        # Clean the list if there are other files than .wav files
        flac_path_names = [filename for filename in paths_flac if
                          filename.endswith('.flac')]
        flac_path_names = np.array(
            sorted(flac_path_names, key=lambda x: int(x.split('\\')[1])))
        del paths_flac
        
        if data_sampling_rate < 1.00:
            # We randomly select a subset of the data
            np.random.seed(3*random_seed)
            num_sampled = int(data_sampling_rate * len(flac_path_names))
            flac_path_names = np.random.choice(flac_path_names, num_sampled, replace=False)

        # Split our data into a train, validation, and test set
        np.random.seed(random_seed)
        mask_traintest_split = np.random.rand(
            len(flac_path_names)) <= train_test_ratio
        trainval_files = flac_path_names[mask_traintest_split]
        test_files = flac_path_names[~mask_traintest_split]
        np.random.seed(
            2 * random_seed)  # We use a different random seed for splitting trainval_files
        mask_trainval_split = np.random.rand(
            len(trainval_files)) <= train_val_ratio
        train_files = trainval_files[mask_trainval_split]
        val_files = trainval_files[~mask_trainval_split]

        if train_val_test == 'train':
            file_names = train_files
        elif train_val_test == 'validation':
            file_names = val_files
        else:
            file_names = test_files
        self.filenames = file_names
        self.train_val_test = train_val_test
        self.noise_level = noise_level



    def __getitem__(self, index):

        x, fs = librosa.core.load(self.filenames[index], sr=None)
        if self.noise_level is not None:
            if self.noise_level == 'low':
                target_snr_db = 20
            elif self.noise_level == 'medium':
                target_snr_db = 10
            else:
                target_snr_db = 0
            sig_avg_db = 10 * np.log10(np.mean(x ** 2))
            noise_avg_db = sig_avg_db - target_snr_db
            noise_avg_watts = 10 ** (noise_avg_db / 10)
            mean_noise = 0
            if self.train_val_test == 'test':
                np.random.seed(32)
            noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x))
            x += noise
        
        num_fft = int(0.03 * fs)
        shift = int(0.01 * fs)

        # Extract the log-mel spectrogram
        melspec = librosa.feature.melspectrogram(x, sr=fs, n_fft=num_fft,
                                                 hop_length=shift,
                                                 n_mels=40).T
        logmel = librosa.core.power_to_db(melspec)
        # We make every sample the same length by taking the mean over the time dimension
        if self.train_val_test == 'test':
            np.random.seed(12)
        part_index = np.random.randint(len(logmel) - self.longest_sample_length + 1)
        feat = logmel[part_index:(part_index + self.longest_sample_length)]
        label = self.filenames[index].split('\\')[1]
        label_index = self.id_list.index(label)
        return feat, label_index

    def __len__(self) -> int:
        return len(self.filenames)








if __name__ == '__main__':
    
    name_of_log_textfile = 'experiment_2.txt'
    
    file = open(name_of_log_textfile, 'w')
    file.close()
    
    device = "cuda" if cuda.is_available() else "cpu"
    with open(name_of_log_textfile, 'a') as f:
        f.write(f'Process on {device}\n\n')

    features = 'raw_audio'  # Options: 'raw_audio', 'logmel'
    dropout = 0.1
    batch_size = 32
    max_epochs = 1000
    learning_rate = 1e-4
    
    train_model = True
    test_model = True
    
    save_best_model = True
    cnn_best_model_name = 'best_cnn_model_experiment_2.pt'
    continue_training = False
    
    # Options: 1.00, 0.5 and 0.25 (equal to using 100% / 50% / 25% of the data)
    data_sampling_rate = 1.00
    
    # Options: None, 'low', 'medium', 'high'
    noise_level = 'low'

    num_input_channels = 1
    conv_out_dim_1 = 64
    conv_out_dim_2 = 64
    conv_out_dim_3 = 64
    kernel_size_mp_1 = (4, 2)
    kernel_size_mp_2 = (5, 2)
    kernel_size_mp_3 = (7, 2)
    linear_1_input_dim = 320
    linear_1_output_dim = 256
    linear_2_input_dim = 256
    linear_2_output_dim = 256
    linear_3_input_dim = 256
    output_dim = 231  # 231 total speakers

    kernel_size = 3
    zero_pad_size = 1

    # Instantiate our DNN
    if features == 'logmel':
        CNN_model = Net(conv_1_in_dim=num_input_channels,
                        conv_1_out_dim=conv_out_dim_1,
                        num_norm_features_1=conv_out_dim_1,
                        mp_1_kernel_size=kernel_size_mp_1,
                        conv_2_in_dim=conv_out_dim_1,
                        conv_2_out_dim=conv_out_dim_2,
                        num_norm_features_2=conv_out_dim_2,
                        mp_2_kernel_size=kernel_size_mp_2,
                        conv_3_in_dim=conv_out_dim_2,
                        conv_3_out_dim=conv_out_dim_3,
                        num_norm_features_3=conv_out_dim_3,
                        mp_3_kernel_size=kernel_size_mp_3,
                        linear_1_input_dim=linear_1_input_dim,
                        linear_1_output_dim=linear_1_output_dim,
                        linear_2_input_dim=linear_2_input_dim,
                        linear_2_output_dim=linear_2_output_dim,
                        linear_3_input_dim=linear_3_input_dim,
                        linear_3_output_dim=output_dim,
                        zero_padding=zero_pad_size,
                        kernel_size=kernel_size,
                        dropout=dropout)
    else:
        CNN_model = RawNet(conv_1_in_dim=num_input_channels,
                           conv_1_out_dim=conv_out_dim_1,
                           num_norm_features_1=conv_out_dim_1,
                           mp_1_kernel_size=kernel_size_mp_1,
                           conv_2_in_dim=conv_out_dim_1,
                           conv_2_out_dim=conv_out_dim_2,
                           num_norm_features_2=conv_out_dim_2,
                           mp_2_kernel_size=kernel_size_mp_2,
                           conv_3_in_dim=conv_out_dim_2,
                           conv_3_out_dim=conv_out_dim_3,
                           num_norm_features_3=conv_out_dim_3,
                           mp_3_kernel_size=kernel_size_mp_3,
                           linear_1_input_dim=linear_1_input_dim,
                           linear_1_output_dim=linear_1_output_dim,
                           linear_2_input_dim=linear_2_input_dim,
                           linear_2_output_dim=linear_2_output_dim,
                           linear_3_input_dim=linear_3_input_dim,
                           linear_3_output_dim=output_dim,
                           zero_padding=zero_pad_size,
                           kernel_size=kernel_size,
                           dropout=dropout)

    CNN_model = CNN_model.to(device)

    optimizer = Adam(params=CNN_model.parameters(), lr=learning_rate)

    loss_function = CrossEntropyLoss()
    
    # Initialize the best versions of our models
    best_model_cnn = None
    
    if continue_training:
        # Load our model weights in order to continue training (or if we just want to test our model)
        try:
            with open(name_of_log_textfile, 'a') as f:
                f.write('Loading model from file...\n')
                f.write(f'Loading model {cnn_best_model_name}...\n')
            CNN_model.load_state_dict(load(cnn_best_model_name, map_location=device))
            best_model_cnn = deepcopy(CNN_model.state_dict())
            with open(name_of_log_textfile, 'a') as f:
                f.write('Done!\n\n')
        except FileNotFoundError:
            with open(name_of_log_textfile, 'a') as f:
                f.write('An error occurred while loading the file! Training without pretrained model weights...\n\n')

    if features == 'raw_audio':
        dataset = Dataset_raw_audio
    elif features == 'logmel':
        dataset = Dataset_logmel
    else:
        sys.exit('Other features not implemented!')

    params_train = {'batch_size': batch_size,
                    'shuffle': True,
                    'drop_last': False}
    
    params_test = {'batch_size': batch_size,
                    'shuffle': False,
                    'drop_last': False}
    
    with open(name_of_log_textfile, 'a') as f:
        f.write('Initializing data loaders...\n')
    training_set = dataset(train_val_test='train', data_sampling_rate=data_sampling_rate, noise_level=noise_level)
    train_data_loader = DataLoader(training_set, **params_train)
    validation_set = dataset(train_val_test='validation', data_sampling_rate=data_sampling_rate, noise_level=noise_level)
    validation_data_loader = DataLoader(validation_set, **params_train)
    test_set = dataset(train_val_test='test', data_sampling_rate=data_sampling_rate, noise_level=noise_level)
    test_data_loader = DataLoader(test_set, **params_test)
    with open(name_of_log_textfile, 'a') as f:
        f.write('Done!\n\n')
    
    # Variables for early stopping
    patience = 30
    lowest_validation_loss = 1e10
    best_validation_epoch = 0
    patience_counter = 0
    
    # Flag for indicating if max epochs are reached
    max_epochs_reached = 1
    
    if train_model:
        with open(name_of_log_textfile, 'a') as f:
            f.write('Starting training...\n')
        for epoch in range(1, max_epochs + 1):
            start_time = time.time()
            epoch_loss_training = []
            epoch_loss_validation = []
            
            # Indicate that we are in training mode, so e.g. dropout will function
            CNN_model.train()
            
            for train_data in train_data_loader:
    
                optimizer.zero_grad()
    
                X, Y_true = [element.to(device) for element in train_data]

                Y_predicted = CNN_model(X)
                loss = loss_function(input=Y_predicted, target=Y_true)
                
                # Perform the backward pass and update the weights of our neural network
                loss.backward()
                optimizer.step()
                
                epoch_loss_training.append(loss.item())
            
            # Indicate that we are in evaluation mode, so e.g. dropout will not function
            CNN_model.eval()
            
            # Make PyTorch not calculate the gradients, so everything will be much faster
            with no_grad():
                for validation_data in validation_data_loader:
                    X, Y_true = [element.to(device) for element in validation_data]
                    Y_predicted = CNN_model(X)
                    loss = loss_function(input=Y_predicted, target=Y_true)
                    epoch_loss_validation.append(loss.item())
            
            # Calculate mean losses
            epoch_loss_training = np.array(epoch_loss_training).mean()
            epoch_loss_validation = np.array(epoch_loss_validation).mean()
            
            # Check early stopping conditions
            if epoch_loss_validation < lowest_validation_loss:
                lowest_validation_loss = epoch_loss_validation
                patience_counter = 0
                best_model_cnn = deepcopy(CNN_model.state_dict())
                best_validation_epoch = epoch
                if save_best_model:
                    save(best_model_cnn, cnn_best_model_name)
            else:
                patience_counter += 1
            
            end_time = time.time()
            epoch_time = end_time - start_time
            
            with open(name_of_log_textfile, 'a') as f:
                f.write(f'Epoch: {epoch:04d} | Mean training loss: {epoch_loss_training:6.4f} | '
                  f'Mean validation loss: {epoch_loss_validation:6.4f} (lowest: {lowest_validation_loss:6.4f}) | '
                  f'Duration: {epoch_time:4.2f} seconds\n')
            
            # If patience counter is fulfilled, stop the training
            if patience_counter >= patience:
                max_epochs_reached = 0
                break
            
        if max_epochs_reached:
            with open(name_of_log_textfile, 'a') as f:
                f.write('\nMax number of epochs reached, stopping training\n\n')
        else:
            with open(name_of_log_textfile, 'a') as f:
                f.write('\nExiting due to early stopping\n\n')
        
        if best_model_cnn is None:
            with open(name_of_log_textfile, 'a') as f:
                f.write('\nNo best model. The criteria for the lowest acceptable validation accuracy not satisfied!\n\n')
            sys.exit('No best model, exiting...')
        else:
            with open(name_of_log_textfile, 'a') as f:
                f.write(f'\nBest epoch {best_validation_epoch} with validation loss {lowest_validation_loss}\n\n')
        
    if test_model:
        with open(name_of_log_textfile, 'a') as f:
            f.write('\nStarting testing... => ')
            
        # Load the best version of the model
        try:
            CNN_model.load_state_dict(load(cnn_best_model_name, map_location=device))
        except (FileNotFoundError, RuntimeError):
            CNN_model.load_state_dict(best_model_cnn)
                
        testing_loss = []
        testing_accuracy = []
        smax = Softmax(dim=1)
        CNN_model.eval()
        with no_grad():
            for test_data in test_data_loader:
                X, Y_true = [element.to(device) for element in test_data]
                Y_predicted = CNN_model(X)
                loss = loss_function(input=Y_predicted, target=Y_true)
                testing_loss.append(loss.item())
                Y_predicted_smax_np = smax(Y_predicted).detach().cpu().numpy()
                predictions = np.argmax(Y_predicted_smax_np, axis=1)
                accuracy = accuracy_score(Y_true.detach().cpu().numpy(), predictions)
                testing_accuracy.append(accuracy)
            testing_loss = np.array(testing_loss).mean()
            testing_accuracy = np.array(testing_accuracy).mean()
            with open(name_of_log_textfile, 'a') as f:
                f.write(f'Testing loss: {testing_loss:7.4f} | Testing accuracy: {testing_accuracy:7.4f}\n\n\n')
            
            
