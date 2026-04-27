"""
CsiNet Implementation for CSI Compression and Reconstruction
This script builds and trains a residual-based autoencoder (CsiNet) for Channel State Information (CSI) compression
in both indoor and outdoor wireless environments. It supports different compression rates via adjustable encoded dimensions,
evaluates performance with NMSE (Normalized Mean Square Error) and correlation coefficient,
and saves training logs, model weights, and reconstruction visualizations.
"""
# 314513025 (b 小題)
# [個人註解]: 標註有"[個人註解]"即為學生個人註解
# [個人註解]: Notes marked "[個人註解]" are student's personal notes.

import os
import tensorflow as tf
from keras.layers import Input, Dense, BatchNormalization, Reshape, Conv2D, add, LeakyReLU
from keras.models import Model
from keras.callbacks import TensorBoard, Callback
import scipy.io as sio 
import numpy as np
import math
import time
import matplotlib.pyplot as plt

# [個人註解]: 確保相容性
try:
    tf.reset_default_graph()
except AttributeError:
    tf.compat.v1.reset_default_graph()

# [個人註解]: 確保結果資料夾存在
if not os.path.exists('result'):
    os.makedirs('result')
# ────────────────────────  Environment Configuration  ───────────────────────── #
envir = 'indoor' #'indoor' or 'outdoor' -> Select wireless propagation environment
# ────────────────────────  CSI Image Parameters  ────────────────────────────── #
img_height = 32        # CSI matrix height (spatial dimension)
img_width = 32         # CSI matrix width (frequency dimension)
img_channels = 2       # Real and imaginary parts of CSI (2 channels)
img_total = img_height*img_width*img_channels  # Total CSI feature dimensions
# ────────────────────────  Network Hyperparameters  ─────────────────────────── #
residual_num = 2       # Number of residual blocks in the decoder
encoded_dim = 512      # Compress rate=1/4->dim.=512, 1/16->128, 1/32->64, 1/64->32
# ────────────────────────  CsiNet Autoencoder Construction  ─────────────────── #
def residual_network(x, residual_num, encoded_dim):
    """
    Build the residual-based encoder-decoder network for CsiNet.
    Encoder: Conv2D -> Flatten -> Dense (compression to encoded_dim)
    Decoder: Dense -> Reshape -> Residual Blocks -> Conv2D (reconstruction to original CSI shape)
    Args:
        x (tensor): Input CSI tensor (shape: [batch, img_channels, img_height, img_width])
        residual_num (int): Number of residual blocks in the decoder
        encoded_dim (int): Dimension of the compressed CSI feature vector
    Returns:
        tensor: Reconstructed CSI tensor with sigmoid activation (range [0,1])
    """
    def add_common_layers(y):
        """Add BatchNormalization and LeakyReLU activation (shared in residual blocks)."""
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        return y

    def residual_block_decoded(y):
        """Residual block for decoder: 3x3 Conv2D stack with shortcut connection."""
        shortcut = y  # Shortcut for residual connection
        y = Conv2D(8, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
        y = add_common_layers(y)
        
        y = Conv2D(16, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
        y = add_common_layers(y)
        
        y = Conv2D(2, kernel_size=(3, 3), padding='same', data_format='channels_first')(y)
        y = BatchNormalization()(y)
        y = add([shortcut, y])  # Residual connection: skip + conv output
        y = LeakyReLU()(y)
        return y
    
    # Encoder part: initial convolution + flatten + dense compression
    x = Conv2D(2, (3, 3), padding='same', data_format="channels_first")(x)
    x = add_common_layers(x)
    
    x = Reshape((img_total,))(x)  # Flatten CSI tensor to 1D vector
    encoded = Dense(encoded_dim, activation='linear')(x)  # Compress to encoded_dim
    
    # Decoder part: dense decompression + reshape + residual blocks + final convolution
    x = Dense(img_total, activation='linear')(encoded)  # Decompress to original flat dimension
    x = Reshape((img_channels, img_height, img_width,))(x)  # Reshape back to 4D CSI tensor
    for i in range(residual_num):
        x = residual_block_decoded(x)  # Stack residual blocks for reconstruction
    
    x = Conv2D(2, (3, 3), activation='sigmoid', padding='same', data_format="channels_first")(x)  # Output [0,1] for real/imag parts
    return x

# Build input tensor and full autoencoder model
image_tensor = Input(shape=(img_channels, img_height, img_width))  # Input shape for CSI data
network_output = residual_network(image_tensor, residual_num, encoded_dim)  # Reconstructed CSI
autoencoder = Model(inputs=[image_tensor], outputs=[network_output])  # Define full autoencoder
autoencoder.compile(optimizer='adam', loss='mse')  # Compile with Adam optimizer and MSE loss (CSI reconstruction)
print(autoencoder.summary())  # Print network architecture and parameter count
# ────────────────────────  Data Loading and Preprocessing  ──────────────────── #
# Load MATLAB-formatted CSI datasets (train/val/test) for selected environment
#if envir == 'indoor':
#    mat = sio.loadmat('data/DATA_Htrainin.mat') 
#    x_train = mat['HT'] # Training CSI data array
#    mat = sio.loadmat('data/DATA_Hvalin.mat')
#    x_val = mat['HT'] # Validation CSI data array
#    mat = sio.loadmat('data/DATA_Htestin.mat')
#    x_test = mat['HT'] # Test CSI data array
#elif envir == 'outdoor':
#    mat = sio.loadmat('data/DATA_Htrainout.mat') 
#    x_train = mat['HT'] # Training CSI data array
#    mat = sio.loadmat('data/DATA_Hvalout.mat')
#    x_val = mat['HT'] # Validation CSI data array
#    mat = sio.loadmat('data/DATA_Htestout.mat')
#    x_test = mat['HT'] # Test CSI data array

# Convert data to float32 for neural network training
#x_train = x_train.astype('float32')
#x_val = x_val.astype('float32')
#x_test = x_test.astype('float32')

# ────────────────────────  [個人註解]: (c) 混合載入 Dataset 1 到 5 進行訓練 ──────────────────── #
print("\nLoading Data 1 to 5 for Training")

train_list = []
val_list =[]

# [個人註解]: 迴圈讀取 5 個資料集並裝進 List
for ds in range(1, 6):
    mat_train = sio.loadmat(f'data/DATA_Htrainin_ds{ds}.mat') 
    train_list.append(mat_train['HT'].astype('float32'))
    
    mat_val = sio.loadmat(f'data/DATA_Hvalin_ds{ds}.mat')
    val_list.append(mat_val['HT'].astype('float32'))

# [個人註解]: 沿著 batch 維度合併資料
x_train = np.concatenate(train_list, axis=0)
x_val = np.concatenate(val_list, axis=0)

# [個人註解]: 打亂混合後的訓練資料
np.random.seed(25) # [個人註解]: 取我的學號末兩碼當種子 :)
np.random.shuffle(x_train)
np.random.seed(25)
np.random.shuffle(x_val)

# Reshape data to fit channels_first format: [batch, channels, height, width]
x_train = np.reshape(x_train, (len(x_train), img_channels, img_height, img_width))
x_val = np.reshape(x_val, (len(x_val), img_channels, img_height, img_width))

print(f"Mixed Training Data Shape: {x_train.shape}")
print(f"Mixed Validation Data Shape: {x_val.shape}")

# ────────────────────────  Custom Loss Callback  ─────────────────────────────── #
class LossHistory(Callback):
    """Custom Keras Callback to record batch-wise training loss and epoch-wise validation loss."""
    def on_train_begin(self, logs={}):
        """Initialize empty loss lists at training start."""
        self.losses_train = []
        self.losses_val = []

    def on_batch_end(self, batch, logs={}):
        """Append training loss of current batch to list."""
        self.losses_train.append(logs.get('loss'))
        
    def on_epoch_end(self, epoch, logs={}):
        """Append validation loss of current epoch to list."""
        self.losses_val.append(logs.get('val_loss'))
        
# Initialize loss history callback
history = LossHistory()

# ────────────────────────  Model Training  ──────────────────────────────────── #
# Generate unique file name with environment, encoded dimension and current date
# [個人註解]: 加 '_mixed' 避免覆蓋(b)
file = 'CsiNet_'+(envir)+'_dim'+str(encoded_dim)+'_mixed_'+time.strftime('%m_%d')
path = 'result/TensorBoard_%s' %file  # TensorBoard log directory

# Train the autoencoder with CSI data (input = target for reconstruction task)
autoencoder.fit(x_train, x_train,
                epochs=1000,               # Total training epochs
                batch_size=200,            # Mini-batch size
                shuffle=True,              # Shuffle training data per epoch
                validation_data=(x_val, x_val),  # Validation dataset
                callbacks=[history,        # Record loss history
                           TensorBoard(log_dir = path)])  # TensorBoard visualization

# Save training and validation loss to CSV files
filename = 'result/trainloss_%s.csv'%file
loss_history = np.array(history.losses_train)
np.savetxt(filename, loss_history, delimiter=",")

filename = 'result/valloss_%s.csv'%file
loss_history = np.array(history.losses_val)
np.savetxt(filename, loss_history, delimiter=",")

# ────────────────────────  Model Saving  ────────────────────────────────────── #
# Serialize model architecture to JSON file
model_json = autoencoder.to_json()
outfile = "result/model_%s.json"%file
with open(outfile, "w") as json_file:
    json_file.write(model_json)
# Serialize model weights to HDF5 file
outfile = "result/model_%s.h5"%file
autoencoder.save_weights(outfile)

# ────────────────────────  [個人註解]: (c) 測試混合模型在 Dataset 1 到 5 的表現  ──────────────────── #
print("\n\n" + "="*50)
print("training finished :)")
print("="*50)

# [個人註解]: 訓練完混合模型後，依序載入dataset 1~5 測試資料進行評估
for i in range(1, 6):
    print(f"\nEvaluating Mixed Model on Dataset {i}:")
    
    mat = sio.loadmat(f'data/DATA_Htestin_ds{i}.mat')
    x_test = mat['HT'].astype('float32')
    x_test = np.reshape(x_test, (len(x_test), img_channels, img_height, img_width))

    mat_f = sio.loadmat(f'data/DATA_HtestFin_all_ds{i}.mat')
    X_test = mat_f['HF_all']
    X_test = np.reshape(X_test, (len(X_test), img_height, 125))

    # ────────────────────────  Model Inference on Test Data  ────────────────────── #
    # Measure inference time for CSI reconstruction
    tStart = time.time()
    x_hat = autoencoder.predict(x_test)  
    tEnd = time.time()

    # ────────────────────────  Performance Evaluation (NMSE & Correlation)  ──────────────────── #
    # Convert reconstructed/raw CSI from [0,1] to complex domain (-0.5~0.5 for real/imag)
    x_test_real = np.reshape(x_test[:, 0, :, :], (len(x_test), -1))
    x_test_imag = np.reshape(x_test[:, 1, :, :], (len(x_test), -1))
    x_test_C = x_test_real-0.5 + 1j*(x_test_imag-0.5)  # Raw complex CSI

    x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
    x_hat_C = x_hat_real-0.5 + 1j*(x_hat_imag-0.5)  # Reconstructed complex CSI

    # Reshape complex CSI to frequency domain and perform FFT for correlation calculation
    x_hat_F = np.reshape(x_hat_C, (len(x_hat_C), img_height, img_width))
    # Zero-padding + FFT to match original frequency-domain CSI dimension
    X_hat = np.fft.fft(np.concatenate((x_hat_F, np.zeros((len(x_hat_C), img_height, 257-img_width))), axis=2), axis=2)
    X_hat = X_hat[:, :, 0:125]  # Truncate to original 125 frequency bins

    # Calculate correlation coefficient (rho) between original and reconstructed frequency-domain CSI
    n1 = np.sqrt(np.sum(np.conj(X_test)*X_test, axis=1)).astype('float64')  # Norm of original CSI
    n2 = np.sqrt(np.sum(np.conj(X_hat)*X_hat, axis=1)).astype('float64')  # Norm of reconstructed CSI
    aa = abs(np.sum(np.conj(X_test)*X_hat, axis=1))     # Cross term for correlation
    rho = np.mean(aa/(n1*n2), axis=1)                   # Correlation coefficient per sample

    # Reshape for NMSE calculation (Normalized Mean Square Error)
    X_hat = np.reshape(X_hat, (len(X_hat), -1))
    X_test = np.reshape(X_test, (len(X_test), -1))

    # Compute NMSE (in dB) for CSI reconstruction
    power = np.sum(abs(x_test_C)**2, axis=1)    
    mse = np.sum(abs(x_test_C-x_hat_C)**2, axis=1)  
    
    print(f"Mixed Model Results for Dataset {i}:")
    print("NMSE is ", 10*math.log10(np.mean(mse/power)))
    print("Correlation is ", np.mean(rho))
    print ("It cost %f sec" % ((tEnd - tStart)/x_test.shape[0]))

    # Save reconstructed CSI and correlation coefficient to CSV files
    filename = f"result/decoded_{file}_ds{i}.csv"
    x_hat1 = np.reshape(x_hat, (len(x_hat), -1))
    np.savetxt(filename, x_hat1, delimiter=",")

    filename = f"result/rho_{file}_ds{i}.csv"
    np.savetxt(filename, rho, delimiter=",")

    # ────────────────────────  CSI Reconstruction Visualization (Optional) ────────────── #
    '''Plot absolute value of original and reconstructed complex CSI (first 10 samples)'''
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original CSI absolute value
        ax = plt.subplot(2, n, i + 1 )
        x_testplo = abs(x_test[i, 0, :, :]-0.5 + 1j*(x_test[i, 1, :, :]-0.5))
        plt.imshow(np.max(np.max(x_testplo))-x_testplo.T)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.invert_yaxis()
        # Display reconstructed CSI absolute value
        ax = plt.subplot(2, n, i + 1 + n)
        decoded_imgsplo = abs(x_hat[i, 0, :, :]-0.5
                               + 1j*(x_hat[i, 1, :, :]-0.5))
        plt.imshow(np.max(np.max(decoded_imgsplo))-decoded_imgsplo.T)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.invert_yaxis()
    plt.show()
    