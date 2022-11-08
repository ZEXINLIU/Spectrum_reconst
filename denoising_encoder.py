import numpy as np
import torch
from solver_util import gen_sensing_matrix, gen_spectrum
from solver_util import rel_reconst_error
from solver_util import train_model_Encoder
from solver_util import awgn, sparse_basis
from weight_init import init_xavier_uniform
from Nets import AE1
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import l1ls as L


# load data
R = gen_sensing_matrix()
x = gen_spectrum()

# scale x to [0,1] for the convenience of training NNs
scaler_x = MinMaxScaler(feature_range=(0., 1.))
x_01= scaler_x.fit_transform(x)
I_ideal = R.dot(x_01)
snr = 40
seed = 108
I_corrupted = awgn(x=I_ideal, snr=snr, seed=seed)

# transfer data from numpy array to torch tensor
train_data_I = torch.tensor(I_corrupted.T).to(torch.float32) # raw measurements
train_data_x = torch.tensor(x_01.T).to(torch.float32) # true spectrum (added noise)

# split the data to training sets, validation set(contained in training set) and test set by 4:1:1
split = int(np.ceil((4 + 1) / (4 + 1 + 1) * train_data_I.shape[0]))
train_data_I, test_data_I = train_data_I[:split], train_data_I[split:]
train_data_x, test_data_x = train_data_x[:split], train_data_x[split:]

try:
    model = torch.load(f'encoder_dB{snr}.pkl')
except:    
    AE = AE1(input_dim=108, emb_size=32, dropout_rate=0)
    model, train_loss, valid_loss = train_model_Encoder(train_data_x=train_data_I, train_data_y=train_data_I,
                                                        model=AE,
                                                        lr=1e-4,
                                                        batch_size=32,
                                                        patience=1000,
                                                        n_epochs=8000)
    torch.save(model, f'encoder_dB{snr}.pkl')

# prediction
emb, recon_I = model(test_data_I)
recon_I = recon_I.detach().numpy().T

# choosee the sparsifying basis
N, S = R.shape
FWHM = [5, 10, 20, 40, 60, 80, 100]
wavelength = np.linspace(400, 750, S)
Phi = sparse_basis(FWHM, wavelength)

# reconstruct spectrum by sparse recovery
if snr == 30:
    lmbda = 1
elif snr == 35:
    lmbda = 0.5
elif snr == 40:
    lmbda = 0.2

# reconstruct spectrum by sparse recovery
xtrue = test_data_x.detach().numpy().T # xtrue = x_01[:,split:]
try:
    x_recon = np.load(f'encoder_x_recon_dB{snr}.npy')
except:
    x_recon = np.zeros((xtrue.shape))
    for i in range(x_recon.shape[1]):
        [c, status, hist] = L.l1ls(A=R.dot(Phi), y=recon_I[:,i], lmbda=lmbda, tar_gap=1e-2)
        x_recon[:,i] = Phi.dot(c)
    np.save(f'encoder_x_recon_dB{snr}.npy', x_recon)

rel_err = rel_reconst_error(x_recon, xtrue)
err_mean = np.mean(rel_err)
err_std = np.std(rel_err)
print (f'denoising encoder + sparse recovery: relative error mean = {err_mean}')
print (f'denoising encoder + sparse recovery: relative error std = {err_std}')

# snr = 30
# denoising encoder + sparse recovery: relative error mean = 0.11615410668404184
# denoising encoder + sparse recovery: relative error std = 0.04691797501537849

# snr = 35
# denoising encoder + sparse recovery: relative error mean = 0.10132645248855914
# denoising encoder + sparse recovery: relative error std = 0.04546565461978635

# snr = 40
# denoising encoder + sparse recovery: relative error mean = 0.09011088615227153
# denoising encoder + sparse recovery: relative error std = 0.04779910444698741
