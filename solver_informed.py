import numpy as np
import torch
from solver_util import gen_sensing_matrix, gen_spectrum
from solver_util import rel_reconst_error
from solver_util import train_model_Encoder, train_model_L2
from solver_util import awgn, sparse_basis
from weight_init import init_xavier_uniform
from Nets import AE1, Net_NonLinReg
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
snr = 30
seed = 108
I_corrupted = awgn(x=I_ideal, snr=snr, seed=seed)

# choosee the sparsifying basis
N, S = R.shape
FWHM = [5, 10, 20, 40, 60, 80, 100]
wavelength = np.linspace(400, 750, S)
Phi = sparse_basis(FWHM, wavelength)

# reconstruct spectrum by sparse recovery
# lmbda parameters setting for sparse recovery only
if snr == 30:
    lmbda = 1
elif snr == 35:
    lmbda = 0.25
elif snr == 40:
    lmbda = 0.1

# lmbda parameters setting for AdamW with weight decay
# if snr == 30:
    # lmbda = 0.5
# elif snr == 35:
    # lmbda = 0.2
# elif snr == 40:
    # lmbda = 0.02

try:
    x_recon = np.load(f'x_recon_dB{snr}_{lmbda}.npy')
except:
    x_recon = np.zeros((x.shape))
    for i in range(x_recon.shape[1]):
        [c, status, hist] = L.l1ls(A=R.dot(Phi), y=I_corrupted[:,i], lmbda=lmbda, tar_gap=1e-2)
        x_recon[:,i] = Phi.dot(c)
    np.save(f'x_recon_dB{snr}_{lmbda}.npy', x_recon)

rel_err = rel_reconst_error(x_recon, x_01)
err_mean = np.mean(rel_err)
print (f'pure sparse recovery: relative error mean = {err_mean}')
err_std = np.std(rel_err)
print (f'pure sparse recovery: relative error std = {err_std}')

x_extracted = x_recon
# transfer data from numpy array to torch tensor
train_data_x_extracted = torch.tensor(x_extracted.T).to(torch.float32) # raw measurements
train_data_x = torch.tensor(x_01.T).to(torch.float32) # true spectrum (added noise)

# split the data to training sets, validation set(contained in training set) and test set by 4:1:1
split = int(np.ceil((4 + 1) / (4 + 1 + 1) * train_data_x_extracted.shape[0]))
train_data_x_extracted, test_data_x_extracted = train_data_x_extracted[:split], train_data_x_extracted[split:]
train_data_x, test_data_x = train_data_x[:split], train_data_x[split:]

try:
    model = torch.load(f'solver_informed_NN_AdamL2_dB{snr}.pkl')
except:
    net = Net_NonLinReg(input_dim=71, n_hidden1=32, n_hidden2=32, out_dim=71)
    net.apply(init_xavier_uniform)
    model, avg_train_losses, avg_valid_losses = train_model_L2(train_data_x=train_data_x_extracted, train_data_y=train_data_x,
                                                                model=net,
                                                                lbd=1e-6,
                                                                lr=2e-4,
                                                                batch_size=32,
                                                                patience=1000,
                                                                n_epochs=50000,
                                                                weight_decay=0)
    # save model
    torch.save(model, f'solver_informed_NN_AdamWL2_dB{snr}.pkl')

# prediction
x_recon = model(test_data_x_extracted)
x_recon = x_recon.detach().numpy().T

# reconstruct spectrum
xtrue = test_data_x.detach().numpy().T
rel_err = rel_reconst_error(x_recon, xtrue)
err_mean = np.mean(rel_err)
err_std = np.std(rel_err)
print (f'solver informed NN: relative error mean = {err_mean}')
print (f'solver informed NN: relative error std = {err_std}')

import matplotlib.pyplot as plt
# ind = np.random.randint(0, len(rel_err))
ind = 107
plt.plot(wavelength, xtrue[:,ind], '-ko', label='truth')
plt.plot(wavelength, x_recon[:,ind], '-rx', label='sparse + NN (AdamW weight decay)')
plt.legend()
plt.show()

# snr = 30
# pure sparse recovery: relative error mean = 0.1548818385498792
# pure sparse recovery: relative error std = 0.048104348623631214

# Adam
# weight_decay = 0
# Early stopping at epoch: 40369
# train loss: 0.00018671110378550913
# valid loss: 0.0002858992892369214
# solver informed NN: relative error mean = 0.051011499017477036
# solver informed NN: relative error std = 0.05024470016360283

# AdamW (weight decay = 1e-3)
# Early stopping at epoch: 23889
# train loss: 0.00021922559843754724
# valid loss: 0.00033928194898180664
# solver informed NN: relative error mean = 0.05029318481683731
# solver informed NN: relative error std = 0.046629633754491806

# AdamW (weight decay = 1e-4)
# Early stopping at epoch: 26737
# train loss: 0.00021521390461951822
# valid loss: 0.0003027078297842915
# solver informed NN: relative error mean = 0.05072183534502983
# solver informed NN: relative error std = 0.05054520070552826

# AdamW (weight decay = 1e-5)
# Early stopping at epoch: 20245
# train loss: 0.00022490972375688965
# valid loss: 0.0003296808507810864
# solver informed NN: relative error mean = 0.052194755524396896
# solver informed NN: relative error std = 0.04977899044752121

# lmbda = 0.5, AdamW (weight decay = 1e-3)
# Early stopping at epoch: 32361
# train loss: 0.00017561218726982856
# valid loss: 0.00025824112849982665
# solver informed NN: relative error mean = 0.0470363013446331
# solver informed NN: relative error std = 0.04353945329785347

# lmbda = 0.5, AdamW (weight decay = 1e-4)
# Early stopping at epoch: 29510
# train loss: 0.0001719988589791362
# valid loss: 0.00026422787373626814
# solver informed NN: relative error mean = 0.04660189151763916
# solver informed NN: relative error std = 0.04585782065987587

# lmbda = 0.5, AdamW (weight decay = 1e-5)
# Early stopping at epoch: 28767
# train loss: 0.0001922526871032246
# valid loss: 0.00029821454114021943
# solver informed NN: relative error mean = 0.049417734146118164
# solver informed NN: relative error std = 0.04596802964806557


###########################
# snr = 35
# pure sparse recovery: relative error mean = 0.12015595731907883
# pure sparse recovery: relative error std = 0.0400612320962266

# Adam
# weight_decay = 0
# Early stopping at epoch: 29418
# train loss: 0.0001794836374328417
# valid loss: 0.0002656471026259371
# solver informed NN: relative error mean = 0.0463392548263073
# solver informed NN: relative error std = 0.04346156492829323

# AdamW (weight decay = 1e-3)
# Early stopping at epoch: 32683
# train loss: 0.0001775790196381208
# valid loss: 0.00025705013023171987
# solver informed NN: relative error mean = 0.045552149415016174
# solver informed NN: relative error std = 0.04208827763795853

# AdamW (weight decay = 1e-4)
# Early stopping at epoch: 30136
# train loss: 0.00018358921250396902
# valid loss: 0.0002895140124665987
# solver informed NN: relative error mean = 0.05093346908688545
# solver informed NN: relative error std = 0.05104278773069382

# AdamW (weight decay = 1e-5)
# Early stopping at epoch: 28797
# train loss: 0.0001869653955081423
# valid loss: 0.0002731673804292869
# solver informed NN: relative error mean = 0.048281583935022354
# solver informed NN: relative error std = 0.04760617017745972

# lmbda = 0.2, AdamW (weight decay = 1e-4)
# Early stopping at epoch: 27222
# train loss: 0.0001617536143064225
# valid loss: 0.00024595300945091166
# solver informed NN: relative error mean = 0.04497414082288742
# solver informed NN: relative error std = 0.04319871962070465


###########################
# snr = 40
# pure sparse recovery: relative error mean = 0.13001135667497607
# pure sparse recovery: relative error std = 0.0531750412098264

# Adam
# weight_decay = 0
# Early stopping at epoch: 27961
# train loss: 0.00014760199872612516
# valid loss: 0.00020513014977849607
# solver informed NN: relative error mean = 0.04235637187957764
# solver informed NN: relative error std = 0.03919161856174469

# AdamW (weight decay = 1e-3)
# Early stopping at epoch: 39050
# train loss: 0.00013143486049717895
# valid loss: 0.00019927580271744065
# solver informed NN: relative error mean = 0.04171454533934593
# solver informed NN: relative error std = 0.03708126023411751

# AdamW (weight decay = 1e-4)
# Early stopping at epoch: 46053
# train loss: 0.00013376694508170818
# valid loss: 0.0002032679767580703
# solver informed NN: relative error mean = 0.04309882968664169
# solver informed NN: relative error std = 0.040660563856363297

# AdamW (weight decay = 1e-5)
# Early stopping at epoch: 36618
# train loss: 0.0001317043679591734
# valid loss: 0.00018608447701101087
# solver informed NN: relative error mean = 0.04106350243091583
# solver informed NN: relative error std = 0.03966367989778519

# lmbda = 0.05, AdamW (weight decay = 1e-4)
# Early stopping at epoch: 32047
# train loss: 0.00014017744572757853
# valid loss: 0.00021993456498926712
# solver informed NN: relative error mean = 0.04358016327023506
# solver informed NN: relative error std = 0.04174710810184479

# lmbda = 0.02, AdamW (weight decay = 1e-4)
# Early stopping at epoch: 23794
# train loss: 0.0001272001104285557
# valid loss: 0.0001765735537952019
# solver informed NN: relative error mean = 0.03942295163869858
# solver informed NN: relative error std = 0.0387846864759922
