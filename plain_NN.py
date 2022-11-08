import numpy as np
import torch
from solver_util import gen_sensing_matrix, gen_spectrum
from solver_util import rel_reconst_error, awgn
from solver_util import train_model_L1, train_model_L2
from weight_init import init_xavier_uniform
from Nets import Net_NonLinReg
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


# load data
R = gen_sensing_matrix()
print (f'RTR cond num is {np.linalg.norm(R.T.dot(R))}')
vif = [variance_inflation_factor(R, i) for i in range(R.shape[1])]
if np.min(vif) < 10:
    print ('there exists no multicollinnearity')
elif np.min(vif) < 100:
    print ('there exists strong multicollinnearity')
else:
    print ('there exists severe multicollinnearity')
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
    model = torch.load(f'plain_NN_Adam_dB{snr}.pkl')
except:
    net = Net_NonLinReg(input_dim=108, n_hidden1=32, n_hidden2=32, out_dim=71)
    net.apply(init_xavier_uniform)
    model, avg_train_losses, avg_valid_losses = train_model_L2(train_data_x=train_data_I, train_data_y=train_data_x,
                                                                model=net,
                                                                lbd=0,
                                                                lr=2e-4,
                                                                batch_size=32,
                                                                patience=1000,
                                                                n_epochs=50000,
                                                                weight_decay=0)
    # save model
    torch.save(model, f'plain_NN_Adam_dB{snr}.pkl')

# prediction
x_recon = model(test_data_I)
x_recon = x_recon.detach().numpy().T

# reconstruct spectrum
xtrue = test_data_x.detach().numpy().T
rel_err = rel_reconst_error(x_recon, xtrue)
err_mean = np.mean(rel_err)
err_std = np.std(rel_err)
print (f'plain NN: relative error mean = {err_mean}')
err_std = np.std(rel_err)
print (f'plain NN: relative error std = {err_std}')


# Adam
# snr = 30
# Early stopping at epoch: 20860
# train loss: 0.0002931512923062067
# valid loss: 0.0003961593417140345
# plain NN: relative error mean = 0.05770346149802208
# plain NN: relative error std = 0.053607027977705

# snr = 35
# Early stopping at epoch: 24596
# train loss: 0.0003138125642127467
# valid loss: 0.00039383068926528923
# plain NN: relative error mean = 0.05415887013077736
# plain NN: relative error std = 0.04725334048271179

# snr = 40
# Early stopping at epoch: 32060
# train loss: 0.00025330346355022975
# valid loss: 0.0003227989105249031
# plain NN: relative error mean = 0.04965626448392868
# plain NN: relative error std = 0.044538337737321854
