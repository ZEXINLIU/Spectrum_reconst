import numpy as np
import torch
import torch.nn as nn
from scipy.fftpack import dct, idct
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import r2_score
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, Dataset, DataLoader
from early_stopping import EarlyStopping
from Nets import AE1, AE2, Net_LinReg
from csv import reader
from scipy.io import loadmat
import random
import pdb

def operator(m, mu):
    helpvar = (np.arange(1, m+100+1).astype(float)**(-mu)).reshape(-1,1)
    perturb = np.exp(0.5 * np.random.randn(m+100,1) - 0.5**2/2)
    helpop = np.sort(perturb*helpvar, axis=0)[::-1]
    A = helpop[:m]
    return np.diag(A.reshape(A.size))

def solution(m, nu):
    helpvar = (np.arange(1,m+1).astype(float)**(-nu)).reshape(-1,1)
    sign = 2*np.ceil(2*np.random.rand(m,1))-3
    perturb = 1 + 0.1*np.random.randn(m,1)
    x = sign*perturb*helpvar
    return x

def dct_test(x): # consistent with MATLAB
    N = len(x)
    y = np.zeros(x.shape)
    for i in range(N):
        for n in range(N):
            y[i] += x[n] * np.cos(np.pi*i*(n+0.5)/N) * np.sqrt(2/N)
    y[0] /= np.sqrt(2)
    return y

def idct_test(y): # consistent with MATLAB
    N = len(y)
    x = np.zeros(y.shape)
    for i in range(N):
        for n in range(N):
            if n == 0:
                x[i] += y[n] * np.cos(np.pi*n*(i+0.5)/N) * np.sqrt(1/N)
            else:
                x[i] += y[n] * np.cos(np.pi*n*(i+0.5)/N) * np.sqrt(2/N)
    return x

def noise(y, logN2S):
    m = len(y)
    N2S = 10**logN2S
    delta = N2S * np.linalg.norm(y) * np.sqrt(m)

    pts = int(np.ceil(1.5*m))
    noisetime = np.random.randn(pts,1)
    std = np.zeros((21,1))
    std[10] = 1
    noisetime = noisetime / np.linalg.norm(std)

    ytime = dct_test(np.append(y, np.zeros([pts-m,1]), axis=0))
    ypertime = ytime + delta * noisetime
    yhelp = idct_test(ypertime)
    ydelta = yhelp[:m]
    return ydelta, delta

def alpha_n(alpha_0=18.42, q=0.5429, n=np.arange(1,51)):
    return alpha_0 * q**n

def gaussian_kernel_inefficient(x, y, sigma=1):
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    d = x - y
    return np.exp(-np.dot(d.T, d) / (2*sigma**2))

def gaussian_kernel(X, Y, sigma=1):
    """
    Inputs
    X, 2d numpy array, n x dim
    """
    d2 = np.sum(X*X, axis=1, keepdims=True) + np.sum(Y*Y, axis=1, keepdims=True).T - 2 * np.dot(X, Y.T)
    return np.exp(-d2 / (2*sigma**2))

def sparse_basis(FWHM, wavelength):
    X = wavelength.reshape(-1,1)
    for i, wi in enumerate(FWHM):
        Phi = gaussian_kernel(X=X, Y=X, sigma=wi/(2*np.sqrt(2*np.log(2))))
        if i == 0:
            res = Phi
        else:
            res = np.concatenate((res, Phi), axis=1)
    return res

def rel_reconst_error(xhat, x):
    """
    Inputs
    xhat: reconstructed spectrum
    x: ground-truth
    """
    # convert tensor to numpy
    if type(xhat) == torch.Tensor:
        xhat = xhat.detach().numpy()
    if type(x) == torch.Tensor:
        x = x.detach().numpy()

    if len(xhat.shape) == 1:
        rel_error = np.linalg.norm(xhat-x, 2) / np.linalg.norm(x, 2)
        return rel_error
    
    rel_error = np.linalg.norm(xhat-x, ord=2, axis=0) / np.linalg.norm(x, ord=2, axis=0)
    return rel_error

def gen_data_diag(mu, m, nu, logN2S):
    A = operator(m, mu)
    x0 = solution(m ,nu)
    y = A.dot(x0)
    ydelta, delta = noise(y, logN2S)
    return A, x0, y, ydelta, delta

def gen_data(N, S, m=1):
    A = np.random.randn(N, S)*5
    x0 = np.random.rand(S, m)
    noise = np.random.normal(0, 0.1, (N, m))
    ydelta = A.dot(x0) + noise
    y = A.dot(x0)
    return A, x0, y, ydelta

def awgn(x, snr, seed=0):
    # np.random.seed(seed)
    snr = 10**(snr/10.0)
    n, m = x.shape
    xpower = np.sum(x**2, axis=0)
    npower = xpower / (n * snr)
    noise = np.random.randn(n).reshape(-1,1) * np.sqrt(npower)
    return x + noise

def gen_sensing_matrix():
    with open('Transmission.CSV', 'r', encoding='utf-8') as f:
        data = list(reader(f))
    R = np.array(data).astype('float64') # shape (77, 209), wavelength 380:5:760
    R = R[4:-2,1:109] # shape (71, 108), wavelength 400:750
    return R.T

def gen_spectrum():
    data = loadmat('munsell380_780_1_glossy.mat')
    spectrum = data['X'] # shape (401, 1600)
    spect = spectrum[20:-30, :][::5].astype('float64') # shape (71, 1600)
    return spect

def gen_measurements(snr, seed=108):
    R = gen_sensing_matrix()
    x = gen_spectrum()
    I_ideal = R.dot(x)
    I_corrupted = awgn(x=I_ideal, snr=snr, seed=seed)
    return R, x, I_corrupted

class MyDataSet(Dataset):  # 定义数据格式
    def __init__(self, train_x, train_y, sample):
        self.train_x = train_x
        self.train_y = train_y
        self._len = sample

    def __getitem__(self, item: int):
        return self.train_x[item], self.train_y[item]

    def __len__(self):
        return self._len

def create_dataset(train_data_x, train_data_y, batch_size):
    valid_size = 0.2
    num_train = len(train_data_x)
    indices = list(range(num_train))
    # np.random.shuffle(indices)
    random.Random(108).shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = DataLoader(MyDataSet(train_data_x, train_data_y, len(train_data_x)), sampler=train_sampler, batch_size=batch_size)
    valid_loader = DataLoader(MyDataSet(train_data_x, train_data_y, len(train_data_x)), sampler=valid_sampler, batch_size=batch_size)
    return train_loader, valid_loader

def create_dataset_concatenate(train_data, test_data, batch_size):
    # percentage of training set to use as validation
    valid_size = 0.2
    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    # loading training data in batches
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    valid_loader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=0)
    return train_loader, valid_loader, test_loader

def train_model_L2(train_data_x, train_data_y, model, lbd, lr, batch_size, patience, n_epochs, weight_decay):
    train_loader, valid_loader = create_dataset(train_data_x=train_data_x, train_data_y=train_data_y, batch_size=batch_size)
    # Per-parameter options
    # weight_p, bias_p = [], []
    # for name, p in model.named_parameters():
        # if 'bias' in name:
            # bias_p += [p]
        # else:
            # weight_p += [p]
    # optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay': weight_decay},
                                # {'params': bias_p, 'weight_decay': weight_decay}],
                                # lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience, verbose=True) # valid loss not decrease in patience epochs -> stop
    
    avg_train_losses = []
    avg_valid_losses = []
    for epoch in range(n_epochs):
        train_losses = []
        valid_losses = []
        # %%% train the model %%%
        model.train() # open BN and Dropout, prep model for training
        for train_x, train_y in train_loader:
            optimizer.zero_grad()
            y_pred = model(train_x)
            loss = criterion(y_pred, train_y)

            reglar_loss = 0
            # set decay or l2 regularization only for weight
            for name, param in model.named_parameters():
                if 'weight' in name:# print (param.shape, torch.sum(torch.abs(param)))
                    reglar_loss += torch.norm(param, p=2)
            # set decay or l2 regularization for weight and bias
            # for name, param in model.named_parameters():
                # reglar_loss += torch.norm(param, p=2)
            loss += lbd * reglar_loss
            
            loss.backward()
            # loss.backward(retain_graph=True)
            optimizer.step()
            train_losses.append(loss.item())
            
        # %%% validate the model %%%
        model.eval() # close BN and Dropout, prep model for evaluation
        for valid_x, valid_y in valid_loader:
            y_pred = model(valid_x)
            loss = criterion(y_pred, valid_y)
            valid_losses.append(loss.item())
        
        # print training/validation statistics
        # calculate average loss over an epoch
        avg_train_loss = np.average(train_losses)
        avg_valid_loss = np.average(valid_losses)
        avg_train_losses.append(avg_train_loss)
        avg_valid_losses.append(avg_valid_loss)

        epoch_len = len(str(n_epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                         f'train_loss: {avg_train_loss:.5f} ' +
                         f'valid_loss: {avg_valid_loss:.5f}')
        # print(print_msg)
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(avg_valid_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch: {epoch+1}")
            break
    
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))
    print (f'train loss: {avg_train_loss}')
    print (f'valid loss: {avg_valid_loss}')
    return model, avg_train_losses, avg_valid_losses

def train_model_L1(train_data_x, train_data_y, model, lbd, lr, batch_size, patience, n_epochs):
    train_loader, valid_loader = create_dataset(train_data_x=train_data_x, train_data_y=train_data_y, batch_size=batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience, verbose=False) # valid loss not decrease in 5 epochs -> stop
    
    avg_train_losses = []
    avg_valid_losses = []
    for epoch in range(n_epochs):
        train_losses = []
        valid_losses = []
        # %%% train the model %%%
        model.train() # open BN and Dropout, prep model for training
        for train_x, train_y in train_loader:
            optimizer.zero_grad()
            y_pred = model(train_x)
            loss = criterion(y_pred, train_y)
            
            reglar_loss = 0
            # set decay or l1 regularization only for weight
            for name, param in model.named_parameters():
                if 'weight' in name:# print (param.shape, torch.sum(torch.abs(param)))
                    reglar_loss += torch.sum(torch.abs(param)) # or torch.norm(param, p=1)
            # set decay or l1 regularization for weight and bias
            # for name, param in model.named_parameters():
                # reglar_loss += torch.sum(torch.abs(param))
            loss += lbd * reglar_loss

            loss.backward()
            # loss.backward(retain_graph=True)
            optimizer.step()
            train_losses.append(loss.item())
            
        # %%% validate the model %%%
        model.eval() # close BN and Dropout, prep model for evaluation
        for valid_x, valid_y in valid_loader:
            y_pred = model(valid_x)
            loss = criterion(y_pred, valid_y)
            valid_losses.append(loss.item())
        
        # print training/validation statistics
        # calculate average loss over an epoch
        avg_train_loss = np.average(train_losses)
        avg_valid_loss = np.average(valid_losses)
        avg_train_losses.append(avg_train_loss)
        avg_valid_losses.append(avg_valid_loss)

        epoch_len = len(str(n_epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                         f'train_loss: {avg_train_loss:.5f} ' +
                         f'valid_loss: {avg_valid_loss:.5f}')
        # print(print_msg)
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(avg_valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    print (f'train loss: {avg_train_loss}')
    print (f'valid loss: {avg_valid_loss}')
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))
    return model, avg_train_losses, avg_valid_losses
 
def train_model_Encoder(train_data_x, train_data_y, model, lr, batch_size, patience, n_epochs):
    train_loader, valid_loader = create_dataset(train_data_x=train_data_x, train_data_y=train_data_y, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(n_epochs//9)+1, verbose=True)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience, verbose=True) # valid loss not decrease in 5 epochs -> stop
    
    avg_train_losses = []
    avg_valid_losses = []
    for epoch in range(n_epochs):
        train_losses = []
        valid_losses = []
        # %%% train the model %%%
        model.train() # open BN and Dropout, prep model for training
        for train_x, train_y in train_loader:
            optimizer.zero_grad()
            emb_train, recon_y_train = model(train_x)
            loss = criterion(recon_y_train, train_y)
            loss.backward()
            # loss.backward(retain_graph=True)
            optimizer.step()
            train_losses.append(loss.item())
            
        # %%% validate the model %%%
        model.eval() # close BN and Dropout, prep model for evaluation
        for valid_x, valid_y in valid_loader:
            emb_valid, recon_y_valid = model(valid_x)
            loss = criterion(recon_y_valid, valid_y)
            valid_losses.append(loss.item())
        
        # print training/validation statistics
        # calculate average loss over an epoch
        avg_train_loss = np.average(train_losses)
        avg_valid_loss = np.average(valid_losses)
        avg_train_losses.append(avg_train_loss)
        avg_valid_losses.append(avg_valid_loss)

        epoch_len = len(str(n_epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                         f'train_loss: {avg_train_loss:.5f} ' +
                         f'valid_loss: {avg_valid_loss:.5f}')
        # print(print_msg)
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(avg_valid_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch: {epoch+1}")
            break

        # adjust learning rate
        # scheduler.step(avg_valid_loss)
        # scheduler.step(epoch)
    print (f'train loss: {avg_train_loss}')
    print (f'valid loss: {avg_valid_loss}')
    # plt.figure()
    # plt.plot(np.arange(1,len(avg_train_losses)+1), avg_train_losses, label='train')
    # plt.plot(np.arange(1,len(avg_valid_losses)+1), avg_valid_losses, label='valid')
    # minposs = avg_valid_losses.index(min(avg_valid_losses))+1 # find position of lowest validation loss
    # plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
    # plt.xlabel('epochs')
    # plt.ylabel('loss')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))
    return model, avg_train_losses, avg_valid_losses


def solver_normal(x, y):
    m, n = x.shape
    X = np.concatenate((np.ones((m,1)),x),axis=1)
    Y = y.reshape(-1,1)
    W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    return W

def solver_ridge(A, ydelta):
    lr = RidgeCV(alphas=np.logspace(-4,-1,1000), fit_intercept=True, normalize=True, cv=10)
    lr.fit(A, ydelta.ravel())
    coef = lr.coef_
    xhat = coef
    return xhat, lr.alpha_

def solver_ridge_mode(x, y, lbd, L=None, mode='normal', lr=1e-4, epochs=1e4):
    m, n = x.shape
    X = np.concatenate((np.ones((m,1)),x),axis=1)
    Y = y.reshape(-1,1)
    if mode == 'normal': # L is identity
        if type(L) != np.ndarray:
            # if J = loss / m + reg
            # W = np.linalg.inv(X.T.dot(X) + m * lbd * np.eye(X.shape[1])).dot(X.T).dot(Y)
            # u, s, vt = np.linalg.svd(X)#, full_matrices=1)
            # W = vt[:len(s), :].T@np.diag(s/(s**2+m*lbd))@(u[:,:len(s)].T@Y)

            # if J = loss + reg
            # W = np.linalg.inv(X.T.dot(X) + lbd * np.eye(X.shape[1])).dot(X.T).dot(Y)
            u, s, vt = np.linalg.svd(X)#, full_matrices=1)
            W = vt[:len(s), :].T@np.diag(s/(s**2+lbd))@(u[:,:len(s)].T@Y)
            
            # for loop, equal to matrix computation above
            # W = 0
            # for i in range(len(s)):
                # W += vt[i, :].T.reshape(-1,1) * (s[i]/(s[i]**2+lbd)) * (u[:,i]@Y)
        
        else:# when L is not an identity matrix, can't use SVD to decompose
            # if J = loss / m + reg
            # M = X.T.dot(X) + m * lbd * (L.T.dot(L))
            # W = np.linalg.inv(M).dot(X.T.dot(Y))

            # if J = loss + reg
            M = X.T.dot(X) + lbd * (L.T.dot(L))
            W = np.linalg.inv(M).dot(X.T.dot(Y))
    elif mode == 'GD':
        if type(L) != np.ndarray:
            W = np.ones([n+1, 1])
            for i in range(int(epochs)):
                # if J = loss / m + reg
                # grad = (X.T.dot(X).dot(W) - X.T.dot(Y)) / m + lbd*W
                # if J = loss + reg
                grad = (X.T.dot(X).dot(W) - X.T.dot(Y)) / m + lbd*W / m
                W -= lr * grad
        else:
            W = np.ones([n+1, 1])
            for i in range(int(epochs)):
                # if J = loss / m + reg
                # grad = (X.T.dot(X).dot(W) - X.T.dot(Y)) / m + lbd*(L.T.dot(L)).dot(W)
                # if J = loss / m + reg
                grad = (X.T.dot(X).dot(W) - X.T.dot(Y)) / m + lbd*(L.T.dot(L)).dot(W) / m
                W -= lr * grad
    return W.ravel()

def mse(y, ypred):
    return np.sum((y - ypred)**2) / len(y)

def rmse(y, ypred):
    return np.sqrt(mse(y, ypred))

def solver_ridge_modeCV(x, y, alphas, L=None, mode='normal', lr=1e-3, epochs=1e4):
    res = float(-np.inf)
    for i, alpha in enumerate(alphas):
        W = solver_ridge_mode(x, y, alpha, L, mode, lr, epochs)
        ypred = x.dot(W[1:]) + W[0]
        score = r2_score(y, ypred)
        if score > res:
            res = score
            bestW = W
            bestalpha = alpha
    return bestW, bestalpha

def first_regularization(m):
    L = np.eye(m,m+1,0) + (-1) * np.eye(m,m+1,1)
    return L

def solver_lasso(A, ydelta):
    lr = LassoCV(alphas=np.logspace(-5,-1,2000), fit_intercept=True, normalize=True, positive=True, tol=1e-4, max_iter=100000, cv=10)
    # lr = LassoCV(eps=1e-5, n_alphas=500, fit_intercept=True, normalize=True, positive=True, tol=1e-4, max_iter=10000, cv=10)
    lr.fit(A, ydelta.ravel())
    coef = lr.coef_
    # intercept = lr.intercept_
    # mse_path = lr.mse_path_
    xhat = coef
    return xhat, lr.alpha_

def solver_lasso_sparse(A, ydelta):
    N, S = A.shape
    # Full width at half maximum (FWHM, nm)
    FWHM = [5, 10, 20, 40, 60, 80, 100]
    # wavelength position
    wavelength = np.linspace(400, 750, S)
    Phi = sparse_basis(FWHM, wavelength)
    lr = LassoCV(alphas=np.logspace(-3,-2,1000), fit_intercept=True, normalize=True, positive=True, tol=1e-4, max_iter=100000, cv=10)
    lr.fit(A.dot(Phi), ydelta.ravel())
    coef = lr.coef_
    xhat = Phi.dot(coef)
    return xhat, lr.alpha_

def solver_elastic(A, ydelta, l1_ratio=0.8):
    lr = ElasticNetCV(alphas=np.logspace(-5,-3,500), l1_ratio=l1_ratio, fit_intercept=True, normalize=True, positive=True, tol=1e-4, max_iter=100000, cv=10)
    lr.fit(A, ydelta.ravel())
    coef = lr.coef_
    xhat = coef
    return xhat, lr.alpha_


if __name__ == '__main__':
    data = np.random.randn(5,3)
