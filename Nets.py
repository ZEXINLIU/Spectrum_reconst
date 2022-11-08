import numpy as np
import torch
import torch.nn as nn


class AE1(nn.Module):
    # one layer
    def __init__(self, input_dim, emb_size, dropout_rate):
        super(AE1, self).__init__()

        self.in_dim = input_dim
		
	## Encoder
        self.fc_encoder = nn.Sequential(
            nn.Linear(self.in_dim, emb_size),
            nn.ReLU(),
            # nn.Sigmoid(),
            nn.Dropout(p=dropout_rate),
        )

	## Decoder
        self.fc_decoder = nn.Sequential(
            nn.Linear(emb_size, input_dim),
            # nn.ReLU(),
            # nn.Sigmoid(),
            # nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        emb = self.fc_encoder(x)
        recon_x = self.fc_decoder(emb)
        return emb, recon_x


class AE2(nn.Module):
    # two layers
    def __init__(self, input_dim, hidden1, emb_size, dropout_rate):
        super(AE2, self).__init__()

        self.in_dim = input_dim
		
	## Encoder
        self.fc_encoder = nn.Sequential(
            nn.Linear(self.in_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden1, emb_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

	## Decoder
        self.fc_decoder = nn.Sequential(
            nn.Linear(emb_size, hidden1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(hidden1, self.in_dim),
            # nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        emb = self.fc_encoder(x)
        recon_x = self.fc_decoder(emb)
        return emb, recon_x


class AE4(nn.Module):
    # four layers
    def __init__(self, input_dim, hidden1, hidden2, hidden3, emb_size, dropout_rate):
        super(AE, self).__init__()

        self.in_dim = input_dim
		
	## Encoder
        self.fc_encoder = nn.Sequential(
            nn.Linear(self.in_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden3, emb_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

	## Decoder
        self.fc_decoder = nn.Sequential(
            nn.Linear(emb_size, hidden3),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(hidden3, hidden2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden1, self.in_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        emb = self.fc_encoder(x)
        recon_x = self.fc_decoder(emb)
        return emb, recon_x


class Net_NonLinReg(torch.nn.Module):
    def __init__(self, input_dim=1, n_hidden1=1, n_hidden2=1, out_dim=1, dropout_rate=0):
        super(Net_NonLinReg, self).__init__()
        self.nonlinears = nn.Sequential(
                nn.Linear(input_dim, n_hidden1),
                # nn.ReLU(inplace=True),
                nn.Sigmoid(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(n_hidden1, n_hidden2),
                # nn.ReLU(inplace=True),
                nn.Sigmoid(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(n_hidden2, out_dim),
                # nn.ReLU(inplace=True),
                nn.Sigmoid(),
                nn.Dropout(p=dropout_rate)
                )

    def forward(self, x):
        return self.nonlinears(x)

class Net_LinReg(torch.nn.Module):
    def __init__(self, input_dim=1, out_dim=1):
        super(Net_LinReg, self).__init__()
        self.linears = nn.Sequential(
                nn.Linear(input_dim, out_dim)
                )

    def forward(self, x):
        return self.linears(x)



if __name__ == "__main__":
    ## 利用GPU加速
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('===== Using device: ' + device)
    
    from solver_util import gen_data, train_model
    A, x0, y, ydelta = gen_data(N=108, S=71, m=1)
    train_data_x = torch.tensor(ydelta.T).to(torch.float32)
    
    AE = AE1(input_dim=108, emb_size=64, dropout_rate=0).to(device)
    model, train_loss, valid_loss = train_model(train_data_x=train_data_x,
                                                model=AE,
                                                lr=1e-4,
                                                batch_size=10,
                                                patience=20,
                                                n_epochs=1000)
    emb1, recon_y1 = model(train_data_x)

    # AE = AE1(input_dim=64, emb_size=32, dropout_rate=0).to(device)
    # model, train_loss, valid_loss = train_model(train_data_x=emb1.detach(),
                                                # model=AE,
                                                # lr=1e-4,
                                                # batch_size=10,
                                                # patience=20,
                                                # n_epochs=500)
    # emb2, recon_y2 = model(emb1)
#
    # AE = AE2(input_dim=108, hidden1=64, emb_size=32, dropout_rate=0).to(device)
    # model, train_loss, valid_loss = train_model(train_data_x=train_data_x,
                                                # model=AE,
                                                # lr=1e-4,
                                                # batch_size=10,
                                                # patience=20,
                                                # n_epochs=500)
    # emb3, recon_y3 = model(train_data_x)
