import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.setting_param import ModelActivation


class Autoencoder(nn.Module):
    def __init__(self, input_dim=None, embedding_dim=None, hidden_dims=None, l1=0.01, l2=1e-5,
                 activation=ModelActivation()):
        super(Autoencoder, self).__init__()
        if input_dim is None or embedding_dim is None:
            return
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.l1 = l1
        self.l2 = l2

        if activation is dict():
            self.activation = ModelActivation(hidden_layer_act=activation['hidden'],
                                              embedding_act=activation['embedding'], output_act=activation['output'])
        else:
            self.activation = activation

        # ======== Create layers
        layers = [nn.Module()] * (len(hidden_dims) * 2 + 2)
        for i in range(len(hidden_dims) + 1):
            if i == 0:
                layers[0] = nn.Linear(in_features=input_dim, out_features=hidden_dims[i])
                layers[-1] = nn.Linear(in_features=hidden_dims[i], out_features=input_dim)
            elif i == len(hidden_dims):
                layers[i] = nn.Linear(in_features=hidden_dims[-1], out_features=embedding_dim)
                layers[i + 1] = nn.Linear(in_features=embedding_dim, out_features=hidden_dims[-1])
            else:
                layers[i] = nn.Linear(in_features=hidden_dims[i - 1], out_features=hidden_dims[i])
                layers[-i - 1] = nn.Linear(in_features=hidden_dims[i], out_features=hidden_dims[i - 1])

        self.layers = nn.ModuleList(layers).apply(weights_init)

    def _encoder(self, x):
        for i in range(len(self.layers) // 2):
            x = self.layers[i](x)
            x = self.activation.hidden_layer_act(x)
        x = self.activation.embedding_act(x)
        return x

    def _decoder(self, x):
        for i in range(len(self.layers) // 2, len(self.layers)):
            x = self.layers[i](x)
            x = self.activation.hidden_layer_act(x)
        x = self.activation.output_act(x)
        return x

    def forward(self, x):
        y = self._encoder(x)
        x_hat = self._decoder(y)
        return x_hat, y

    def get_embedding(self, x):
        embedding = self._encoder(x)
        embedding = embedding.clone().detach().cpu()
        return embedding.detach().numpy()

    def get_reconstruction(self, x):
        reconstruction = self._decoder(self._encoder(x))
        reconstruction = reconstruction.clone().detach().cpu()
        return reconstruction.detach().numpy()

    def get_hidden_dims(self):
        '''
        Suppose encoder part and decoder part have symmetric size. So just return encoder part.
        Decoder part is reverse of encoder part
        :param half_model:
        :return:
        '''
        hidden_dims = []
        for i in range(1, len(self.layers) // 2):
            hidden_dims.append(self.layers[i].in_features)
        return hidden_dims

    def expand_first_layer(self, layer_dim):
        self.input_dim = layer_dim
        prev_layer_dim = self.layers[0].in_features
        self.layers.insert(0, nn.Linear(in_features=layer_dim, out_features=prev_layer_dim))
        self.layers.append(nn.Linear(in_features=prev_layer_dim, out_features=layer_dim))

    def get_input_dim(self):
        return self.input_dim

    def get_embedding_dim(self):
        return self.embedding_dim

    def get_config_layer(self):
        config_layer = {
            "input_dim": self.input_dim,
            "embedding_dim": self.embedding_dim,
            "hidden_dims": self.get_hidden_dims(),
            "l1": self.l1,
            "l2": self.l2,
            "activation": self.activation.config()
        }
        return config_layer


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0.01)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(6)

    num_epochs = 1000
    dataset = torch.randn(2, 20).uniform_(0, 1)
    dataset[dataset > 0.5] = 1.
    dataset[dataset <= 0.5] = 0.
    # print(dataset)
    # print(torch.transpose(dataset, 0, 1))
    # dataset = (dataset + torch.transpose(dataset, 0, 1)) / 2
    # print(dataset)

    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)

    #  create dataset

    ae = Autoencoder(input_dim=20, embedding_dim=4, hidden_dims=[16, 10, 8], activation=ModelActivation()).to(device)

    print(ae)
    print(ae.get_hidden_dims())
    ae.expand_first_layer(30)
    ae.expand_first_layer(40)
    print(ae)
    print(ae.get_hidden_dims())
    #
    # optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-5)
    #
    # # mean-squared error loss
    # criterion = nn.MSELoss()
    #
    # for epoch in range(num_epochs):
    #     for data in dataloader:
    #         inp = data
    #         # print("inp: ", inp)
    #         inp = Variable(inp).to(device)
    #         # ===================forward=====================
    #         output, embed_out = ae(inp)
    #         loss = criterion(output, inp)
    #         # ===================backward====================
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     # ===================log========================
    #     # print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss))
    #
    # # print(ae.get_hidden_dims())
    # # ae.info(show_weights=True)
    #
    # print("Original: ", dataset)
    # output, e = ae(torch.tensor(dataset).clone().detach().to(device))
    # print("Embdding: ", e.cpu().data)
    # print("Reconstruction: ", output.cpu().data)

    # print("\nExpand autoencoder")
    # ae.expand_first_layer(layer_dim=6)
    # ae.info(show_weights=True)
    # print(ae)
