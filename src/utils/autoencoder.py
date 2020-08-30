import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

from src.net2net.net2deeper import net_2_deeper_net
from src.net2net.net2wider import net_2_wider_net

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

    def info(self, show_weights=False):
        print(f"Number of layers: {len(self.layers)}")
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                print(f"Layer {i + 1} \t Shape =({layer.in_features},{layer.out_features})")
                if show_weights:
                    print(f"\tWeight= {layer.weight.data} \n\tBias= {layer.bias.data}")

    def get_size(self):
        return len(self.layers)

    def get_layer_dims(self):
        dims = []
        for i in range(len(self.layers)):
            dims.append(self.layers[i].in_features)
            if i == len(self.layers) - 1:
                dims.append(self.layers[i].out_features)
        return dims

    # def insert_first_layer(self, layer_dim):
    #     out_feature = self.layers[0].in_features
    #     layer = nn.Linear(in_features=layer_dim, out_features=out_feature).apply(weights_init)
    #     self.layers.insert(0, layer)
    #
    # def insert_last_layer(self, layer_dim):
    #     in_feature = self.layers[-1].out_features
    #     layer = nn.Linear(in_features=in_feature, out_features=layer_dim).apply(weights_init)
    #     self.layers.append(layer)

    # def _expand_first(self, layer_dim):
    #     layer: nn.Linear = self.layers[0]
    #     weight = layer.weight.detach().numpy()
    #     add_unit_size = layer_dim - layer.in_features
    #
    #     add_layer = nn.Linear(in_features=add_unit_size, out_features=layer.out_features).apply(weights_init)
    #     add_weight = add_layer.weight.detach().numpy()
    #
    #     new_layer = nn.Linear(in_features=layer_dim, out_features=layer.out_features).apply(weights_init)
    #     new_layer.weight = torch.nn.Parameter(torch.Tensor(np.hstack((weight, add_weight))))
    #     self.layers[0] = new_layer
    #
    # def _expand_last(self, layer_dim):
    #     layer: nn.Linear = self.layers[-1]
    #     weight, bias = layer.weight.detach().numpy(), layer.bias.detach().numpy()
    #     add_unit_size = layer_dim - layer.out_features
    #
    #     add_layer = nn.Linear(in_features=layer.in_features, out_features=add_unit_size).apply(weights_init)
    #     add_weight = add_layer.weight.detach().numpy()
    #     add_bias = add_layer.bias.detach().numpy()
    #
    #     new_layer = nn.Linear(in_features=layer.in_features, out_features=layer_dim).apply(weights_init)
    #     weight_new_layer = np.hstack((weight.T, add_weight.T)).T
    #     bias_new_layer = np.hstack((bias, add_bias))
    #
    #     new_layer.weight = torch.nn.Parameter(torch.tensor(weight_new_layer, dtype=torch.float))
    #     new_layer.bias = torch.nn.Parameter(torch.tensor(bias_new_layer, dtype=torch.float))
    #     self.layers[-1] = new_layer
    #
    # def _deeper(self, pos_layer):
    #     layer: nn.Linear = self.layers[pos_layer]
    #     # print(layer.weight)
    #     weight_new_layer, bias_new_layer = net_2_deeper_net(layer.bias.detach().numpy(), noise_std=0.0001)
    #     new_layer = nn.Linear(in_features=layer.out_features, out_features=layer.out_features)
    #     new_layer.weight = torch.nn.Parameter(torch.tensor(weight_new_layer, dtype=torch.float))
    #     new_layer.bias = torch.nn.Parameter(torch.tensor(bias_new_layer, dtype=torch.float))
    #     self.layers.insert(pos_layer + 1, new_layer)
    #     # print(new_layer.weight)
    #
    # def _wider(self, pos_layer, new_layer_size=None):
    #     layer: nn.Linear = self.layers[pos_layer]
    #     next_layer: nn.Linear = self.layers[pos_layer + 1]
    #
    #     if new_layer_size is None:
    #         new_layer_size = layer.out_features + 1
    #     elif new_layer_size <= layer.out_features:
    #         raise ValueError(f"new_layer_size ={new_layer_size} is invalid.")
    #
    #     weight, bias = layer.weight.detach().numpy(), layer.bias.detach().numpy()
    #     weight_next_layer = next_layer.weight.detach().numpy()
    #
    #     weight = np.transpose(weight)
    #     weight_next_layer = np.transpose(weight_next_layer)
    #
    #     new_weight, new_bias, new_weight_next_layer = net_2_wider_net(weight, bias,
    #                                                                   weight_next_layer,
    #                                                                   new_layer_size=new_layer_size,
    #                                                                   noise_std=0.0001,
    #                                                                   split_max_weight_else_random=True)
    #
    #     new_weight = np.transpose(new_weight)
    #     new_weight_next_layer = np.transpose(new_weight_next_layer)
    #
    #     new_layer = nn.Linear(in_features=layer.in_features, out_features=new_layer_size).apply(weights_init)
    #     new_layer.weight = torch.nn.Parameter(torch.tensor(new_weight, dtype=torch.float))
    #     new_layer.bias = torch.nn.Parameter(torch.tensor(new_bias, dtype=torch.float))
    #
    #     new_next_layer = nn.Linear(in_features=new_layer_size, out_features=next_layer.out_features).apply(weights_init)
    #     new_next_layer.weight = torch.nn.Parameter(torch.tensor(new_weight_next_layer, dtype=torch.float))
    #     new_next_layer.bias = next_layer.bias
    #
    #     self.layers[pos_layer] = new_layer
    #     self.layers[pos_layer + 1] = new_next_layer

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
        # TODO: expand layer by adding new node instead of add new layer
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

    # def deeper(self, pos_layer):
    #     size_part = self.encoder.get_size()
    #     if pos_layer == size_part - 1 or size_part - pos_layer - 2 < 0:
    #         raise ValueError(f"pos_layer={pos_layer} is invalid.")
    #
    #     self.encoder._deeper(pos_layer)
    #     self.decoder._deeper(size_part - pos_layer - 2)
    #     self.hidden_dims = self.get_hidden_dims()
    #
    # def wider(self, pos_layer, new_layer_size=None):
    #     size_part = self.encoder.get_size()
    #     if pos_layer == size_part - 1 or size_part - pos_layer - 2 < 0:
    #         raise ValueError(f"pos_layer={pos_layer} is invalid.")
    #
    #     self.encoder._wider(pos_layer, new_layer_size)
    #     self.decoder._wider(size_part - pos_layer - 2, new_layer_size)
    #     self.hidden_dims = self.get_hidden_dims()


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0.01)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(6)

    num_epochs = 1000
    dataset = torch.randn(1, 6).uniform_(0, 1)
    dataset[dataset > 0.5] = 1.
    dataset[dataset <= 0.5] = 0.

    dataset_2 = torch.randn(1, 7).uniform_(0, 1)
    dataset_2[dataset_2 > 0.5] = 1.
    dataset_2[dataset_2 <= 0.5] = 0.

    ae = Autoencoder(input_dim=6, embedding_dim=2, hidden_dims=[5],
                     activation=ModelActivation(hidden_layer_act='sigmoid', embedding_act='sigmoid', )).to(device)
    print(ae(dataset.to(device)))
    # ae.expand_first_layer(7)
    # ae.deeper(0)
    print(ae)

    # print(ae(dataset_2))

    # ae.wider(pos_layer=0, new_layer_size=7)
    # print(ae)
    # print(ae(dataset))

    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-5)

    # mean-squared error loss
    criterion = nn.MSELoss()

    dataloader = DataLoader(dataset=dataset, batch_size=dataset.shape[0])

    for epoch in range(num_epochs):
        for data in dataloader:
            inp = data
            # print("inp: ", inp)
            inp = Variable(inp).to(device)
            # ===================forward=====================
            output, embed_out = ae(inp)
            loss = criterion(output, inp)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        if epoch % 100 == 0:
            print('Epoch [{}/{}] \t\tloss:{:.4f}'.format(epoch + 1, num_epochs, loss))

    print(dataset)
    print(ae(dataset.to(device)))
