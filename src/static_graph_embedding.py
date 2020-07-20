from abc import ABCMeta

class StaticGraphEmbedding:
    __metaclass__ = ABCMeta

    def __init__(self, d):
        '''
        Initialize the embedding class
        :param d: dimension of embedding
        '''
        pass

    def get_method_name(self):
        '''
        Name of method
        :return: name
        '''
        return ""

    def get_method_summary(self):
        '''
        Summary for the embedding include method name and parameter setting
        :return: a summary string of  the method
        '''
        return ""

    def learn_embedding(self, graph):
        '''

        :param graph:
        :return:
        '''
        pass

    def get_embedding(self):
        '''

        :return:
        '''
        pass

    def get_edge_weight(self,i,j):
        '''
        Compute the weight for edge between node i and node j
        :param i:
        :param j:
        :return:
        '''
        pass

    def get_reconstructed_adj(self):
        '''
        Compute the adjacence matrix from the learned embedding
        :return:
        '''
        pass