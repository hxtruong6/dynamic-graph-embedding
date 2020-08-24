from sklearn.linear_model import LogisticRegression
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from munkres import Munkres
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn import metrics

from src.data_preprocessing.graph_preprocessing import read_node_label
from src.utils.classify import Classifier


def classify_embeddings_evaluate(embeddings, label_file=None, test_percent=0.25, seed=0):
    if label_file is None:
        raise ValueError("Must provide label_file name.")
    X, Y = read_node_label(filename=label_file, skip_head=True)
    print("Training classifier using {:.2f}% nodes...".format((1 - test_percent) * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_percent, random_state=seed, shuffle=True)
    clf.train(X_train, y_train, Y)
    return clf.evaluate(X_test, y_test)


# https://github.com/bdy9527/SDCN
def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc, f1_macro


def cluster_evaluate(y_true, y_pred, alg=0):
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    print(alg, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
          ', f1 {:.4f}'.format(f1))


def reconstruction_mse(reconstruction, graph: nx.Graph):
    X = np.array(nx.adjacency_matrix(graph).todense())
    X_hat = reconstruction
    err = np.sum(np.sqrt(np.square(X_hat - X)))
    return err


def dy_reconstruction_accuracy(reconstructions, graphs):
    accs = []
    for i in range(len(graphs)):
        acc = reconstruction_accuracy(reconstruction=reconstructions[i], graph=graphs[i])
        accs.append(acc)
    print("==>\tMean accuracy: ", np.mean(accs))


def reconstruction_accuracy(reconstruction, graph: nx.Graph):
    print("\nReconstruction accuracy:")
    max_acc = 0
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for threshold in thresholds[:1]:
        g = np.zeros_like(reconstruction)
        g[reconstruction >= threshold] = 1.0
        pred_count = 0
        for u, v in graph.edges():
            if g[u][v] == 1.0:
                pred_count += 1
        print(f"With threshold {threshold}: {pred_count / graph.number_of_edges()}")
        max_acc = max(max_acc, pred_count / graph.number_of_edges())
    return max_acc


if __name__ == "__main__":
    G = nx.karate_club_graph().to_undirected()
