import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from Data import train_loader, test_loader
from Loss import custom_loss
from Model import Encoder, Decoder
from Utils import VSA_fitting_report, get_mapping
from VSASpace import VSASpace, matrix_to_vector

# hyperparameters
VSA_dim = 10000
lr = 0.001
epoch_VSA = 10
epoch_alignment = 1

# descriptions:
# Default values: g_num_e=20，r_proportion=0.2，g_k_proportion=0.2，g_k_bias=0.5，l_num_proportion=10
# each test will only change 1 aspect
# each test will run 5 times and get the average
# 3 evaluations: 1）avg consistency，2）avg similarity，3）avg boolean loss

g_num_e = np.array(list(range(10, 51, 10)))
r_proportions = np.array(list(range(10, 41, 5))) / 100
g_k_proportions = np.array(list(range(10, 41, 5))) / 100
g_k_bias = np.array(list(range(0, 101, 10))) / 100
l_num_proportion = np.array(list(range(5, 16, 1))) / 10

test_code = 2


def generate_pdf(alpha, ret_dim):
    # make sure alpha is in [0, 1]
    alpha = min(max(alpha, 0), 1)
    # use 0.5 as the default mean value
    means = np.full(ret_dim, 0.5)
    # adjust var using alpha, when alpha=0, var is max
    # when alpha=1, val is min
    variance = 0.1 * (1 - alpha) + 0.01 * alpha
    # use normal distribution
    pdf_values = np.random.normal(loc=means, scale=np.sqrt(variance), size=ret_dim)
    return pdf_values


if __name__ == "__main__":

    consistency = []
    similarity = []
    boolean_loss = []
    torch.set_printoptions(2, linewidth=150, sci_mode=False)

    num_g_entities = 20
    l_num_proportion = 1
    r_proportion = 0.2
    g_k_bias = 0.5

    for _ in range(5):
        c, s, bl = [], [], []
        for g_k_proportion in g_k_proportions:  # test on the third aspect
            tc = []
            ts = []
            tbl = []
            keyword = ["_H", "_R", "_T"]
            g_entities = ["gC_" + str(i) for i in range(num_g_entities)]
            num_g_relations = int(num_g_entities * r_proportion)
            g_relations = ["gR_" + str(i) for i in range(num_g_relations)]
            g_space = VSASpace(VSA_dim, keyword, g_entities, g_relations)

            structure_size = [num_g_relations, num_g_entities, num_g_entities]
            num_l_entities = structure_size[1] * l_num_proportion
            l_entities = ["C_" + str(i) for i in range(num_l_entities)]
            num_l_relations = structure_size[0] * l_num_proportion
            l_relations = ["R_" + str(i) for i in range(num_l_relations)]
            l_space = VSASpace(VSA_dim, keyword, l_entities, l_relations, True)
            l_space.assign_k(g_space.codes_k)

            g_structure_mat = torch.rand([1, num_g_relations, num_g_entities, num_g_entities])
            bias_mask = generate_pdf(g_k_bias, num_g_relations)
            for i in range(g_structure_mat.shape[1]):
                g_structure_mat[0, i] = torch.where(g_structure_mat[0, i] < g_k_proportion * bias_mask[i],
                                                    torch.tensor(1.), torch.tensor(0.))
            g_knowledge = [[g_entities[each[1]], g_relations[each[0]], g_entities[each[2]]] for each in
                           torch.nonzero(g_structure_mat)]
            g_structure_vec = matrix_to_vector(g_structure_mat, g_space)

            # ==========================================================================================================

            encoder = Encoder(10, structure_size)
            decoder = Decoder(structure_size)
            optimizer_e = torch.optim.Adam(encoder.parameters(), lr=lr)
            optimizer_d = torch.optim.Adam(decoder.parameters(), lr=lr)
            optimizer_VSA = torch.optim.Adam([l_space.codes_er], lr=lr * 5e2)

            for ep in range(1, epoch_VSA + 1):
                encoder.train()
                for count, data in tqdm(enumerate(train_loader), total=len(train_loader)):
                    img, label = data
                    img = img.view(img.size(0), -1)
                    C, S = encoder(img)
                    R = decoder(S)
                    S = S.view([-1] + structure_size)
                    mapping, mapping_i = get_mapping(l_space, g_space, 0.5)
                    _, loss = custom_loss(C, label, matrix_to_vector(S, l_space), g_structure_vec, l_space, g_space,
                                          img, R,
                                          mapping_i, g_knowledge, S, True)
                    optimizer_e.zero_grad()
                    optimizer_d.zero_grad()
                    optimizer_VSA.zero_grad()
                    loss.backward()
                    optimizer_e.step()
                    optimizer_d.step()
                    optimizer_VSA.step()

                encoder.eval()
                VSA_fitting_report(l_space, g_space, tc, ts, tbl)
                correct = 0
                with torch.no_grad():
                    for count, data in enumerate(test_loader):
                        img, target = data
                        img = img.view(-1, 28 * 28)
                        C, _ = encoder(img)
                        pred = C.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                    print("Classification Acc: ", correct / len(test_loader.dataset))

            c.append(tc)
            s.append(ts)
            bl.append(tbl)
        consistency.append(c)
        similarity.append(s)
        boolean_loss.append(bl)

    print(consistency)
    print(similarity)
    print(boolean_loss)

    consistency = np.array(consistency)
    similarity = np.array(similarity)
    boolean_loss = np.array(boolean_loss)

    np.save("./Saves/test_2_consistency.npy", consistency)
    np.save("./Saves/test_2_similarity.npy", similarity)
    np.save("./Saves/test_2_boolean_loss.npy", boolean_loss)

    plt.figure()
    plt.grid()
    plt.xlabel("epoch")
    plt.ylabel("consistency")
    for i, each in enumerate(consistency.mean(0)):
        plt.plot(range(epoch_VSA), each, label="%k=" + str(g_k_proportions[i]))
    plt.legend()
    plt.show()

    plt.figure()
    plt.grid()
    plt.xlabel("epoch")
    plt.ylabel("similarity")
    for i, each in enumerate(similarity.mean(0)):
        plt.plot(range(epoch_VSA), each, label="%k=" + str(g_k_proportions[i]))
    plt.legend()
    plt.show()

    plt.figure()
    plt.grid()
    plt.xlabel("epoch")
    plt.ylabel("boolean_loss")
    for i, each in enumerate(boolean_loss.mean(0)):
        plt.plot(range(epoch_VSA), each, label="%k=" + str(g_k_proportions[i]))
    plt.legend()
    plt.show()
