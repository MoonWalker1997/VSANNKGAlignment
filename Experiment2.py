import numpy as np
import torch
from tqdm import tqdm

from Data import train_loader, test_loader
from Loss import custom_loss
from Model import Encoder, Decoder
from Utils import get_given_structure_mat, VSA_fitting_report, get_mapping
from VSASpace import VSASpace, matrix_to_vector

VSA_dim = 10000
lr = 0.001
epoch_VSA = 5
check_train = True
task_output_size = 10

# experiment_index = [0, 0, 0]
# experiment_index = [0, 0, 1]
# experiment_index = [0, 1, 0]
# experiment_index = [0, 1, 1]
# experiment_index = [1, 0, 0]
# experiment_index = [1, 0, 1]
# experiment_index = [1, 1, 0]
experiment_index = [1, 1, 1]

num_g_entities = [20, 40]
num_knowledge = [80, 160]
KG_NN_item_p = [0.5, 2]

if __name__ == "__main__":

    cs = []

    for _ in range(5):

        torch.set_printoptions(2, linewidth=150, sci_mode=False)

        # define keywords
        keyword = ["_H", "_R", "_T"]

        # define KG_G
        g_entities = ["gC_%i" % i for i in range(num_g_entities[experiment_index[0]])]
        g_relations = ["gR_%i" % i for i in range(int(num_g_entities[experiment_index[0]] * 0.2))]

        num_relations_per_rel = num_knowledge[experiment_index[1]] // len(g_relations)
        g_structure_mat = torch.zeros([len(g_relations), len(g_entities), len(g_entities)])
        for i in range(len(g_relations)):
            random_positions = np.random.choice(len(g_entities) * len(g_entities), size=num_relations_per_rel,
                                                replace=False)
            random_positions_grid = [(pos // len(g_entities), pos % len(g_entities)) for pos in random_positions]
            for each in random_positions_grid:
                g_structure_mat[i, each[0], each[1]] = 1.
        g_knowledge = [[g_entities[each[1]], g_relations[each[0]], g_entities[each[2]]] for each in
                       torch.nonzero(g_structure_mat)]

        # define VSA_G
        g_space = VSASpace(VSA_dim, keyword, g_entities, g_relations)

        g_structure_mat = get_given_structure_mat(g_knowledge, g_entities, g_relations)
        g_structure_vec = matrix_to_vector(g_structure_mat, g_space)

        # define VSA_NN
        structure_size = [max(1, int(len(g_relations) * KG_NN_item_p[experiment_index[2]])),
                          max(1, int(len(g_entities) * KG_NN_item_p[experiment_index[2]])),
                          max(1, int(len(g_entities) * KG_NN_item_p[experiment_index[2]]))]
        l_entities = ["C_%i" % i for i in range(structure_size[1])]
        l_relations = ["R_%i" % i for i in range(structure_size[0])]

        l_space = VSASpace(VSA_dim, keyword, l_entities, l_relations, True)
        l_space.assign_k(g_space.codes_k)

        # ==============================================================================================================

        # create the model
        encoder = Encoder(10, structure_size)
        decoder = Decoder(structure_size)
        optimizer_e = torch.optim.Adam(encoder.parameters(), lr=lr)
        optimizer_d = torch.optim.Adam(decoder.parameters(), lr=lr)
        optimizer_VSA = torch.optim.Adam([l_space.codes_er], lr=lr * 5e2)

        c = []
        s = []
        bl = []

        for ep in range(1, epoch_VSA + 1):
            # training
            encoder.train()
            for count, data in tqdm(enumerate(train_loader), total=len(train_loader)):
                img, label = data
                img = img.view(img.size(0), -1)
                C, S = encoder(img)
                R = decoder(S)
                S = S.view([-1] + structure_size)
                # get the mapping
                mapping, mapping_i = get_mapping(l_space, g_space, 0.5)
                # calculate the loss
                tmp_ctl, loss = custom_loss(C, label, matrix_to_vector(S, l_space), g_structure_vec, l_space, g_space,
                                            img, R, mapping_i, g_knowledge, S, True)
                optimizer_e.zero_grad()
                optimizer_d.zero_grad()
                optimizer_VSA.zero_grad()
                loss.backward()
                optimizer_e.step()
                optimizer_d.step()
                optimizer_VSA.step()

            encoder.eval()
            # print the learning results (consistency, similarity, Boolean loss)
            VSA_fitting_report(l_space, g_space, c, s, bl)
            # test on the classification task
            correct = 0
            with torch.no_grad():
                for count, data in enumerate(test_loader):
                    img, target = data
                    img = img.view(-1, 28 * 28)
                    C, _ = encoder(img)
                    pred = C.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                print("Classification Acc: ", correct / len(test_loader.dataset))

        cs.append(c[-1])

    print(cs)
