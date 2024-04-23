import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from Data import train_loader, test_loader
from Loss import custom_loss
from Model import structure_size, Encoder, Decoder
from Utils import get_given_structure_mat, VSA_fitting_report, get_mapping
from VSASpace import VSASpace, matrix_to_vector

VSA_dim = 10000
lr = 0.001
epoch_VSA = 1
epoch_alignment = 1
check_train = True
task_output_size = 10

if __name__ == "__main__":
    torch.set_printoptions(2, linewidth=150, sci_mode=False)

    # define keywords
    keyword = ["_H", "_R", "_T"]

    # define KG_G
    g_knowledge = [["one", "contains", "straight"],
                   ["two", "contains", "curve"],
                   ["two", "contains", "straight"],
                   ["three", "contains", "curve"],
                   ["four", "contains", "straight"],
                   ["four", "contains", "cross"],
                   ["five", "contains", "straight"],
                   ["five", "contains", "curve"],
                   ["six", "contains", "curve"],
                   ["seven", "contains", "straight"],
                   ["eight", "contains", "curve"],
                   ["eight", "contains", "cross"],
                   ["nine", "contains", "straight"],
                   ["nine", "contains", "curve"],
                   ["zero", "contains", "curve"],
                   ]
    g_entities = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero", "straight", "curve",
                  "cross"]
    g_relations = ["contains"]
    # define VSA_G
    g_space = VSASpace(VSA_dim, keyword, g_entities, g_relations)

    g_structure_mat = get_given_structure_mat(g_knowledge, g_entities, g_relations)
    g_structure_vec = matrix_to_vector(g_structure_mat, g_space)

    # define VSA_NN
    l_space = VSASpace(VSA_dim, keyword, ["C_" + str(i) for i in range(structure_size[1])],
                       ["R_" + str(i) for i in range(structure_size[0])],
                       True)
    l_space.assign_k(g_space.codes_k)

    # ==================================================================================================================

    # create the model
    encoder = Encoder(task_output_size)
    decoder = Decoder()
    optimizer_e = torch.optim.Adam(encoder.parameters(), lr=lr)
    optimizer_d = torch.optim.Adam(decoder.parameters(), lr=lr)
    optimizer_VSA = torch.optim.Adam([l_space.codes_er], lr=lr * 5e2)

    # print the VSA_NN codebook before training (first 50-D)
    plt.imshow(l_space.codes_er[:, :50].detach())
    plt.show()

    for ep in range(1, epoch_VSA + 1):
        # training
        encoder.train()
        # ctl = 0
        for count, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            img, label = data
            img = img.view(img.size(0), -1)
            C, S = encoder(img)
            R = decoder(S)
            S = S.view([-1] + structure_size)
            # get the mapping
            mapping, mapping_i = get_mapping(l_space, g_space, 0.5)
            # calculate the loss
            tmp_ctl, loss = custom_loss(C, label, matrix_to_vector(S, l_space), g_structure_vec, l_space, g_space, img,
                                        R, mapping_i, g_knowledge, S, True)
            # ctl += tmp_ctl
            optimizer_e.zero_grad()
            optimizer_d.zero_grad()
            optimizer_VSA.zero_grad()
            loss.backward()
            optimizer_e.step()
            optimizer_d.step()
            optimizer_VSA.step()
        # print(ctl)

        encoder.eval()
        # print the learning results (consistency, similarity, Boolean loss)
        VSA_fitting_report(l_space, g_space, [], [], [])
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

    # print the VSA_NN codebook after training (first 50-D)
    plt.imshow(l_space.codes_er[:, :50].detach())
    plt.show()

    # print the mapping
    mapping, mapping_i = get_mapping(l_space, g_space)
    print("========")
    print("Mapping:")
    # print(mapping)
    print("--------")
    for each in mapping:
        print(each, mapping[each])

    # ==================================================================================================================

    # summarize the KGV_NN for each number
    S_zero = None
    S_one = None
    S_two = None
    S_three = None
    S_four = None
    S_five = None
    S_six = None
    S_seven = None
    S_eight = None
    S_nine = None
    with torch.no_grad():
        for count, data in enumerate(test_loader):
            img, target = data
            img = img.view(-1, 28 * 28)
            C, S = encoder(img)
            S = S.view([-1] + structure_size)
            for i in range(S.shape[0]):
                if target[i] == 0:
                    if S_zero is None:
                        S_zero = S[i]
                    else:
                        S_zero = (S_zero + S[i]) / 2
                elif target[i] == 1:
                    if S_one is None:
                        S_one = S[i]
                    else:
                        S_one = (S_one + S[i]) / 2
                elif target[i] == 2:
                    if S_two is None:
                        S_two = S[i]
                    else:
                        S_two = (S_two + S[i]) / 2
                elif target[i] == 3:
                    if S_three is None:
                        S_three = S[i]
                    else:
                        S_three = (S_three + S[i]) / 2
                elif target[i] == 4:
                    if S_four is None:
                        S_four = S[i]
                    else:
                        S_four = (S_four + S[i]) / 2
                elif target[i] == 5:
                    if S_five is None:
                        S_five = S[i]
                    else:
                        S_five = (S_five + S[i]) / 2
                elif target[i] == 6:
                    if S_six is None:
                        S_six = S[i]
                    else:
                        S_six = (S_six + S[i]) / 2
                elif target[i] == 7:
                    if S_seven is None:
                        S_seven = S[i]
                    else:
                        S_seven = (S_seven + S[i]) / 2
                elif target[i] == 8:
                    if S_eight is None:
                        S_eight = S[i]
                    else:
                        S_eight = (S_eight + S[i]) / 2
                elif target[i] == 9:
                    if S_nine is None:
                        S_nine = S[i]
                    else:
                        S_nine = (S_nine + S[i]) / 2

    # Visualize all KGV_NN's above
    for i, each in enumerate([S_zero, S_one, S_two, S_three, S_four, S_five, S_six, S_seven, S_eight, S_nine]):
        plt.subplot(2, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(torch.clamp(each.mean(0) - 0.5, 0, 1))
    plt.show()

    # Visualize the decoded KGV_NN
    for i, each in enumerate([S_zero, S_one, S_two, S_three, S_four, S_five, S_six, S_seven, S_eight, S_nine]):
        plt.subplot(2, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(decoder(each.view([1, -1])).view(structure_size).mean(0).detach())
    plt.show()
