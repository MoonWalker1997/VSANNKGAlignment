import json

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from Loss import custom_loss_2
from Utils import get_mapping, VSA_fitting_report
from VSASpace import VSASpace, matrix_to_vector

# ======================================================================================================================
# pre process

# package_file = "/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_1_university_ground_truth.jsonl"
# package_file = "/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_2_musicalwork_ground_truth.jsonl"
# package_file = "/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_3_airport_ground_truth.jsonl"
# package_file = "/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_4_building_ground_truth.jsonl"
# package_file = "/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_5_athlete_ground_truth.jsonl"
# package_file = "/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_6_politician_ground_truth.jsonl"
# package_file = "/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_7_company_ground_truth.jsonl"
# package_file = "/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_8_celestialbody_ground_truth.jsonl"
# package_file = "/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_9_astronaut_ground_truth.jsonl"
# package_file = "/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_10_comicscharacter_ground_truth.jsonl"
# package_file = "/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_11_meanoftransportation_ground_truth.jsonl"
# package_file = "/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_12_monument_ground_truth.jsonl"
# package_file = "/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_13_food_ground_truth.jsonl"
# package_file = "/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_14_writtenwork_ground_truth.jsonl"
# package_file = "/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_15_sportsteam_ground_truth.jsonl"
# package_file = "/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_16_city_ground_truth.jsonl"
# package_file = "/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_17_artist_ground_truth.jsonl"
# package_file = "/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_18_scientist_ground_truth.jsonl"
package_file = "/Text2KGBench-main/data/dbpedia_webnlg/ground_truth/ont_19_film_ground_truth.jsonl"

g_entities = set()
g_relations = set()

package_data = []

with open(package_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        tmp = [data["sent"], []]
        for each in data["triples"]:
            g_entities.add(each["sub"])
            g_relations.add(each["rel"])
            g_entities.add(each["obj"])
            tmp[1].append([each["sub"], each["rel"], each["obj"]])
        package_data.append(tmp)

# ======================================================================================================================
# pretrained LLM definition

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def text_to_embedding(text, llm):
    x = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        y = llm(**x)
    return y.last_hidden_state.mean(dim=1)


# text2embedding("I love ITZY.", model)

# ======================================================================================================================
# lens definition


class Lens(torch.nn.Module):
    def __init__(self, ssize):
        """
        :param ssize: structure size.
        """
        super(Lens, self).__init__()
        # generator (generate KGV_NN)
        self.generator = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, ssize[0] * ssize[1] * ssize[2]),
            torch.nn.Sigmoid()  # activated with Sigmoid to make sure the value is in [0,1]
        )

    def forward(self, e):
        structure = self.generator(e)
        return structure


# ======================================================================================================================
# util

def knowledge_to_matrix(ssize, knowledge, space):
    mask = torch.zeros(ssize)
    for each in knowledge:
        mask[space.codebook[each[1]] - len(space.entities), space.codebook[each[0]], space.codebook[each[1]]] = 1.
    return mask


def knowledge_to_vectors(knowledge, space):
    ret = []
    for each in knowledge:
        ret.append(space.bundle(space.bind(space.get("_H"), space.get(each[0])),
                                space.bind(space.get("_R"), space.get(each[1])),
                                space.bind(space.get("_T"), space.get(each[2]))))
    return torch.vstack(ret)


def KG_to_triples(S, l_space, mapping):
    if mapping is None:
        return []
    output_triples = []
    for rel_idx in range(S.shape[1]):
        for ent_idx_1 in range(S.shape[2]):
            for ent_idx_2 in range(S.shape[3]):
                R = l_space.codebook_i[len(l_space.entities) + rel_idx]
                E1 = l_space.codebook_i[ent_idx_1]
                E2 = l_space.codebook_i[ent_idx_2]

                if E1 in mapping and R in mapping and E2 in mapping:
                    output_triples.append([S[0, rel_idx, ent_idx_1, ent_idx_2].item(),
                                           (mapping[E1], mapping[R], mapping[E2])])
    output_triples = sorted(output_triples, reverse=True)
    return output_triples

# ======================================================================================================================


VSA_dim = 10000
lr = 0.001
epoch_VSA = 10

l_num_proportion = 0.2

# VSA definition
keyword = ["_H", "_R", "_T"]
g_space = VSASpace(VSA_dim, keyword, list(g_entities), list(g_relations))

structure_size = [int(len(g_relations) * l_num_proportion),
                  int(len(g_entities) * l_num_proportion),
                  int(len(g_entities) * l_num_proportion)]

num_l_entities = structure_size[1]
l_entities = ["C_" + str(i) for i in range(num_l_entities)]
num_l_relations = structure_size[0]
l_relations = ["R_" + str(i) for i in range(num_l_relations)]
l_space = VSASpace(VSA_dim, keyword, l_entities, l_relations, True)
l_space.assign_k(g_space.codes_k)

lens = Lens(structure_size)
optimizer_lens = torch.optim.Adam(lens.parameters(), lr=lr)
optimizer_VSA = torch.optim.Adam([l_space.codes_er], lr=lr * 5e2)

tc, ts, tbl = [], [], []

training_package_data = package_data[:int(len(package_data) * 0.7)]
testing_package_data = package_data[int(len(package_data) * 0.7):]

for ep in range(1, epoch_VSA + 1):

    # training
    lens.train()

    train_loss = 0
    train_abs_matching = 0

    for count, data in tqdm(enumerate(training_package_data)):
        sent, triples = data
        embedding = text_to_embedding(sent, model)
        S = lens(embedding)
        S = S.view([-1] + structure_size)
        mapping, mapping_i = get_mapping(l_space, g_space, 0.5)
        # print(mapping)
        # print(mapping_i)

        output_triples = KG_to_triples(S, l_space, mapping)

        [loss_to_show, abs_matching], loss = custom_loss_2(matrix_to_vector(S, l_space),
                                                           knowledge_to_vectors(triples, g_space),
                                                           l_space, g_space,
                                                           mapping_i, S, triples,
                                                           True)

        train_loss += loss_to_show.item()
        train_abs_matching += abs_matching.item()

        optimizer_lens.zero_grad()
        optimizer_VSA.zero_grad()
        loss.backward()
        optimizer_lens.step()
        optimizer_VSA.step()

    print("train KG loss:", round(train_loss / len(training_package_data), 5))
    print("train KG abs matching:", round(train_abs_matching / len(training_package_data), 5))

    VSA_fitting_report(l_space, g_space, tc, ts, tbl)

    lens.eval()

    test_loss = 0
    test_abs_matching = 0

    with torch.no_grad():
        for count, data in tqdm(enumerate(testing_package_data)):
            sent, triples = data
            embedding = text_to_embedding(sent, model)
            S = lens(embedding)
            S = S.view([-1] + structure_size)
            mapping, mapping_i = get_mapping(l_space, g_space, 0.5)

            output_triples = KG_to_triples(S, l_space, mapping)

            # print(output_triples[:10])

            [loss, abs_matching], _ = custom_loss_2(matrix_to_vector(S, l_space),
                                                    knowledge_to_vectors(triples, g_space),
                                                    l_space, g_space,
                                                    mapping_i, S, triples,
                                                    True)

            test_loss += loss.item()
            test_abs_matching += abs_matching.item()

    print("test KG loss:", round(test_loss / len(testing_package_data), 5))
    print("test KG abs matching:", round(test_abs_matching / len(testing_package_data), 5))

print(tc)
print(ts)
print(tbl)
