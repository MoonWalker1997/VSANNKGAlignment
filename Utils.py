import torch
from scipy.optimize import linear_sum_assignment
import numpy as np


def cosine_similarity(vecs_1, vecs_2):
    """
    can be used to calculate the cosine similarity between 1) some vectors and 2) some vectors
    the function in pytorch only support calculating the cosine similarity between 1) one vector and 2) some vectors
    """
    # can be used to calculate the cosine similarity between 1) some vectors and 2) some vectors
    # the function in pytorch only support calculating the cosine similarity between 1) one vector and 2) some vectors
    return torch.einsum("ik,jk->ij", vecs_1, vecs_2) / torch.einsum("i,j->ij",
                                                                    vecs_1.norm(dim=-1),
                                                                    vecs_2.norm(dim=-1))


def get_given_structure_mat(given_knowledge, given_e, given_r):
    """
    Turn triplets to a matrix.
    :param given_knowledge: given triplets.
    :param given_e: entities used.
    :param given_r: relations used.
    """
    mat = torch.zeros([1, len(given_r), len(given_e), len(given_e)])
    for each in given_knowledge:
        mat[0, given_r.index(each[1]), given_e.index(each[0]), given_e.index(each[2])] = 1
    return mat


def VSA_fitting_report(learnable_space, given_space, c, s, bl, p=True):
    """
    Check 1) whether the fitting of VSA_NN and VSA_G is good, and 2) check some properties of VSA_NN.
    :param learnable_space: VSA_NN.
    :param given_space: VSA_G.
    :param c: for storing the consistency.
    :param s: for storing the similarity.
    :param bl: for storing the Boolean loss.
    :param p: whether print the evaluations
    """
    batch_cos_similarity = cosine_similarity(learnable_space.codes_er, given_space.codes_er).detach()
    batch_cost = -batch_cos_similarity
    batch_row, batch_col = linear_sum_assignment(batch_cost)
    # consistency, the higher, the better
    tc = batch_cos_similarity[batch_row, batch_col].mean().item()
    c.append(tc)
    if p:
        print("Avg consistency:", tc)

    # independence, the lower, the better
    batch_cos_similarity = cosine_similarity(learnable_space.codes_er, learnable_space.codes_er).detach()
    batch_cos_similarity -= torch.diag_embed(batch_cos_similarity.diag())
    ts = abs(batch_cos_similarity.sum()) / (batch_cos_similarity.shape[0] ** 2)
    s.append(ts.item())
    if p:
        print("Avg similarity:", ts.item())

    # Boolean loss, the lower, the better.
    tbl = torch.min((learnable_space.codes_er + 1) ** 2, (learnable_space.codes_er - 1) ** 2).mean()
    bl.append(tbl.item())
    if p:
        print("Avg Boolean loss:", tbl.item())


def get_mapping(learnable_space, given_space, threshold=-1):
    """
    Find the matched concepts in VSA_NN and VSA_G. With the threshold as the minimum matching requirement.
    :param learnable_space: VSA_NN.
    :param given_space: VSA_G.
    :param threshold: the matching threshold.
    """
    finished_cos_similarity = cosine_similarity(learnable_space.codes_er, given_space.codes_er).detach()
    finished_cost = -finished_cos_similarity

    # filter out the low-quality matching
    feasible_matching = torch.nonzero(finished_cost <= -threshold)
    finished_row, finished_col = linear_sum_assignment(finished_cost)
    finished_row_thresholding, finished_col_thresholding = [], []
    for i in range(finished_row.shape[0]):
        if finished_row[i] in feasible_matching[:, 0] and finished_col[i] in feasible_matching[:, 1]:
            finished_row_thresholding.append(finished_row[i])
            finished_col_thresholding.append(finished_col[i])
    finished_row, finished_col = np.array(finished_row_thresholding), np.array(finished_col_thresholding)

    # get the mapping (concepts to concepts), and mapping_i (mapping_index) for concepts and indices.
    mapping = {
        (learnable_space.entities + learnable_space.relations)[finished_row[i]]:
            (given_space.entities + given_space.relations)[finished_col[i]]
        for i in range(finished_row.shape[0])}
    mapping_i = {(given_space.entities + given_space.relations)[finished_col[i]]: finished_row[i] for i in
                 range(finished_row.shape[0])}

    if len(mapping) == 0 or len(mapping_i) == 0:
        return None, None
    else:
        return mapping, mapping_i
