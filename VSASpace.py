import torch

import Utils

"""
_H: triplet head role in a product
_T: triplet tail role in a product
_R: the relation role in an image
"""


def matrix_to_vector(mat, space, threshold: float = 0.5):
    """
    Used to turn KGV_NN (or any such matrices) to vectors.
    :param mat: KGV_NN or similar matrices.
    :param space: The corresponding VSA.
    :param threshold: Threshold deciding what values can be turned to vectors.
    :return:
    """
    _H, _R, _T = space.get("_H"), space.get("_R"), space.get("_T")  # get the keywords
    RHT = torch.nonzero(mat.mean(0) > threshold)
    H = space.codes_er[RHT[:, 1]]
    T = space.codes_er[RHT[:, 2]]
    R = space.codes_er[len(space.entities) + RHT[:, 0]]
    return VSASpace.bundle(VSASpace.bind(H, _H), VSASpace.bind(R, _R), VSASpace.bind(T, _T))


def matrix_to_triplet(mat, space, threshold: float = 0.5, overwrite=None, show=False):
    """
    Used to turn KGV_NN (or any such matrices) to triplets, mainly for printing the results.
    :param mat: as above.
    :param space: as above.
    :param threshold: as above.
    :param overwrite: whether we already have a mapping.
    :param show: print the output.
    """
    if overwrite is None:
        overwrite = {}
    RHT = torch.nonzero(mat.mean(0) > threshold)
    triplets = []
    for each in RHT:
        triplets.append(overwrite.get(space.entities[each[1]], space.entities[each[1]]) + " --" +
                        overwrite.get(space.relations[each[0]], space.relations[each[0]]) + "-> " +
                        overwrite.get(space.entities[each[2]], space.entities[each[2]]))
        if show:
            print(triplets[-1])
    return triplets


class VSASpace:

    def __init__(self, dim, k, e, r, learnable=False):
        """
        A bipolar VSA.

        :param dim: dim of a VSA space, e.g., 10000
        :param k: keyword
        :param e: entities
        :param r: relations
        """
        self.dim = dim
        self.keyword = k
        self.entities = e
        self.relations = r

        # codes (vectors) of keywords
        self.codes_k = torch.bernoulli(torch.ones([len(k), dim]) * 0.5) * 2 - 1
        # codes (vectors) of entities and relations
        self.codes_er = (torch.bernoulli(torch.ones([len(e) + len(r), dim]) * 0.5) * 2 - 1).requires_grad_(learnable)

        # the codebook for keywords
        self.codebook_k = {each: i for i, each in enumerate(k)}

        # the codebook for entities and relations
        self.codebook = {each: i for i, each in enumerate(e + r)}
        self.codebook_i = {i: each for i, each in enumerate(e + r)}

    def assign_k(self, vecs):
        # assign the keyword with vecs input.
        self.codes_k = vecs

    def get(self, word):
        # get the code (vector) of a word (str).
        if word in self.codebook_k:
            return self.codes_k[self.codebook_k[word]]
        elif word in self.codebook:
            return self.codes_er[self.codebook[word]]

    def find(self, vecs):
        """
        Find the most matched vectors (of atomic concepts) given the input.
        """
        tmp = Utils.cosine_similarity(vecs, self.codes_er)
        similarities, indices = tmp.max(dim=1)

        return self.codes_er[indices]

    @staticmethod
    def bind(vec_1: torch.tensor, vec_2: torch.tensor):
        # binding is elementwise multiplication
        # supports self-inverse bind
        ret = vec_1 * vec_2
        return ret

    @staticmethod
    def bundle(*vecs: torch.tensor):
        # bundling is just elementwise addition
        if len(vecs) > 1:
            ret = vecs[0]
            for i in range(1, len(vecs)):
                ret += vecs[i]
            return ret

    def restore(self, vecs: torch.tensor):
        """
        Restore means finding the H, R, T of a vector.
        """
        _H = self.get("_H")
        _R = self.get("_R")
        _T = self.get("_T")
        # extract H
        H_vecs = self.find(VSASpace.bind(vecs, _H))
        # extract R
        R_vecs = self.find(VSASpace.bind(vecs, _R))
        # extract T
        T_vecs = self.find(VSASpace.bind(vecs, _T))

        return H_vecs, R_vecs, T_vecs


if __name__ == "__main__":
    keywords = ["_H", "_R", "_T"]
    entities = ["E1", "E2", "E3"]
    relations = ["R1"]

    s = VSASpace(1000, keywords, entities, relations)
    m = torch.rand([1, 1, 3, 3])
    print(matrix_to_vector(m, s, 0.5))
