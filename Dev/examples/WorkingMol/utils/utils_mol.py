import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdm
from utils.utils_features import (atom_to_feature_vector, bond_to_feature_vector)


# #
# spe_vob = codecs.open('/usr/src/kdd/examples/utils/SPE_ChEMBL.txt')
# spe = SPE_Tokenzer(spe_vob)


def get_global_features(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    u = []
    # Now get some specific features
    # fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    #  factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    # feats = factory.GetFeaturesForMol(mol)

    # First get some basic features
    # natoms = mol.GetNumAtoms()
    # nbonds = mol.GetNumBonds()
    # mw = Descriptors.ExactMolWt(mol)
    # HeavyAtomMolWt = Descriptors.HeavyAtomMolWt(mol)
    # NumValenceElectrons = Descriptors.NumValenceElectrons(mol)
    ''' # These four descriptors are producing the value of infinity for refcode_csd = YOLJUF (CCOP(=O)(Cc1ccc(cc1)NC(=S)NP(OC(C)C)(OC(C)C)[S])OCC\t\n)
    MaxAbsPartialCharge = Descriptors.MaxAbsPartialCharge(mol)
    MaxPartialCharge = Descriptors.MaxPartialCharge(mol)
    MinAbsPartialCharge = Descriptors.MinAbsPartialCharge(mol)
    MinPartialCharge = Descriptors.MinPartialCharge(mol)
    '''
    # FpDensityMorgan1 = Descriptors.FpDensityMorgan1(mol)
    # FpDensityMorgan2 = Descriptors.FpDensityMorgan2(mol)
    # FpDensityMorgan3 = Descriptors.FpDensityMorgan3(mol)

    # # Get some features using chemical feature factory
    #
    # nbrAcceptor = 0
    # nbrDonor = 0
    # nbrHydrophobe = 0
    # nbrLumpedHydrophobe = 0
    # nbrPosIonizable = 0
    # nbrNegIonizable = 0
    #
    # for j in range(len(feats)):
    #     # print(feats[j].GetFamily(), feats[j].GetType())
    #     if ('Acceptor' == (feats[j].GetFamily())):
    #         nbrAcceptor = nbrAcceptor + 1
    #     elif ('Donor' == (feats[j].GetFamily())):
    #         nbrDonor = nbrDonor + 1
    #     elif ('Hydrophobe' == (feats[j].GetFamily())):
    #         nbrHydrophobe = nbrHydrophobe + 1
    #     elif ('LumpedHydrophobe' == (feats[j].GetFamily())):
    #         nbrLumpedHydrophobe = nbrLumpedHydrophobe + 1
    #     elif ('PosIonizable' == (feats[j].GetFamily())):
    #         nbrPosIonizable = nbrPosIonizable + 1
    #     elif ('NegIonizable' == (feats[j].GetFamily())):
    #         nbrNegIonizable = nbrNegIonizable + 1
    #     else:
    #         pass
    #         # print(feats[j].GetFamily())
    # u = [natoms, nbonds, mw, HeavyAtomMolWt, NumValenceElectrons, FpDensityMorgan1, FpDensityMorgan2, \
    #      FpDensityMorgan3, nbrAcceptor, nbrDonor, nbrHydrophobe, nbrLumpedHydrophobe, \
    #      nbrPosIonizable, nbrNegIonizable]
    # Now get some features using rdMolDescriptors

    moreGlobalFeatures = [rdm.CalcNumRotatableBonds(mol), rdm.CalcChi0n(mol), rdm.CalcChi0v(mol), \
                          rdm.CalcChi1n(mol), rdm.CalcChi1v(mol), rdm.CalcChi2n(mol), rdm.CalcChi2v(mol), \
                          rdm.CalcChi3n(mol), rdm.CalcChi4n(mol), rdm.CalcChi4v(mol), \
                          rdm.CalcFractionCSP3(mol), rdm.CalcHallKierAlpha(mol), rdm.CalcKappa1(mol), \
                          rdm.CalcKappa2(mol), rdm.CalcLabuteASA(mol), \
                          rdm.CalcNumAliphaticCarbocycles(mol), rdm.CalcNumAliphaticHeterocycles(mol), \
                          rdm.CalcNumAliphaticRings(mol), rdm.CalcNumAmideBonds(mol), \
                          rdm.CalcNumAromaticCarbocycles(mol), rdm.CalcNumAromaticHeterocycles(mol), \
                          rdm.CalcNumAromaticRings(mol), rdm.CalcNumBridgeheadAtoms(mol), rdm.CalcNumHBA(mol), \
                          rdm.CalcNumHBD(mol), rdm.CalcNumHeteroatoms(mol), rdm.CalcNumHeterocycles(mol), \
                          rdm.CalcNumLipinskiHBA(mol), rdm.CalcNumLipinskiHBD(mol), rdm.CalcNumRings(mol), \
                          rdm.CalcNumSaturatedCarbocycles(mol), rdm.CalcNumSaturatedHeterocycles(mol), \
                          rdm.CalcNumSaturatedRings(mol), rdm.CalcNumSpiroAtoms(mol), rdm.CalcTPSA(mol)]

    # u = u + moreGlobalFeatures
    # zp
    u = moreGlobalFeatures
    u = np.array(u).T
    # Some of the descriptors produice NAN. We can convert them to 0
    # If you are getting outliers in the training or validation set this could be
    # Because some important features were set to zero here because it produced NAN
    # Removing those features from the feature set might remove the outliers

    # u[np.isnan(u)] = 0

    # u = torch.tensor(u, dtype=torch.float)
    return (u)


# def smiles2pes(smiles_string):
# Did nopt work well
#
#     subgraphs = []
#     ret = spe.tokenize(smiles_string).split(" ")
#     for r in ret:
#         try:
#             graph = smiles2graph(r)
#             subgraphs.append(graph)
#         except:
#             pass
#   #  print(len(ret), len(subgraphs))
#     return subgraphs

def smiles2subgraphs(smiles_string, path_size=5):
    m = Chem.MolFromSmiles(smiles_string)

    ret = computesubgraphsmarts(m, path_size)
    subgraphs = []
    for r in ret:
        try:
            subgraphs.append(smiles2graph(r))
        except:
            pass
    # print(len(ret), len(subgraphs))
    return subgraphs


def smiles2graph(smiles_string):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string)
    # atoms
    atom_features_list = []

    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    edges_list = []
    if len(mol.GetBonds()) > 0:  # mol has bonds

        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:  # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = dict()

    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['nodes_num'] = len(x)
    graph['smiles'] = smiles_string

    return graph


def computepathsmarts(mol, size=5):
    if size <= 0: size = 7  # default to 7 atoms
    ret = set()
    for length in range(2, size + 1):
        paths = Chem.FindAllPathsOfLengthN(mol, length, useBonds=False)
        for path in paths:
            smi = Chem.MolFragmentToSmiles(mol, path, canonical=True, allBondsExplicit=True)
            ret.add(smi)
    return ret


def computesubgraphsmarts(mol, size=5):
    '''Given an rdkit mol, extract all the paths up to length size (if zero,
    use default length).  Return smarts expressions for these paths.'''
    if size <= 0: size = 6  # default to 6 bonds
    # find all paths of various sizes
    paths = Chem.FindAllSubgraphsOfLengthMToN(mol, 1, size)
    # paths is a list of lists, each of which is the bond indices for paths of a different length
    paths = [path for subpath in paths for path in subpath]
    ret = set()
    for path in paths:
        atoms = set()
        for bidx in path:
            atoms.add(mol.GetBondWithIdx(bidx).GetBeginAtomIdx())
            atoms.add(mol.GetBondWithIdx(bidx).GetEndAtomIdx())
        smi = Chem.MolFragmentToSmiles(mol, atoms, canonical=True, allBondsExplicit=True)
        ret.add(smi)
    return ret


def computecircularsmarts(mol, size=5):
    '''Given an rdkit mol, extract all the circular type descriptors up to
    radius size (if zero, use default length).  Return smarts expressions for these fragments.'''
    if size <= 0: size = 2
    ret = set()
    for a in mol.GetAtoms():
        for r in range(1, size + 1):
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, r, a.GetIdx())
            atoms = set()
            for bidx in env:
                atoms.add(mol.GetBondWithIdx(bidx).GetBeginAtomIdx())
                atoms.add(mol.GetBondWithIdx(bidx).GetEndAtomIdx())
            smi = Chem.MolFragmentToSmiles(mol, atoms, canonical=True, allBondsExplicit=True)
            ret.add(smi)
    return ret


if __name__ == '__main__':

    smi = 'O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5'
    smi = 'CCCOC1CC(C1)Cl'
    m = Chem.MolFromSmiles(smi)
    ret = computesubgraphsmarts(m, 5)

    ok = []
    for r in ret:
        try:
            graph = smiles2graph(r)
            ok.append(r)
        except:
            pass
    print(len(ret), len(ok))
    print(ok)
#     #
# #'/home/zpehlivan/Ina/Workplace/pycharm-projects/KDD/Docker/ogb/ogb/utils/SPE_ChEMBL.txt'
#     spe_vob= codecs.open(os.path.dirname(os.path.abspath(__file__)) + '/SPE_ChEMBL.txt')
#     spe = SPE_Tokenizer(spe_vob)
#     ok = []
#     sss = spe.tokenize(smi).split(" ")
#     for r in sss:
#         try:
#             graph = smiles2graph(r)
#
#             ok.append(r)
#         except :
#             pass
#     print(len(sss), len(ok))
