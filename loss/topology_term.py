
import torch
import torch.linalg as tla
import numpy as np

from torch import Tensor
from torch import nn

from .loss_classes import LossTerm


"""
Topology - Topology Helper Classes
-------------------------------------------------------------------------------------------------------------------------------------------
Taken from the repository of:
Moor, Horn, Rieck, and Borgwardt "Topological Autoencoders" arXiv:1906.00722 (2020)
"""
class UnionFind:
    '''
    An implementation of a Union--Find class. The class performs path
    compression by default. It uses integers for storing one disjoint
    set, assuming that vertices are zero-indexed.
    '''

    def __init__(self, n_vertices):
        '''
        Initializes an empty Union--Find data structure for a given
        number of vertices.
        '''

        self._parent = np.arange(n_vertices, dtype=int)

    def find(self, u):
        '''
        Finds and returns the parent of u with respect to the hierarchy.
        '''

        if self._parent[u] == u:
            return u
        else:
            # Perform path collapse operation
            self._parent[u] = self.find(self._parent[u])
            return self._parent[u]

    def merge(self, u, v):
        '''
        Merges vertex u into the component of vertex v. Note the
        asymmetry of this operation.
        '''

        if u != v:
            self._parent[self.find(u)] = self.find(v)

    def roots(self):
        '''
        Generator expression for returning roots, i.e. components that
        are their own parents.
        '''

        for vertex, parent in enumerate(self._parent):
            if vertex == parent:
                yield vertex




class PersistentHomologyCalculation:
    def __call__(self, matrix):

        n_vertices = matrix.shape[0]
        uf = UnionFind(n_vertices)

        triu_indices = np.triu_indices_from(matrix)
        edge_weights = matrix[triu_indices]
        edge_indices = np.argsort(edge_weights, kind='stable')

        # 1st dimension: 'source' vertex index of edge
        # 2nd dimension: 'target' vertex index of edge
        persistence_pairs = []

        for edge_index, edge_weight in \
                zip(edge_indices, edge_weights[edge_indices]):

            u = triu_indices[0][edge_index]
            v = triu_indices[1][edge_index]

            younger_component = uf.find(u)
            older_component = uf.find(v)

            # Not an edge of the MST, so skip it
            if younger_component == older_component:
                continue
            elif younger_component > older_component:
                uf.merge(v, u)
            else:
                uf.merge(u, v)

            if u < v:
                persistence_pairs.append((u, v))
            else:
                persistence_pairs.append((v, u))

        # Return empty cycles component
        return np.array(persistence_pairs)




class TopologicalSignatureDistanceOrg(nn.Module):
    """Topological signature."""

    def __init__(self, match_edges=None):
        """Topological signature computation.

        """
        super().__init__()

        self.match_edges = match_edges

        self.signature_calculator = PersistentHomologyCalculation()


    def _get_pairings(self, distances):
        pairs_0 = self.signature_calculator(
            distances.detach().cpu().numpy())

        return pairs_0


    def _select_distances_from_pairs(self, distance_matrix, pairs):
        
        selected_distances = distance_matrix[(pairs[:, 0], pairs[:, 1])]

        return selected_distances


    @staticmethod
    def sig_error(signature1, signature2):
        """Compute distance between two topological signatures."""
        return ((signature1 - signature2)**2).sum(dim=-1)


    @staticmethod
    def _count_matching_pairs(pairs1, pairs2):
        def to_set(array):
            return set(tuple(elements) for elements in array)
        return float(len(to_set(pairs1).intersection(to_set(pairs2))))


    def forward(self, distances_X, distances_Z):
        """Return topological distance of two pairwise distance matrices.

        Args:
            distances1: Distance matrix in space 1
            distances2: Distance matrix in space 2

        Returns:
            distance, dict(additional outputs)
        """
        pairs_X = self._get_pairings(distances_X)
        pairs_Z = self._get_pairings(distances_Z)

        distance_components = {
            'metrics.matched_pairs_0D': self._count_matching_pairs(
                pairs_X, pairs_Z)
        }
        
        if self.match_edges is None:
            sig_X = self._select_distances_from_pairs(distances_X, pairs_X)
            sig_Z = self._select_distances_from_pairs(distances_Z, pairs_Z)
            distance = self.sig_error(sig_X, sig_Z)

        elif self.match_edges == 'symmetric':
            sig_X = self._select_distances_from_pairs(distances_X, pairs_X)
            sig_Z = self._select_distances_from_pairs(distances_Z, pairs_Z)
            # Selected pairs of 1 on distances of 2 and vice versa
            sigX_Z = self._select_distances_from_pairs(distances_Z, pairs_X)
            sigZ_X = self._select_distances_from_pairs(distances_X, pairs_Z)

            distanceX_Z = self.sig_error(sig_X, sigX_Z)
            distanceZ_X = self.sig_error(sig_Z, sigZ_X)

            distance_components['metrics.distanceX-Z'] = distanceX_Z
            distance_components['metrics.distanceZ-X'] = distanceZ_X

            distance = distanceX_Z + distanceZ_X

        elif self.match_edges == 'random':
            # Create random selection in oder to verify if what we are seeing
            # is the topological constraint or an implicit latent space prior
            # for compactness
            n_instances = len(pairs_X[0])
            pairs_X = torch.cat([
                torch.randperm(n_instances)[:, None],
                torch.randperm(n_instances)[:, None]
            ], dim=1)
            pairs_Z = torch.cat([
                torch.randperm(n_instances)[:, None],
                torch.randperm(n_instances)[:, None]
            ], dim=1)

            sigX_X = self._select_distances_from_pairs(
                distances_X, (pairs_X, None))
            sigX_Z = self._select_distances_from_pairs(
                distances_Z, (pairs_X, None))

            sigZ_Z = self._select_distances_from_pairs(
                distances_Z, (pairs_Z, None))
            sigZ_X = self._select_distances_from_pairs(
                distances_X, (pairs_Z, None))

            distanceX_Z = self.sig_error(sigX_X, sigX_Z)
            distanceZ_X = self.sig_error(sigZ_X, sigZ_Z)
            distance_components['metrics.distanceX-Z'] = distanceX_Z
            distance_components['metrics.distanceZ-X'] = distanceZ_X

            distance = distanceX_Z + distanceZ_X

        return distance, distance_components




class TopologicalSignatureDistance:
    """Topological signature."""

    def __init__(self, match_edges=None):
        """Topological signature computation.

        """
        super().__init__()

        self.match_edges = match_edges

        self.signature_calculator = PersistentHomologyCalculation()


    def _get_pairings(self, distances):
        #print(distances.shape)
        pairs = self.signature_calculator(
            distances.detach().cpu().numpy())
        #print(pairs.shape)
        return pairs


    def _select_distances_from_pairs(self, distance_matrix, pairs):
        
        selected_distances = distance_matrix[(pairs[:, 0], pairs[:, 1])]

        return selected_distances


    @staticmethod
    def sig_error(signature1, signature2):
        """Compute distance between two topological signatures."""
        #print(signature1.shape[0])
        return ((signature1 - signature2)**2).sum(dim=-1)


    @staticmethod
    def _count_matching_pairs(pairs1, pairs2):
        def to_set(array):
            return set(tuple(elements) for elements in array)
        return float(len(to_set(pairs1).intersection(to_set(pairs2))))


    def __call__(self, distances_X, distances_Z):
        """Return topological distance of two pairwise distance matrices.

        Args:
            distances1: Distance matrix in space 1
            distances2: Distance matrix in space 2

        Returns:
            distance, dict(additional outputs)
        """
        pairs_X = self._get_pairings(distances_X)
        pairs_Z = self._get_pairings(distances_Z)

        distance_components = {
            'metrics.matched_pairs_0D': self._count_matching_pairs(
                pairs_X, pairs_Z)
        }
        
        if self.match_edges is None:
            sig_X = self._select_distances_from_pairs(distances_X, pairs_X)
            sig_Z = self._select_distances_from_pairs(distances_Z, pairs_Z)
            distance = self.sig_error(sig_X, sig_Z)

        elif self.match_edges == 'symmetric':
            sig_X = self._select_distances_from_pairs(distances_X, pairs_X)
            sig_Z = self._select_distances_from_pairs(distances_Z, pairs_Z)
            # Selected pairs of 1 on distances of 2 and vice versa
            sigX_Z = self._select_distances_from_pairs(distances_Z, pairs_X)
            sigZ_X = self._select_distances_from_pairs(distances_X, pairs_Z)

            distanceX_Z = self.sig_error(sig_X, sigX_Z)
            distanceZ_X = self.sig_error(sig_Z, sigZ_X)

            distance_components['metrics.distanceX-Z'] = distanceX_Z
            distance_components['metrics.distanceZ-X'] = distanceZ_X

            distance = distanceX_Z + distanceZ_X

        elif self.match_edges == 'random':
            # Create random selection in oder to verify if what we are seeing
            # is the topological constraint or an implicit latent space prior
            # for compactness
            n_instances = len(pairs_X[0])
            pairs_X = torch.cat([
                torch.randperm(n_instances)[:, None],
                torch.randperm(n_instances)[:, None]
            ], dim=1)
            pairs_Z = torch.cat([
                torch.randperm(n_instances)[:, None],
                torch.randperm(n_instances)[:, None]
            ], dim=1)

            sigX_X = self._select_distances_from_pairs(
                distances_X, (pairs_X, None))
            sigX_Z = self._select_distances_from_pairs(
                distances_Z, (pairs_X, None))

            sigZ_Z = self._select_distances_from_pairs(
                distances_Z, (pairs_Z, None))
            sigZ_X = self._select_distances_from_pairs(
                distances_X, (pairs_Z, None))

            distanceX_Z = self.sig_error(sigX_X, sigX_Z)
            distanceZ_X = self.sig_error(sigZ_X, sigZ_Z)
            distance_components['metrics.distanceX-Z'] = distanceX_Z
            distance_components['metrics.distanceZ-X'] = distanceZ_X

            distance = distanceX_Z + distanceZ_X

        return distance, distance_components


"""
Topology - Topological
-------------------------------------------------------------------------------------------------------------------------------------------
"""
class Topological(LossTerm):

    def __init__(self, p: int):
        self.p = p
        self.topo_sig = TopologicalSignatureDistance()


    def __call__(self, X_batch: Tensor, Z_batch: Tensor, **tensors: Tensor) -> Tensor:

        X_batch_dists = torch.cdist(X_batch, X_batch, p = self.p)
        Z_batch_dists = torch.cdist(Z_batch, Z_batch, p = self.p)

        topo_error, topo_components = self.topo_sig(distances_X = X_batch_dists, distances_Z = Z_batch_dists)
        #print(topo_components)
        topo_error = topo_error / X_batch.shape[0]
        
        return topo_error