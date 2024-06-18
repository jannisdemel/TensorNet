# This file contains code inspired by a source distributed under the MIT License by Universitat Pompeu Fabra 2020-2023 (https://www.compscience.org)
# The original source code can be found at: https://github.com/torchmd/torchmd-net/blob/main/torchmdnet/models/tensornet.py
# The original source code is distributed under the MIT License. (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)


# Adapted by Jannis Demel, 2024


import torch
import torch.nn as nn
from torchmdnet.models.utils import OptimizedDistance

__all__=['TensorNet'] # if someone imports * from this module, they will only get TensorNet

class TensorNet(nn.Module):
    '''
    TensorNet's architecture. 
    From TensorNet: Cartesian Tensor Representations for Efficient Learning of Molecular Potentials; G. Simeon and G. de Fabritiis. NeurIPS 2023. (https://arxiv.org/abs/2306.06482)

    Parameters:
    -----------
    hidden_channels: int
        Number of hidden channels. Default is 128.
    num_layers: int
        Number of interaction layers. Default is 2.
    max_z: int
        Maximum atomic number used for embedding. Default is 128. Should be adapted to dataset to not create unnecessary embeddings.
    num_rbf: int
        Number of radial basis functions. Default is 50.
    cutoff_lower: float
        Lower cutoff distance for interatomic interactions. Default is 0.0.
    cutoff_upper: float
        Upper cutoff distance for interatomic interactions. Default is 4.5.
    max_num_neighbors: int
        Maximum number of neighbors to return for a given node/atom when constructing the molecular graph during forward passes. Default is 64. 
        Should be adapted to dataset since (if static_shapes is True) the model will always expect this number of neighbors and decrease performance.
    activation: str
        Activation function to use. Can be "silu", "tanh" or "sigmoid". Default is "silu".
    equivariance_invariance_group: str
        Group for equivariance and invariance. Can be "O(3)" or "SO(3)". Default is "O(3)".
    check_errors: bool
        Whether to check for errors in the distance module. Default is True.
    static_shapes: bool
        Whether to enforce static shapes. Default is True.
    trainable_rbf: bool
        Whether the radial basis functions are trainable. Default is True.
    dtype: str
        Data type of the tensors. Default is torch.float32.
    '''
    def __init__(self,
                 hidden_channels = 128,
                 num_layers = 2,
                 max_z = 128,
                 num_rbf = 50,
                 cutoff_lower = 0.0, 
                 cutoff_upper = 4.5, 
                 max_num_neighbors = 64, 
                 activation="silu",
                 equivariance_invariance_group="O(3)",
                 check_errors = True, 
                 static_shapes = True,
                 trainable_rbf = True,
                 dtype="torch.float32",):
        
        super(TensorNet, self).__init__()

        # define attributes
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.max_z = max_z
        self.num_rbf = num_rbf
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_num_neighbors = max_num_neighbors
        if activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Activation function {activation} not supported.")
        if equivariance_invariance_group in ['O(3)', 'SO(3)']:
            self.equivariance_invariance_group = equivariance_invariance_group
        else:
            raise ValueError(f"Equivariance group {equivariance_invariance_group} not supported.")
        self.check_errors = check_errors
        self.static_shapes = static_shapes
        self.trainable_rbf = trainable_rbf
        if dtype =="torch.float32":
            self.dtype = torch.float32
        elif dtype == "torch.float64":
            self.dtype = torch.float64
        else:
            raise ValueError(f"Data type {dtype} not supported.")


        # define the distance module that creates the graph out of the atomic positions by connecting atoms that are closer than a certain cutoff
        self.distance = OptimizedDistance(
            self.cutoff_lower,
            self.cutoff_upper,
            max_num_pairs=-self.max_num_neighbors,
            return_vecs=True,
            loop=True,
            check_errors=self.check_errors,
            resize_to_fit= not self.static_shapes,
            box=None, 
            long_edge_index=True,
        )
        # define the module for expanding the distances in terms of exponential radial basis functions
        self.distance_expansion = ExpNormalSmearing(self.cutoff_lower, self.cutoff_upper, self.num_rbf, self.trainable_rbf, self.dtype)


        # define the tensor embedding module
        self.tensor_embedding = TensorEmbedding(
            self.hidden_channels,
            self.num_rbf,
            self.activation,
            self.cutoff_lower,
            self.cutoff_upper,
            self.trainable_rbf,
            self.max_z,
            self.dtype)
        

        # define the interaction layers
        self.layers = nn.ModuleList()
        if self.num_layers > 0:
            self.layers.append(
                Interaction(
                    self.num_rbf,
                    self.hidden_channels,
                    self.activation,
                    self.cutoff_lower,
                    self.cutoff_upper,
                    self.equivariance_invariance_group,
                    self.dtype))

        # layer norm for scalar output
        self.outnorm = nn.LayerNorm(3 * self.hidden_channels, dtype=self.dtype)
        # Linear layer for scalar output
        self.Linear_out = nn.Linear(3 * self.hidden_channels, self.hidden_channels, dtype=self.dtype)

        # reset all parameters
        self.reset_parameters()

    def forward(self, batch_input):
        """
        Performs the forward pass.
        The input will be M molecules that are made up by N atoms.

        Parameters:
        -----------
        batch_input : 
        Object containing the input data. The object has the following attributes:
             - atom_ind (torch.tensor): Tensor of shape [N] containing the atomic numbers (not the real ones but starting from 0).
             - pos (torch.tensor): Tensor of shape [N,3] containing the atomic positions.
             - batch (torch.tensor): Tensor of shape [N] containing the batch indices, telling which atom belongs to which molecule. The batch indices are sorted in ascending order from 0 to M-1.
        
        Returns:
        --------
        torch.tensor: Tensor of shape [M] containing the scalar output for each molecule.
        """
        # extract from batch_input the atom indices, positions and batch indices
        z, pos, batch = batch_input.atom_ind, batch_input.pos.to(self.dtype), batch_input.batch
        # Obtain graph, with distances and relative position vectors
        # edge_index: tensor of shape (2,max_num_neighbors*N,M), where each column represents a pair of atoms that are connected by an edge. Nonexisting edges are represented by (-1,-1) pairs.
        # edge_weight: tensor of shape (max_num_neighbors*N,M), containing the distances between the atoms that are connected by an edge. Nonexisting edges are represented by 0
        # edge_vec: tensor of shape (max_num_neighbors*N,M, 3) containing the relative position vectors between the atoms that are connected by an edge. Nonexisting edges are represented by 0
        edge_index, edge_weight, edge_vec = self.distance(pos=pos, batch=batch)
        # make to the specified dtype
        edge_weight, edge_vec = edge_weight.to(self.dtype), edge_vec.to(self.dtype)
 
        # if we want static shapes, we can't delete the nonexisting edges. The idea is to make them pertain to a ghost atom, which is the new last atom in the batch
        if self.static_shapes:
            # create a mask that is True for the nonexisting edges
            mask = (edge_index[0] < 0).unsqueeze(0).expand_as(edge_index)
            # I trick the model into thinking that the masked edges pertain to the extra atom with index N=z.shape[0]
            # WARNING: This can hurt performance if max_num_pairs >> actual_num_pairs
            edge_index = edge_index.masked_fill(mask, z.shape[0])
            edge_weight = edge_weight.masked_fill(mask[0], 0)
            edge_vec = edge_vec.masked_fill(mask[0].unsqueeze(-1).expand_as(edge_vec), 0)
            # Since we now have N+1 atoms, we need to add the ghost atom to the atomic numbers and choose 0 
            z = torch.cat((z, torch.zeros(1, device=z.device, dtype=z.dtype)), dim=0)

        # expand distances in terms of exponential radial basis functions as described in formula (6) of the paper
        # edge_attr will add a dimension 1 with num_rbf elements to edge_weight
        edge_attr = self.distance_expansion(edge_weight)

        # we need to normalize the edge vectors. Just dividing them by their length will lead to NaNs since there are edges with length 0 (the ghost atom has self-loops)
        # therefore, all self-loops get a weight of 1 which allows dividing by the length
        mask_self_loops = edge_index[0] == edge_index[1]
        edge_vec = edge_vec / edge_weight.masked_fill(mask_self_loops, 1).unsqueeze(1)

        # now we get to the embedding part as descibed in the paper
        X = self.tensor_embedding(z, edge_index, edge_weight, edge_vec, edge_attr)
        
        # interaction and node update
        for layer in self.layers:
            X = layer(X, edge_index, edge_weight, edge_attr)
        
        # We now have the tensor representation for each atom and channel in X
        # This is used to get a scalar output as described in formula (12) in the paper
        # In principle, one can also get a vector or tensor output (described in the paper) but this is not implemented here
        # decompose tensors into irreducible representations
        I, A, S = decompose_tensor(X)
        x = torch.cat((tensor_norm(I), tensor_norm(A), tensor_norm(S)), dim=-1)
        x = self.outnorm(x)
        x = self.activation(self.Linear_out(x))
        # we can now remove the ghost atom
        if self.static_shapes:
            x = x[:-1]
        # sum over atomic contributions
        U_i = torch.zeros((batch.max()+1, x.shape[1]), dtype=x.dtype, device=x.device)
        U_i.index_add_(0, batch, x)
        # sum over channels
        U = torch.sum(U_i, dim=1) 
        return U, torch.zeros_like(batch_input.coeffs)
    
    def reset_parameters(self):
        """
        Resets all parameters of the model.
        """
        self.tensor_embedding.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.outnorm.reset_parameters()
        self.Linear_out.reset_parameters()

def vector_to_skewtensor(vectors):
    """
    This function converts a vector to a skew tensor.

    .. math::

        \\begin{bmatrix} v_1 \\\\ v_2 \\\\ v_3 \\end{bmatrix} \\rightarrow \\begin{bmatrix} 0 & -v_3 & v_2 \\\\ v_3 & 0 & -v_1 \\\\ -v_2 & v_1 & 0 \\end{bmatrix}
    
    Parameters:
    -----------
    vectors: torch.tensor
    Tensor of shape [N,3] representing N vectors.

    Returns:
    --------
    torch.tensor: Tensor of shape [N,3,3] representing N skew-symmetric matrices.
    """
    # get number of vectors N
    N = vectors.shape[0]
    # create zero tensor of shape [N]
    zero = torch.zeros(N, device=vectors.device, dtype=vectors.dtype)
    # create anti-symmetric matrix with via formula
    matrices = torch.stack(
        (   zero, -vectors[:, 2], vectors[:, 1],
            vectors[:, 2], zero, -vectors[:, 0],
            -vectors[:, 1], vectors[:, 0], zero, ),dim=1,)
    # matrices has shape [N,9], reshape to [N,3,3]
    matrices = matrices.view(-1, 3, 3)
    # if N=1, remove first dimension
    return matrices.squeeze(0)

def vector_to_symtensor(vector):
    """
    This function converts a vector into a symmetric (traceless) matrix by taking the outer product and then subtracting the trace.
    .. math::

        \\vec{v} = \\begin{pmatrix} v_x \\\\ v_y \\\\ v_z \\end{pmatrix} \\rightarrow S = \\vec{v}\\vec{v}^T - \\frac{1}{3}\\text{Tr}(\\vec{v}\\vec{v}^T)\\text{I} = \\begin{pmatrix} v_x^2 & v_xv_y & v_xv_z \\\\ v_xv_y & v_y^2 & v_yv_z \\\\ v_xv_z & v_yv_z & v_z^2 \\end{pmatrix} - \\frac{v^2}{3} \\text{I}

    Parameters:
    -----------
    vectors: torch.tensor
    Tensor of shape [N,3] representing N vectors.

    Returns:
    --------
    torch.tensor: Tensor of shape [N,3,3] representing N symmetric, traceless matrices.
    """
    # compute outer product
    tensor = torch.matmul(vector.unsqueeze(-1), vector.unsqueeze(-2))
    # compute trace and add dimensions to get shape [N,1,1]
    trace = torch.diagonal(tensor, dim1=-2, dim2=-1).mean(-1)[..., None, None]
    # create identity matrix
    I = torch.eye(3, device=tensor.device, dtype=tensor.dtype)
    # subtract trace and ensure symmetry of outer product
    S = 0.5 * (tensor + tensor.transpose(-2, -1)) - trace*I
    return S

def decompose_tensor(X):
    """
    Decomposes a tensor into its irreducible representations.

    .. math::

        X = \\underbrace{\\frac{1}{3}\\text{Tr}(X)\\text{I}}_{=I^X} + \\underbrace{\\frac{1}{2}(X-X^T)}_{=A^X} + \\underbrace{\\frac{1}{2}(X+X^T-\\frac{2}{3}\\text{Tr}(X)\\text{I})}_{=S^X}

    Parameters:
    -----------
    X : torch.tensor
    Tensor of shape [N,3,3] representing N 3x3 matrices.

    Returns:
    --------
    tuple: A tuple containing the l=0, l=1 and l=2 irreducible representations (I,A,S), where I,A,S all are tensors of shape [N,3,3].
    """
    # compute trace and add dimensions to get shape [N,1,1]
    I = (X.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[ ..., None, None] * torch.eye(3, 3, device=X.device, dtype=X.dtype)
    # compute antisymmetric part
    A = 0.5 * (X - X.transpose(-2, -1))
    # compute symmetric, traceless part
    S = 0.5 * (X + X.transpose(-2, -1)) - I
    return I, A, S

def tensor_norm(tensor):
    """
    Computes squared Tr(X^T X) for a tensor X.
    This is the squared Frobenisu norm but in the paper it is understood as the tensor norm ||X||.

    Parameters:
    -----------
    tensor: torch.tensor
        Input tensor
    
    Returns:
    --------
    torch.tensor: Tensor containing the squared tensor norm.
    """
    return (tensor**2).sum((-2, -1))

class ExpNormalSmearing(nn.Module):
    r'''
    Expansion in terms of radial basis functions as described in formula (6) of the paper.

    Parameters:
    -----------
    cutoff_lower: float
        Lower cutoff radius. Default is 0.0.
    cutoff_upper: float
        Upper cutoff radius. Default is 4.5.
    num_rbf: int
        Number of radial basis functions. Default is 50.
    trainable: bool
        Whether the means and betas are trainable. Default is True.
    dtype: torch.dtype
        Data type of the tensors. Default is torch.float32.
    '''
    def __init__(
        self,
        cutoff_lower=0.0,
        cutoff_upper=4.5,
        num_rbf=50,
        trainable=True,
        dtype=torch.float32):

        super(ExpNormalSmearing, self).__init__()

        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable
        self.dtype = dtype
        self.cutoff_fn = CosineCutoff(0, self.cutoff_upper)
        self.alpha = 5.0 / (cutoff_upper - cutoff_lower)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        '''
        Function to initialize the means and betas of the radial basis functions according to the default values in PhysNet (https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181)
        '''
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower, dtype=self.dtype))
        means = torch.linspace(start_value, 1, self.num_rbf, dtype=self.dtype)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf,dtype=self.dtype)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        '''
        Applies the radial basis functions to the distances. Notably, it also applies the cosine cutoff function, which is not shown in formula (6) of the paper.
        '''
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(-self.betas * (torch.exp(self.alpha * (-dist + self.cutoff_lower)) - self.means) ** 2)
    
class CosineCutoff(nn.Module):
    r'''
    Cosine cutoff function with cutoff radius :math:`r_c` as defined in the paper

    .. math::
        \\phi(r) = \\frac{1}{2} \\left( \\cos \\left( \\frac{\\pi r}{r_c} \\right) + 1 \\right) \\Theta(r_c - r)

    Additionally, the function can also have a lower cutoff radius.

    Parameters:
    -----------
    cutoff_lower: float
        Lower cutoff radius. Default is 0.0.
    cutoff_upper: float
        Upper cutoff radius. Default is 4.5.
    '''
    def __init__(self, cutoff_lower=0.0, cutoff_upper=4.5):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances):
        '''
        Applies the cosine cutoff function to the distances.

        Parameters:
        -----------
        distances: torch.tensor
            Tensor of shape [n] containing the distances.
        
        Returns:
        --------
        torch.tensor: Tensor of shape [n] containing cosine cutoff function applied to the distances.
        '''
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (torch.cos(torch.pi * ( 2 * (distances - self.cutoff_lower) / (self.cutoff_upper - self.cutoff_lower)+ 1.0))+ 1.0)
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper)
            cutoffs = cutoffs * (distances > self.cutoff_lower)
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * torch.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper)
            return cutoffs

class TensorEmbedding(nn.Module):
    """
    Tensor embedding layer. Will do the embeddig as describe din the paper.

    Parameters:
    -----------
    hidden_channels: int
        Number of hidden channels. 
    num_rbf: int
        Number of radial basis functions.
    activation: 
        Activation function to use.
    cutoff_lower: float
        Lower cutoff radius.
    cutoff_upper: float
        Upper cutoff radius.
    trainable_rbf: bool
        Whether the radial basis functions are trainable.
    max_z: int
        Maximum atomic number used for embedding.
    dtype: torch.dtype
        Data type of the tensors.

    Returns:
    --------
    torch.tensor: Tensor of shape [N+1,hidden_channels,3,3] containing the tensor embeddings for each atom and channel.
    """

    def __init__(
        self,
        hidden_channels,
        num_rbf,
        activation,
        cutoff_lower,
        cutoff_upper,
        trainable_rbf,
        max_z,
        dtype=torch.float32):

        super(TensorEmbedding, self).__init__()

        # define attributes
        self.hidden_channels = hidden_channels
        self.num_rbf = num_rbf
        self.activation = activation
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.trainable_rbf = trainable_rbf
        self.max_z = max_z
        self.dtype = dtype
        # define cosine cutoff function
        self.cutoff = CosineCutoff(self.cutoff_lower, self.cutoff_upper)
        # embedding for atomic numbers
        self.emb = nn.Embedding(num_embeddings=self.max_z, embedding_dim=self.hidden_channels)
        self.emb2 = nn.Linear(in_features=2*self.hidden_channels, out_features=hidden_channels, dtype=self.dtype)
        # Linear layersto produce the f as defined in fomrula (7)
        self.distance_proj1 = nn.Linear(in_features = num_rbf, out_features = hidden_channels, dtype=self.dtype)
        self.distance_proj2 = nn.Linear(in_features = num_rbf, out_features = hidden_channels, dtype=self.dtype)
        self.distance_proj3 = nn.Linear(in_features = num_rbf, out_features = hidden_channels, dtype=self.dtype)
        # LayerNorm for embedding
        self.init_norm = nn.LayerNorm(hidden_channels, dtype=self.dtype)
        # MLP for embedding as defined in formula (9)
        self.MLP1 = nn.ModuleList([
             nn.Linear(hidden_channels, 2 * hidden_channels, bias=True, dtype=self.dtype),
            nn.Linear( 2* hidden_channels, 3 * hidden_channels, bias=True, dtype=self.dtype)])
        # MLP for embedding as defined in formula (10)
        self.MLP2 = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels, dtype=self.dtype, bias=False) for _ in range(3)])
        # reset parameters
        self.reset_parameters()
     
    def forward(self, z, edge_index, edge_weight, edge_vec_norm, edge_attr):
        '''
        Forward pass of the tensor embedding layer.

        Parameters:
        -----------
        z: torch.tensor
            Tensor of shape [N+1] containing the atomic numbers. Note that the +1 is due to the ghost atom that is only added
            if static_shapes is True.
        edge_index: torch.tensor
            Tensor of shape [2,num_pairs] containing the indices of the atoms that are connected by an edge. The number
            of pairs is num_pairs = max_num_neighbors*N,M. 
        edge_weight: torch.tensor
            tensor of shape [num_pairs], containing the distances between the atoms that are connected by an edge.
            Nonexisting edges are represented by 0
        edge_vec_norm: torch.tensor
            Tensor of shape [num_pairs,3] containing the normalized relative position vectors between the atoms that are connected by an edge.
            Nonexisting edges are represented by 0
        edge_attr: torch.tensor
            Tensor of shape [num_pairs,num_rbf] containing the radial basis functions applied to the distances between the atoms that are connected by an edge.
            Nonexisting edges are represented by 0
        '''
        # get for each edge an embedding of the atomic numbers belonging to that edge
        # Z_ij will have shape [num_pairs,hidden_channels,1,1]
        Z_ij = self._get_atomic_number_message(z, edge_index)
        # get the tensor messages as described in (7) and (8)
        I_ij, A_ij, S_ij = self._get_tensor_messages(Z_ij, edge_weight, edge_vec_norm, edge_attr)
        # get atom-wise tensor representations by aggregating (summing) all neighboring edge-wise features. 
        source = torch.zeros(z.shape[0], self.hidden_channels, 3, 3, device=z.device, dtype=I_ij.dtype)
        I = source.index_add(dim=0, index=edge_index[0], source=I_ij)
        A = source.index_add(dim=0, index=edge_index[0], source=A_ij)
        S = source.index_add(dim=0, index=edge_index[0], source=S_ij)
        # now we implement formula (9)
        norm = self.init_norm(tensor_norm(I+A+S))
        for layer in self.MLP1:
            norm = self.activation(layer(norm))
        # we reshape to 3 scalars per channel, this are the f_I, f_A and f_S as described in formula (9)
        norm = norm.reshape(-1,self.hidden_channels,3)
        # now we implement formula (10)
        # One should note that the linear layer is applied to the last dimension of I, A and S, but we want it applied
        # to the first dimension (which iterates over atoms). Therefore, we permute the dimensions and then permute them back
        I = self.MLP2[0](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) * norm[..., 0, None, None]
        A = self.MLP2[1](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) * norm[..., 0, None, None]
        S = self.MLP2[2](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) * norm[..., 0, None, None]
        # finally sum up to get full rank 2 tensor per atom and channel
        X = I + A + S
        # X will have shape [N+1,hidden_channels,3,3] and defines aa rank 2 tenspr for each node (atom) and channel
        return X

    def _get_atomic_number_message(self, z, edge_index):
        '''
        Get the atomic number message as described in the paper.

        Parameters:
        -----------
        z: torch.tensor
            Tensor of shape [N+1] containing the atomic numbers. Note that the +1 is due to the ghost atom that is only added
            if static_shapes is True.
        edge_index: torch.tensor
            Tensor of shape [2,num_pairs] containing the indices of the atoms that are connected by an edge. The number
            of pairs is num_pairs = max_num_neighbors*N,M.

        Returns:
        --------
        torch.tensor: Tensor of shape [num_pairs,hidden_channels,1,1] containing the atomic number message.
        '''
        # create an embedding of the atomic numbers 
        Z = self.emb(z)
        # for every edge (ij) we map with a linear layer the concatenation of Z_i and Z_j to n pair-wise invariant representations Z_ij 
        Zij = self.emb2(Z.index_select(0, edge_index.t().reshape(-1)).view(-1, self.hidden_channels * 2 ))[..., None, None]
        return Zij

    def _get_tensor_messages(self, Z_ij, edge_weight, edge_vec_norm, edge_attr):
        '''
        Implements formula (7) in the paper but already multiplies with the cutoff function and the atomic number 
        (in the paper this is only done in formula 8)

        Parameters:
        -----------
        Z_ij: torch.tensor
            Tensor of shape [num_pairs,hidden_channels,1,1] containing the atomic number message.
        edge_weight: torch.tensor
            tensor of shape [num_pairs], containing the distances between the atoms that are connected by an edge.
            Nonexisting edges are represented by 0
        edge_vec_norm: torch.tensor
            Tensor of shape [num_pairs,3] containing the normalized relative position vectors between the atoms that are connected by an edge.
            Nonexisting edges are represented by 0
        edge_attr: torch.tensor
            Tensor of shape [num_pairs,num_rbf] containing the radial basis functions applied to the distances between the atoms that are connected by an edge.
            Nonexisting edges are represented by 0
        
        Returns:
        --------
        tuple: A tuple of 3 tensors (I_ij, A_ij, S_ij) containing the irreducible representations of the tensor messages (the summands in formula (8))
        '''
        # cutoff function times the atomic number message to multiply 
        C = self.cutoff(edge_weight).reshape(-1, 1, 1, 1) * Z_ij
        # initializes I_0^{ij} as the identity matrix
        eye = torch.eye(3, 3, device=edge_vec_norm.device, dtype=edge_vec_norm.dtype)[None, None, ...]
        # calcualtes f_I^0 I_0^{ij} times C
        I_ij = self.distance_proj1(edge_attr)[..., None, None] * eye * C
        # calculates f_A^0 A_0^{ij} times C
        A_ij = self.distance_proj2(edge_attr)[...,None, None] * vector_to_skewtensor(edge_vec_norm)[..., None, :, :] * C
        # calculates f_S^0 S_0^{ij} times C
        S_ij = self.distance_proj3(edge_attr)[..., None, None] * vector_to_symtensor(edge_vec_norm)[..., None, :, :] * C

        return I_ij, A_ij, S_ij

    def reset_parameters(self):
        '''
        Function that resets all parameters.
        '''
        self.emb.reset_parameters()
        self.emb2.reset_parameters()
        self.distance_proj1.reset_parameters()
        self.distance_proj2.reset_parameters()
        self.distance_proj3.reset_parameters()
        self.init_norm.reset_parameters()
        for layer in self.MLP1:
            layer.reset_parameters()
        for layer in self.MLP2:
            layer.reset_parameters()

class Interaction(nn.Module):
    '''
    Interaction Layer

    Parameters:
    -----------
    num_rbf: int
        Number of radial basis functions.
    hidden_channels: int
        Number of hidden channels.
    activation:
        Activation function to use.
    '''
    def __init__(self, num_rbf, hidden_channels, activation, cutoff_lower, cutoff_upper, equivariance_invariance_group , dtype):
        super(Interaction, self).__init__()
        # define attributes
        self.num_rbf = num_rbf
        self.hidden_channels = hidden_channels
        self.activation = activation
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)
        self.equivariance_invariance_group = equivariance_invariance_group
        self.dtype = dtype

        # define Linear layers for constructing Y in interaction step
        self.Linears1 = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels, dtype=self.dtype) for _ in range(3)])

        # define two linear layers for transforming embedded distances
        self.Linears2 = nn.ModuleList([
             nn.Linear(self.num_rbf,  self.hidden_channels, bias=True, dtype=self.dtype),
             nn.Linear(self.hidden_channels, 2 * self.hidden_channels, bias=True, dtype=self.dtype),
             nn.Linear( 2* self.hidden_channels, 3 * self.hidden_channels, bias=True, dtype=self.dtype)])

        # define linear layers for constructing Y after message passing step
        self.Linears3 = nn.ModuleList([nn.Linear(self.hidden_channels, self.hidden_channels, dtype=self.dtype) for _ in range(3)])

        # reset parameters
        self.reset_parameters()
        

    def forward(self, X, edge_index, edge_weight, edge_attr):
        '''
        Forward pass of the interaction layer.
        
        Parameters:
        -----------
        X: torch.tensor
            Tensor of shape [N+1,hidden_channels,3,3] containing the tensor embeddings for each atom and channel.
        edge_index: torch.tensor
            Tensor of shape [2,num_pairs] containing the indices of the atoms that are connected by an edge. The number
            of pairs is num_pairs = max_num_neighbors*N,M.
        edge_weight: torch.tensor
            tensor of shape [num_pairs], containing the distances between the atoms that are connected by an edge.
            Nonexisting edges are represented by 0
        edge_attr: torch.tensor
            Tensor of shape [num_pairs,num_rbf] containing the radial basis functions applied to the distances between the atoms that are connected by an edge.
            Nonexisting edges are represented by 0

        Returns:
        --------
        torch.tensor: Tensor of shape [N+1,hidden_channels,3,3] containing the tensor after interaction and node update
        '''
        # normalize each nodes tensor representation
        X = X/(tensor_norm(X)+1)[..., None, None]
        # decompose
        I, A, S = decompose_tensor(X)
        # linear layer
        I = self.Linears1[0](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        A = self.Linears1[1](A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        S = self.Linears1[2](S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # compute Y^i
        Y = I + A + S
        # create scalars by putting embedded distances in MLP
        for layer in self.Linears2:
            edge_attr = self.activation(layer(edge_attr))
        # calculate cosine cutoff of the distances
        C = self.cutoff(edge_weight)
        # get three scalars via formula (11)
        edge_attr = (edge_attr * C.view(-1,1)).reshape(edge_attr.shape[0], self.hidden_channels, 3)
        # do message passing for l=0,l=1,l=2 individually
        msg_I = self.tensor_message_passing(edge_index, edge_attr[..., 0, None, None], I)
        msg_A = self.tensor_message_passing(edge_index, edge_attr[..., 1, None, None], A)
        msg_S = self.tensor_message_passing(edge_index, edge_attr[..., 2, None, None], S)
        msg = msg_I + msg_A + msg_S
        # now we have to separate if we look at equivariance w.r.t O(3) or SO(3)
        if self.equivariance_invariance_group == 'O(3)':
            # in this case we have to make sure that that the new features have the same parity as before
            new_features = torch.matmul(msg,Y)+torch.matmul(Y,msg)
        elif self.equivariance_invariance_group == 'SO(3)':
            new_features = 2*torch.matmul(msg,Y)
        # decompose new features
        I, A, S = decompose_tensor(new_features)
        # normalize
        norm = (tensor_norm(I+A+S)+1)[..., None, None]
        I, A, S = I / norm, A / norm, S / norm
        # another linear layer
        I = self.Linears3[0](I.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        A = self.Linears3[1](A.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        S = self.Linears3[2](S.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # residual update
        dX = I + A + S
        # update X with parity preserving polynomial
        X = X + dX + torch.matrix_power(dX,2)
        return X
    def reset_parameters(self):
        '''
        Function that resets all parameters.
        '''
        for layer in self.Linears1:
            layer.reset_parameters()
        for layer in self.Linears2:
            layer.reset_parameters()
        for layer in self.Linears3:
            layer.reset_parameters()

    def tensor_message_passing(self, edge_index, factor, tensor):
        '''
        Message passing for tensors 
        
        Parameters:
        -----------
        edge_index: torch.tensor
            Tensor of shape [2,num_pairs] containing the indices of the atoms that are connected by an edge. 
        factor: torch.tensor
            scalar factor to multiply the tensors on the nodes to get M^ij, called f_I,f_A or f_S in the paper (paragraph after formula 11)
        tensor: torch.tensor
            Tensor of shape [N+1,hidden_channels,3,3] containing the tensor embeddings for each atom and channel. Called Y^i in the paper.
       
        Returns:
        --------
        torch.tensor: Tensor of shape [N+1,hidden_channels,3,3] containing the tensor messages for each node.
          '''
        # get messages
        message = factor * tensor.index_select(0, edge_index[1])
        # aggregate messages
        tensor_m = torch.zeros(tensor.shape, device=tensor.device, dtype=tensor.dtype)
        tensor_m = tensor_m.index_add(dim=0, index=edge_index[0], source=message)
        return tensor_m
