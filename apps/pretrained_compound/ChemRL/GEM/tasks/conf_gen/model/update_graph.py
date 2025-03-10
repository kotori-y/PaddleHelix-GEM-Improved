import paddle

try:
    from apps.pretrained_compound.ChemRL.GEM.tasks.conf_gen.model.utils import get_bond_length, get_bond_angle, \
        get_dihedral_angle
    from apps.pretrained_compound.ChemRL.GEM.tasks.conf_gen.utils import scatter_mean
except:
    from conf_gen.model.utils import get_bond_length, get_bond_angle, get_dihedral_angle
    from conf_gen.utils import scatter_mean

import copy


def move2origin(poses, batch, num_nodes):
    dim_size = batch.max() + 1
    index = paddle.to_tensor(batch)
    poses_mean = scatter_mean(poses, index, 0, dim_size)
    _poses_mean = poses_mean.numpy().repeat(num_nodes, axis=0)
    _poses_mean = paddle.to_tensor(_poses_mean, dtype=poses_mean.dtype)
    return poses - _poses_mean, poses_mean


def update_atom_bond_graph(atom_bond_graph, feed_dict, new_positions):
    new_atom_bond_graph = copy.deepcopy(atom_bond_graph)

    bond_length, masked_bond_length = get_bond_length(new_positions, feed_dict)

    new_atom_bond_graph.edge_feat['bond_length'] = bond_length.squeeze()
    new_atom_bond_graph.edge_feat['bond_length'] = new_atom_bond_graph.edge_feat['bond_length'].detach()

    return new_atom_bond_graph, masked_bond_length, bond_length


def update_bond_angel_graph(bond_angel_graph, atom_bond_graph, feed_dict, new_positions):
    new_bond_angel_graph = copy.deepcopy(bond_angel_graph)

    bond_angle, masked_bond_angle = get_bond_angle(new_positions, feed_dict)

    angle_atoms = atom_bond_graph.edges.gather(new_bond_angel_graph.edges.flatten()).reshape((-1, 4))
    mask = (angle_atoms[:, 0] == angle_atoms[:, 1]) | \
           (angle_atoms[:, 0] == angle_atoms[:, -1]) | \
           (angle_atoms[:, 1] == angle_atoms[:, -1])

    new_bond_angel_graph.edge_feat['bond_angle'][~mask] = bond_angle.squeeze()
    new_bond_angel_graph.edge_feat['bond_angle'][mask] = 0

    new_bond_angel_graph.edge_feat['bond_angle'] = new_bond_angel_graph.edge_feat['bond_angle'].detach()

    return new_bond_angel_graph, masked_bond_angle, bond_angle


def update_angle_dihedral_graph(angle_dihedral_graph, bond_angel_graph, atom_bond_graph, feed_dict, new_positions):
    new_angle_dihedral_graph = copy.deepcopy(angle_dihedral_graph)

    dihedral_angle, masked_dihedral_angle = get_dihedral_angle(new_positions, feed_dict)

    # ultra_edges为所有的组成二面角的键角
    # super_edges为所有的组成键角的化学键
    ultra_edges = angle_dihedral_graph.edges
    super_edges = bond_angel_graph.edges
    edges = atom_bond_graph.edges

    # 过滤出首尾为同一化学键的可能
    head_edges = super_edges.gather(ultra_edges[:, 0])[:, 0]
    tail_edges = super_edges.gather(ultra_edges[:, 1])[:, -1]

    head_edge_atoms = edges.gather(head_edges)
    tail_edge_atoms = edges.gather(tail_edges)
    _edges = paddle.concat([head_edge_atoms, tail_edge_atoms], axis=1)

    mask_1 = (_edges[:, 0] == _edges[:, 1]) | (_edges[:, 0] == _edges[:, 2]) | (_edges[:, 0] == _edges[:, 3])
    mask_2 = (_edges[:, 1] == _edges[:, 2]) | (_edges[:, 1] == _edges[:, 3])
    mask_3 = (_edges[:, 2] == _edges[:, 3])

    # ring_edges = edges[atom_bond_graph.edge_feat['is_in_ring'] != 1]
    # med_edges = _edges[:, 1:3]
    # ring_mask = paddle.concat(list(map(lambda x: (ring_edges == x).all(axis=1).any(), med_edges)))

    mask = mask_1 | mask_2 | mask_3

    new_angle_dihedral_graph.edge_feat["dihedral_angle"][~mask] = dihedral_angle.squeeze()
    new_angle_dihedral_graph.edge_feat["dihedral_angle"][mask] = 0

    new_angle_dihedral_graph.edge_feat['dihedral_angle'] = new_angle_dihedral_graph.edge_feat['dihedral_angle'].detach()

    return new_angle_dihedral_graph, masked_dihedral_angle, dihedral_angle


def updated_graph(graph, feed_dict, now_positions, delta_positions, update_target, batch, num_nodes, move=True):
    new_graph = copy.deepcopy(graph)

    if move:
        new_positions, _ = move2origin(now_positions + delta_positions, batch, num_nodes)
    else:
        new_positions = now_positions + delta_positions

    atom_bond_graph = new_graph['atom_bond_graph']
    bond_angel_graph = new_graph['bond_angle_graph']
    angle_dihedral_graph = new_graph["angle_dihedral_graph"]
    # new_positions, _ = move2origin(new_positions, batch, num_nodes)

    if update_target == "bond_length":
        new_atom_bond_graph, masked_bond_length, bond_length = update_atom_bond_graph(
            atom_bond_graph,
            feed_dict,
            new_positions
        )

        new_graph['atom_bond_graph'] = new_atom_bond_graph
        return new_graph, new_positions, masked_bond_length, []

    if update_target == "bond_angle":
        new_atom_bond_graph, masked_bond_length, bond_length = update_atom_bond_graph(
            atom_bond_graph,
            feed_dict,
            new_positions
        )
        new_bond_angel_graph, masked_bond_angle, bond_angle = update_bond_angel_graph(
            bond_angel_graph,
            atom_bond_graph,
            feed_dict,
            new_positions
        )

        new_graph['atom_bond_graph'] = new_atom_bond_graph
        new_graph['bond_angle_graph'] = new_bond_angel_graph
        return new_graph, new_positions, masked_bond_angle, [masked_bond_length]

    if update_target == "dihedral_angle":
        new_atom_bond_graph, masked_bond_length, bond_length = update_atom_bond_graph(
            atom_bond_graph,
            feed_dict,
            new_positions
        )
        new_bond_angel_graph, masked_bond_angle, bond_angle = update_bond_angel_graph(
            bond_angel_graph,
            atom_bond_graph,
            feed_dict,
            new_positions
        )
        new_angle_dihedral_graph, masked_dihedral_angle, dihedral_angle = update_angle_dihedral_graph(
            angle_dihedral_graph,
            bond_angel_graph,
            atom_bond_graph,
            feed_dict,
            new_positions
        )

        new_graph['atom_bond_graph'] = new_atom_bond_graph
        new_graph['bond_angle_graph'] = new_bond_angel_graph
        new_graph["angle_dihedral_graph"] = new_angle_dihedral_graph
        return new_graph, new_positions, masked_dihedral_angle, [masked_bond_length, masked_bond_angle]
