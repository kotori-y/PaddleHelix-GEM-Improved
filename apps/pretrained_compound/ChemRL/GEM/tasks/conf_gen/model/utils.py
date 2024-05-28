import paddle


def acos_safe(x, eps=1e-4):
    slope = paddle.acos(paddle.to_tensor(1 - eps)) / eps
    # TOD0: stop doing this allocation once sparse gradients with NaNs (like in# th.where)are handled differently.
    buf = paddle.empty_like(x)
    good = paddle.abs(x) <= 1-eps
    bad = ~good
    sign = paddle.sign(x[bad])
    buf[good] = paddle.acos(x[good])
    if bad.any():
        buf[bad] = paddle.acos(sign * (1-eps)) - slope * sign * (paddle.abs(x[bad]) - 1 + eps)
    return buf


def get_bond_length(positions, feed_dict):
    position_i = paddle.gather(positions, feed_dict['Bl_node_i'])
    position_j = paddle.gather(positions, feed_dict['Bl_node_j'])

    masked_position_i = paddle.gather(positions, feed_dict['masked_Bl_node_i'])
    masked_position_j = paddle.gather(positions, feed_dict['masked_Bl_node_j'])

    bond_length = paddle.norm(position_i - position_j, p=2, axis=1).unsqueeze(1)
    masked_bond_length = paddle.norm(masked_position_i - masked_position_j, p=2, axis=1).unsqueeze(1)

    return bond_length, masked_bond_length


def get_bond_angle(positions, feed_dict):
    def _get_angle(vec1, vec2):
        norm1 = paddle.norm(vec1, p=2, axis=1)
        norm2 = paddle.norm(vec2, p=2, axis=1)

        mask = (norm1.unsqueeze(axis=1) == 0) | (norm2.unsqueeze(axis=1) == 0)

        vec1 = vec1 / (norm1.unsqueeze(axis=1) + 1e-5)  # 1e-5: prevent numerical errors
        vec2 = vec2 / (norm2.unsqueeze(axis=1) + 1e-5)

        angle = paddle.acos(paddle.dot(vec1, vec2))
        # angle = paddle.acos(paddle.dot(vec1, vec2).squeeze(1))
        angle[mask] = 0

        return angle

    position_i = paddle.gather(positions, feed_dict['Ba_node_i'])
    position_j = paddle.gather(positions, feed_dict['Ba_node_j'])
    position_k = paddle.gather(positions, feed_dict['Ba_node_k'])

    masked_position_i = paddle.gather(positions, feed_dict['masked_Ba_node_i'])
    masked_position_j = paddle.gather(positions, feed_dict['masked_Ba_node_j'])
    masked_position_k = paddle.gather(positions, feed_dict['masked_Ba_node_k'])

    bond_angle = _get_angle(position_j - position_i, position_j - position_k)
    masked_bond_angle = _get_angle(masked_position_j - masked_position_i, masked_position_j - masked_position_k)

    return bond_angle, masked_bond_angle


def get_dihedral_angle(positions, feed_dict):
    def _get_dihedral_angle(vec1, vec2, vec3):
        nABC = paddle.cross(vec1, vec2)
        nBCD = paddle.cross(vec2, vec3)

        b = paddle.cross(nABC, nBCD)

        tmp = paddle.sum(nABC * nBCD, axis=1) / (paddle.linalg.norm(nABC, axis=1) * paddle.linalg.norm(nBCD, axis=1))
        unsigned_rad = acos_safe(tmp)

        unsigned_rad = paddle.where(paddle.isnan(unsigned_rad), paddle.zeros(unsigned_rad.shape), unsigned_rad)

        return paddle.where(paddle.sum(vec2 * b, axis=1) > 0, unsigned_rad, -unsigned_rad)

    position_a = paddle.gather(positions, feed_dict["Adi_node_a"])
    position_b = paddle.gather(positions, feed_dict["Adi_node_b"])
    position_c = paddle.gather(positions, feed_dict["Adi_node_c"])
    position_d = paddle.gather(positions, feed_dict["Adi_node_d"])

    masked_position_a = paddle.gather(positions, feed_dict["masked_Adi_node_a"])
    masked_position_b = paddle.gather(positions, feed_dict["masked_Adi_node_b"])
    masked_position_c = paddle.gather(positions, feed_dict["masked_Adi_node_c"])
    masked_position_d = paddle.gather(positions, feed_dict["masked_Adi_node_d"])

    dihedral_angle = _get_dihedral_angle(
        vec1=position_b - position_a,
        vec2=position_c - position_b,
        vec3=position_d - position_c
    )

    masked_dihedral_angle = _get_dihedral_angle(
        vec1=masked_position_b - masked_position_a,
        vec2=masked_position_c - masked_position_b,
        vec3=masked_position_d - masked_position_c
    )

    return dihedral_angle, masked_dihedral_angle
