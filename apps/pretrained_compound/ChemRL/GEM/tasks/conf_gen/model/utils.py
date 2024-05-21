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

    bond_length = paddle.norm(position_i - position_j, p=2, axis=1).unsqueeze(1)
    return bond_length


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

    bond_angle = _get_angle(position_j - position_i, position_j - position_k)
    return bond_angle


def get_dihedral_angle(positions, feed_dict):
    position_a = paddle.gather(positions, feed_dict['Adi_node_a'])
    position_b = paddle.gather(positions, feed_dict['Adi_node_b'])
    position_c = paddle.gather(positions, feed_dict['Adi_node_c'])
    position_d = paddle.gather(positions, feed_dict['Adi_node_d'])

    rAB = position_b - position_a
    rBC = position_c - position_b
    rCD = position_d - position_c

    nABC = paddle.cross(rAB, rBC)
    # nABCSqLength = paddle.sum(nABC * nABC, axis=1)

    nBCD = paddle.cross(rBC, rCD)
    # nBCDSqLength = paddle.sum(nBCD * nBCD, axis=1)

    # m = paddle.cross(nABC, rBC)

    # angles_1 = -paddle.atan2(
    #     paddle.sum(m * nBCD, axis=1) / (paddle.sqrt(nBCDSqLength * paddle.sum(m * m, axis=1)) + 1e-4),
    #     paddle.sum(nABC * nBCD, axis=1) / (paddle.sqrt(nABCSqLength * nBCDSqLength) + 1e-4)
    # )

    b = paddle.cross(nABC, nBCD)

    tmp = paddle.sum(nABC * nBCD, axis=1) / (paddle.linalg.norm(nABC, axis=1) * paddle.linalg.norm(nBCD, axis=1))
    unsigned_rad = acos_safe(tmp)

    unsigned_rad = paddle.where(paddle.isnan(unsigned_rad), paddle.zeros(unsigned_rad.shape), unsigned_rad)

    return paddle.where(paddle.sum(rBC * b, axis=1) > 0, unsigned_rad, -unsigned_rad)
    # return unsigned_rad
