import tvm
import math
import numpy as np
from functools import reduce
from itertools import permutations, product
from utils import (assert_print, gen_enum, any_factor_split, get_factor_lst, gen_group,
                              is_power_of_x)


def able_inline(op, down_graph):
    is_compute = isinstance(op, tvm.tensor.ComputeOp)
    has_reduce = hasattr(op, "reduce_axis") and op.reduce_axis
    is_output = False
    for i in range(op.num_outputs):
        if op.output(i) not in down_graph:
            is_output = True
            break
    return is_compute and (not has_reduce) and (not is_output)


class Space(object):
    def __init__(self):
        self.subspaces = {}
        self.types = {}
        self.valid_type_keys = [
            "fuse", "spatial", "reduce", "reorder", "inline", "unroll", "merge", "special", "intrin"]
        for type_key in self.valid_type_keys:
            self.types[type_key] = []
        self.dim = 0

    def add_subspace(self, name, subspace, type_key, override=False):
        if name in self.subspaces and not override:
            raise RuntimeError("Same subspace name")
        assert_print(type_key in self.valid_type_keys)
        self.subspaces[name] = subspace
        self.types[type_key].append(name)
        self.dim += subspace.dim

    def items(self):
        return self.subspaces.items()

    def __len__(self):
        ret = 1
        for _, subspace in self.subspaces.items():
            ret *= len(subspace)
        return ret

    def length(self):
        ret = {}
        total = 1
        added = 0
        for name, subspace in self.subspaces.items():
            ret[name] = len(subspace)
            total *= ret[name]
            added += ret[name]
        ret["total"] = total
        ret["added"] = added
        return ret


DirectedSubSpaceTypeKeys = ["spatial", "reduce"]
UndirectedSubSpaceTypeKeys = ["fuse", "reorder", "unroll", "inline", "merge", "special"]


class SubSpace(object):
    def __init__(self):
        self.dim = 0
        self.static_entities = []
        self.size = 0
        self.num_direction = 0

    def random_entity(self):
        return np.random.choice(self.static_entities)

    def next_entity(self, *args, **kwargs):
        raise NotImplementedError()

    def get_entity(self, p):
        return self.static_entities[p]

    def get_direction(self, num):
        raise NotImplementedError()

    def __len__(self):
        return self.size


class SplitSpace(SubSpace):
    def __init__(self, dim, total, allow_non_divisible='off'):
        super(SplitSpace, self).__init__()
        self.total = total
        self.allow_non_divisible = allow_non_divisible
        self.dim = dim
        self.static_entities = any_factor_split(total, dim, allow_non_divisible=allow_non_divisible)
        # print(self.static_entities)
        self.size = len(self.static_entities)
        self.num_direction = dim * (dim - 1)
        self.directions = []
        for i in range(self.dim):
            for j in range(self.dim):
                if i != j:
                    self.directions.append((i, j))
        self.type_key = "split"

    def next_entity(self, pos, d):
        # d is tuple
        if len(d) == 1:
            next_pos = (pos + d[0]) % self.size
            return next_pos
        elif len(d) == 2:
            asc_pos, dec_pos = d[0], d[1]
            assert_print(0 <= asc_pos < self.dim)
            assert_print(0 <= dec_pos < self.dim)
            assert_print(asc_pos != dec_pos)
            current = self.static_entities[pos]
            ret = current.copy()
            left = current[asc_pos] * current[dec_pos]
            canout = False
            next_pos = -1
            while not canout:
                tmp = ret[asc_pos] + 1
                while tmp <= left:
                    if self.allow_non_divisible == 'continuous':
                        break
                    elif self.allow_non_divisible == 'power2' and is_power_of_x(2, tmp):
                        break
                    elif left % tmp == 0:
                        break
                    tmp += 1
                tmp = min(tmp, left)
                ret[asc_pos] = tmp
                ret[dec_pos] = math.ceil(left / tmp)
                try:
                    next_pos = self.static_entities.index(ret)
                    canout = True
                except ValueError:
                    canout = False
            return next_pos
        else:
            raise RuntimeError(
                "Not support for direction more than two dims: {}".format(d))

    def get_direction(self, num):
        return self.directions[num % self.num_direction]


class IntrinSplitSpace(SubSpace):
    def gen_pow2_factors(self, dim):
        DIM = 4  # TODO
        dim = max(dim, DIM)
        low, high = int(math.log2(DIM)), math.ceil(math.log2(dim)) + 1
        res = map(lambda i: 2 ** i, range(low, high))
        return [[int(math.ceil(dim / x)), x] for x in res]

    def __init__(self, length):
        super(IntrinSplitSpace, self).__init__()
        self.dim = 2
        self.static_entities = self.gen_pow2_factors(length)
        self.num_direction = 2
        self.directions = [(1, 0), (0, 1)]
        self.size = len(self.static_entities)
        self.type_key = "split"

    def next_entity(self, pos, d):
        if len(d) == 1:
            next_pos = (pos + d[0]) % self.size
            return next_pos
        else:
            next_pos = max(0, min(self.size - 1, pos + d[0] - d[1]))
            return next_pos

    def get_direction(self, num):
        return self.directions[num % self.num_direction]


class FuseSpace(SubSpace):
    def __init__(self, dim, elements):
        self.dim = dim
        self.static_entities = gen_group(elements, most_groups=self.dim)
        self.size = len(self.static_entities)
        self.num_direction = 2
        self.directions = [(-1,), (1,)]
        self.type_key = "fuse"

    def next_entity(self, pos, d):
        # d is tuple
        if len(d) == 1:
            pos = (pos + d[0]) % self.size
            return pos
        else:
            raise RuntimeError(
                "Not support for direction more than one dim: {}".format(d))

    def get_direction(self, num):
        return self.directions[num % self.num_direction]


class ReorderSpace(SubSpace):
    def __init__(self, num_spatial_axis):
        self.dim = 1
        self.static_entities = [[i] for i in range(num_spatial_axis)]
        self.size = len(self.static_entities)
        print(num_spatial_axis, self.static_entities, self.size)
        self.num_direction = 2
        self.directions = [(-1,), (1,)]
        self.type_key = "reorder"

    def next_entity(self, pos, d):
        # d is tuple
        if len(d) == 1:
            pos = (pos + d[0]) % self.size
            return pos
        else:
            raise RuntimeError(
                "Not support for direction more than one dim: {}".format(d))

    def get_direction(self, num):
        return self.directions[num % self.num_direction]


class UnrollSpace(SubSpace):
    def __init__(self, total, slevel, explicit=False):
        super(UnrollSpace, self).__init__()
        self.dim = 2
        self.static_entities = []
        # self.steps = steps
        # explicits = [1] if explicit else [0, 1]
        # for step in steps:
        #     for _explicit in explicits:
        #         self.static_entities.append([step, _explicit])
        if slevel == 2:
            for i in total:
                for j in total:
                    if i != j:
                        self.static_entities.append([i, j])
        elif slevel == 3:
            for i in total:
                for j in total:
                    for k in total:
                        self.static_entities.append([i, j, k])
        # self.static_entities = [[0, 1], [1, 0], [0, 2], [2, 0], [0, 3], [3, 0], [1, 2], [2, 1], [1, 3], [3, 1], [2, 3],
        #                         [3, 2]]
        self.size = len(self.static_entities)
        self.num_direction = 2
        self.directions = [(-1,), (1,)]
        self.type_key = "unroll"

    def next_entity(self, pos, d):
        # d is tuple
        if len(d) == 1:
            pos = (pos + d[0]) % self.size
            return pos
        else:
            raise RuntimeError(
                "Not support for direction more than one dim: {}".format(d))

    def get_direction(self, num):
        return self.directions[num % self.num_direction]


class PosSpace(SubSpace):
    def __init__(self, parts, num_axis):
        self.dim = 2
        self.static_entities = []
        self.parts = parts
        self.num_axis = num_axis
        for i in range(parts):
            for j in range(num_axis):
                self.static_entities.append([i, j])
        self.size = len(self.static_entities)
        self.num_direction = 2
        self.directions = [(-1,), (1,)]
        self.type_key = "local"

    def next_entity(self, pos, d):
        # d is tuple
        if len(d) == 1:
            pos = (pos + d[0]) % self.size
            return pos
        else:
            raise RuntimeError(
                "Not support for direction more than one dim: {}".format(d))

    def get_direction(self, num):
        return self.directions[num % self.num_direction]


class InlineSpace(SubSpace):
    def __init__(self, inline_op_pos, op_num, force_inline=False):
        self.dim = op_num
        self.static_entities = []
        self.able_inline_list = inline_op_pos
        if force_inline:
            entity = [0] * op_num
            for pos in inline_op_pos:
                entity[pos] = 1
            self.static_entities.append(entity)
        else:
            num_inline_ops = len(inline_op_pos)
            enums = gen_enum([1, 0], num_inline_ops)
            for enum in enums:
                entity = [0] * op_num
                for i in range(num_inline_ops):
                    entity[inline_op_pos[i]] = enum[i]
                self.static_entities.append(entity)
        self.size = len(self.static_entities)
        self.num_direction = 2
        self.directions = [(-1,), (1,)]
        self.type_key = "inline"

    def next_entity(self, pos, d):
        # d is tuple
        if len(d) == 1:
            pos = (pos + d[0]) % self.size
            return pos
        else:
            raise RuntimeError(
                "Not support for direction more than one dim: {}".format(d))

    def get_direction(self, num):
        return self.directions[num % self.num_direction]

    def able_inline(self, pos):
        return pos in self.able_inline_list


class MergeSpce(SubSpace):
    def __init__(self, merge_op_pos, op_num, force_merge=False):
        self.dim = op_num
        self.static_entities = []
        self.able_merge_list = merge_op_pos
        if force_merge:
            entity = [0] * op_num
            for pos in merge_op_pos:
                entity[pos] = 1
            self.static_entities.append(entity)
        else:
            num_merge_ops = len(merge_op_pos)
            enums = gen_enum([1, 0], num_merge_ops)
            for enum in enums:
                entity = [0] * op_num
                for i in range(num_merge_ops):
                    entity[merge_op_pos[i]] = enum[i]
                self.static_entities.append(entity)
        self.size = len(self.static_entities)
        self.num_direction = 2
        self.directions = [(-1,), (1,)]
        self.type_key = "merge"

    def next_entity(self, pos, d):
        # d is tuple
        if len(d) == 1:
            pos = (pos + d[0]) % self.size
            return pos
        else:
            raise RuntimeError(
                "Not support for direction more than one dim: {}".format(d))

    def get_direction(self, num):
        return self.directions[num % self.num_direction]

    def able_merge(self, pos):
        return pos in self.able_merge_list


class EnumSpace(SubSpace):
    def __init__(self, knobs):
        self.dim = 2
        self.static_entities = knobs
        self.size = len(self.static_entities)
        self.num_direction = 2
        self.directions = [(-1,), (1,)]

    def next_entity(self, pos, d):
        # d is tuple
        if len(d) == 1:
            pos = (pos + d[0]) % self.size
            return pos
        else:
            raise RuntimeError(
                "Not support for direction more than one dim: {}".format(d))

    def get_direction(self, num):
        return self.directions[num % self.num_direction]


class IntrinSpace(SubSpace):
    def __init__(self, lst):
        self.dim = 1
        self.static_entities = lst
        self.size = len(self.static_entities)
        self.num_direction = 2
        self.directions = [(-1,), (1,)]

    def next_entity(self, pos, d):
        # d is tuple
        if len(d) == 1:
            pos = (pos + d[0]) % self.size
            return pos
        else:
            raise RuntimeError(
                "Not support for direction more than one dim: {}".format(d))

    def get_direction(self, num):
        return self.directions[num % self.num_direction]


def generate_inline_space(op_lst, down_graph, force_inline=False):
    inline_op_pos = []
    for i, op in enumerate(op_lst):
        if able_inline(op, down_graph):
            inline_op_pos.append(i)
    return InlineSpace(inline_op_pos, len(op_lst), force_inline=force_inline)


def generate_merge_space(op_lst, down_graph, force_merge=False):
    merge_ops = list(range(len(op_lst)))
    return MergeSpce(merge_ops, len(op_lst), force_merge=force_merge)


def generate_fuse_space(loops, groups):
    return FuseSpace(groups, loops)


def generate_split_space(extent, nparts, allow_non_divisible='off'):
    return np.array(SplitSpace(nparts, extent, allow_non_divisible=allow_non_divisible).static_entities)


def generate_reorder_space(num_spatial_axis):
    return ReorderSpace(num_spatial_axis)


def generate_unroll_space(slevel, par_k=True, explicit=False):
    # if par_k:
    #     return np.array(UnrollSpace(total=[1, 2, 3, 4, 5], slevel=slevel, explicit=explicit).static_entities)
    # else:
    return np.array(UnrollSpace(total=[2, 3, 4, 5], slevel=slevel, explicit=explicit).static_entities)

def generate_intrin_space(op, target):
    key = target_embedding[target]
    INTRIN_TABLE = get_intrin_table()
    if key not in INTRIN_TABLE:
        raise RuntimeError("Can't find any pre-defined intrinsic for target %s." % target)
    assert op.num_outputs == 1, "Only support one output"
    # out_t = op.output(0)
    expr = op.body[0]
    new_axis = []
    for ax in expr.axis:
        var_name = ax.var.name
        ax_type = ax.iter_type
        if ax_type == 2:  # reduce axis
            new_ax = tvm.reduce_axis((0, 2), name=var_name)
            new_axis.append(new_ax)
        else:
            new_axis.append(ax)

    new_expr = tvm.expr.Reduce(expr.combiner, expr.source, new_axis, tvm.api.convert(True), expr.value_index)
    new_op = tvm._api_internal._ComputeOp(
        op.name, op.tag, op.attrs, op.axis, [new_expr])
    out_t = new_op.output(0)

    candidates = []
    # key = target_embedding[target]
    # print("?"*15)
    INTRIN_TABLE = get_intrin_table()
    for no, intrin in enumerate(INTRIN_TABLE[key]):
        intrin_t, _ = intrin.func(*intrin.args)
        # print("*"*20)
        # print(no, intrin)
        intrin_axis = intrin_t.op.axis
        if hasattr(intrin_t.op, "reduce_axis"):
            intrin_reduce_axis = intrin_t.op.reduce_axis
        else:
            intrin_reduce_axis = []

        permute_axis = permutations(range(len(op.axis)), r=len(intrin_axis))
        if hasattr(op, "reduce_axis"):
            op_reduce_axis = op.reduce_axis
            # for re in op.reduce_axis:
            #     var_name = re.var.name
            #     op_reduce_axis.append(tvm.reduce_axis((0, 2), name=var_name))
            # op.reduce_axis
            permute_reduce_axis = permutations(range(len(op_reduce_axis)), r=len(intrin_reduce_axis))
        else:
            op_reduce_axis = []
            permute_reduce_axis = []

        # print("~~~~~~~~~~~~~~~~~~~~~~~~")
        # print(out_t.op.body)
        # print(intrin_t.op.body)
        # print("$$$$$$$$$$$$$$$$$$$$$$")
        # print(op.axis)
        # print("1!!!!!!!!!!!!!!!")
        # print(intrin_t.op.axis)
        for sp, re in product(permute_axis, permute_reduce_axis):
            # sp = [1, 2, 3]
            # for re in permute_reduce_axis:
            axis = [op.axis[i].var for i in sp]
            reduce_axis = [op_reduce_axis[i].var for i in re]
            # print(axis)
            # print(type(reduce_axis))
            # print(type(reduce_axis[0]))
            match = tvm.ir_pass.intrinsic_match(out_t, intrin_t, axis, reduce_axis)
            if match:
                candidates.append((key, no, sp, re))
    # print(len(candidates))
    if len(candidates) == 0:
        raise RuntimeError("Can't match any intrinsic for given compute %s." % (str(op.body)))
    return IntrinSpace(candidates)


def generate_space_intra_op(op, down_graph, slevel=4, rlevel=3, groups=3, split_policy="off",
                            unroll_policy="off", fuse_policy="fuse_spatial", reorder_policy="last"):
    spatial_axis_names = [x.var.name for x in op.axis]
    spatial_axis_extents = [x.dom.extent.value for x in op.axis]
    reduced_axis_names = [x.var.name for x in op.reduce_axis]
    reduced_axis_extents = [x.dom.extent.value for x in op.reduce_axis]

    ##############################################################
    # generate space:
    schedule_space = Space()

    # - fuse space
    if fuse_policy == "fuse_spatial":
        fuse_space = generate_fuse_space(spatial_axis_names, groups)
        schedule_space.add_subspace("fuse_spatial", fuse_space, "fuse")

    # - split space
    for i, (name, extent) in enumerate(zip(spatial_axis_names, spatial_axis_extents)):
        split_space = generate_split_space(extent, slevel, allow_non_divisible=split_policy)
        schedule_space.add_subspace("split_{}_{}".format(name, i), split_space, "spatial")
    for i, (name, extent) in enumerate(zip(reduced_axis_names, reduced_axis_extents)):
        split_space = generate_split_space(extent, rlevel, allow_non_divisible=split_policy)
        schedule_space.add_subspace("split_{}_{}".format(name, i), split_space, "reduce")

    # - reorder space
    if reorder_policy == "last":
        reorder_space = generate_reorder_space(groups)
        schedule_space.add_subspace("reorder", reorder_space, "reorder")

    # -unroll space
    unroll_space = generate_unroll_space(explicit=(unroll_policy == "explicit"))
    schedule_space.add_subspace("unroll", unroll_space, "unroll")

    # - other special spaces can be added

    return schedule_space


def generate_op_space_with_intrin(op, target, slevel=2, rlevel=2, split_policy="off", unroll_policy="off"):
    # print(op.axis)
    spatial_axis_names = [x.var.name for x in op.axis]
    spatial_axis_extents = [x.dom.extent.value for x in op.axis]
    reduced_axis_names = [x.var.name for x in op.reduce_axis]
    reduced_axis_extents = [x.dom.extent.value for x in op.reduce_axis]

    ##############################################################
    # generate space:
    schedule_space = Space()

    # - intrin space
    intrin_space = generate_intrin_space(op, target)
    schedule_space.add_subspace("intrinsic", intrin_space, "intrin")

    # - split space
    for i, (name, extent) in enumerate(zip(spatial_axis_names, spatial_axis_extents)):
        split_space = generate_split_space(extent, slevel, allow_non_divisible=split_policy)
        schedule_space.add_subspace("split_{}_{}".format(name, i), split_space, "spatial")
    for i, (name, extent) in enumerate(zip(reduced_axis_names, reduced_axis_extents)):
        split_space = generate_split_space(extent, rlevel, allow_non_divisible=split_policy)
        # if i in re_intrin_idx_lst:
        #     split_space = IntrinSplitSpace(extent)
        # else:
        #     split_space = generate_split_space(extent, 1, allow_non_divisible=split_policy)
        schedule_space.add_subspace("split_{}_{}".format(name, i), split_space, "reduce")

    unroll_space = generate_unroll_space(explicit=(unroll_policy == "explicit"))
    schedule_space.add_subspace("unroll", unroll_space, "unroll")

    # print(intrin_space.static_entities)
    # sp_intrin_idx_lst = reduce(lambda a, b: a + b, [cfg[2] for cfg in intrin_space.static_entities])
    # re_intrin_idx_lst = reduce(lambda a, b: a + b, [cfg[3] for cfg in intrin_space.static_entities])

    # # - split space
    # for i, (name, extent) in enumerate(zip(spatial_axis_names, spatial_axis_extents)):
    #     split_space = generate_split_space(extent, slevel if i in sp_intrin_idx_lst else 1,
    #                                        allow_non_divisible=split_policy)
    #     # if i in sp_intrin_idx_lst:
    #     #     split_space = IntrinSplitSpace(extent)
    #     # else:
    #     #     split_space = generate_split_space(extent, 1, allow_non_divisible=split_policy)
    #     print("split_{}_{}".format(name, i), extent, slevel if i in sp_intrin_idx_lst else 1)
    #     schedule_space.add_subspace("split_{}_{}".format(name, i), split_space, "spatial")
    # for i, (name, extent) in enumerate(zip(reduced_axis_names, reduced_axis_extents)):
    #     split_space = generate_split_space(extent, rlevel if i in re_intrin_idx_lst else 1,
    #                                        allow_non_divisible=split_policy)
    #     # if i in re_intrin_idx_lst:
    #     #     split_space = IntrinSplitSpace(extent)
    #     # else:
    #     #     split_space = generate_split_space(extent, 1, allow_non_divisible=split_policy)
    #     schedule_space.add_subspace("split_{}_{}".format(name, i), split_space, "reduce")

    # - reorder space
    reorder_space = generate_reorder_space(len(op.axis) + len(op.reduce_axis))
    schedule_space.add_subspace("reorder", reorder_space, "reorder")

    # - other special spaces can be added
    return schedule_space


def generate_space_inter_op(op_lst, down_graph, force_inline=False, force_merge=False, special_space=None):
    ##############################################################
    # generate space:
    schedule_space = Space()
    # - inline space
    inline_space = generate_inline_space(op_lst, down_graph, force_inline=force_inline)
    schedule_space.add_subspace("inline", inline_space, "inline")
    # - merge space
    # merge_space = generate_merge_space(op_lst, down_graph, force_merge=force_merge)
    # schedule_space.add_subspace("merge", merge_space, "merge")

    # - other special spaces can be added
    special_space = {} if special_space is None else special_space
    for key, sspace in special_space.items():
        schedule_space.add_subspace(key, sspace, "special")

    return schedule_space


def generate_empty_space_inter_op():
    ##############################################################
    # generate space:
    schedule_space = Space()

    return schedule_space


def get_hw_space():
    # hw_space = {'pe_x': np.array([4, 8, 16, 32, 64, 128, 256]), 'l2_kb': np.array([32, 64, 128, 256, 512, 1024, 2048]),
    #             'l1_byte': np.array([256, 512, 1024, 2048, 4096, 8192, 16384]), 'banks': np.array([1, 2, 4, 8, 16, 32, 64])}
    hw_space = {'pe_x': np.array([4, 8, 16, 32, 64]), 'l2_kb': np.array([32, 64, 128, 256, 512]),
                'l1_byte': np.array([256, 512, 1024, 2048, 4096]),
                'banks': np.array([1, 2, 4, 8, 16])}
    # hw_space = {'pe_x': np.array([4, 8, 16, 32]), 'l2_kb': np.array([32, 64, 128, 256]),
    #             'l1_byte': np.array([256, 512, 1024, 2048]),
    #             'banks': np.array([1, 2, 4, 8])}
    return hw_space


def get_sw_space(dimension, slevel, par_k=True, split_policy='off'):
    split_space = {'N': generate_split_space(dimension[0], slevel, allow_non_divisible=split_policy),
                   'K': generate_split_space(dimension[1], slevel, allow_non_divisible=split_policy),
                   'C': generate_split_space(dimension[2], slevel, allow_non_divisible=split_policy),
                   'Y': generate_split_space(dimension[3], slevel, allow_non_divisible=split_policy),
                   'X': generate_split_space(dimension[4], slevel, allow_non_divisible=split_policy),
                   'R': generate_split_space(dimension[5], slevel, allow_non_divisible=split_policy),
                   'S': generate_split_space(dimension[6], slevel, allow_non_divisible=split_policy)}
    sw_space = {'split_space': split_space, 'unroll_space': generate_unroll_space(slevel=2, par_k=par_k)}
    return sw_space

