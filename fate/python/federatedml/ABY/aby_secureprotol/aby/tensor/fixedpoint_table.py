import operator
from collections import Iterable
import functools

import numpy as np

from fate_arch.common import Party
from fate_arch.session import is_table
from federatedml.ABY.aby_secureprotol.aby.beaver_triples import beaver_triplets
from federatedml.ABY.aby_secureprotol.aby.tensor import fixedpoint_numpy
from federatedml.ABY.aby_secureprotol.aby.tensor.base import ABYTensorBase
from federatedml.ABY.aby_secureprotol.aby.utils import NamingService
from federatedml.ABY.aby_secureprotol.aby.utils import urand_tensor
# from federatedml.ABY.aby_secureprotol.aby.tensor.fixedpoint_endec import FixedPointEndec
# from federatedml.ABY.aby_secureprotol.fixedpoint import FixedPointEndec
from federatedml.ABY.aby_secureprotol.aby_fixedpoint import ABYFixedPointEndec, ABYFixedPointNumber
from federatedml.ABY.operator.vector_operator64 import vector_add_operator_client, vector_mul_operator_client, \
    vector_operator_execute


def _table_binary_op(x, y, op):
    return x.join(y, lambda a, b: op(a, b))


def _table_binary_mod_op(x, y, q_field, op):
    return x.join(y, lambda a, b: op(a, b) % q_field)


def _table_scalar_op(x, d, op):
    return x.mapValues(lambda a: op(a, d))


def _table_scalar_mod_op(x, d, q_field, op):
    return x.mapValues(lambda a: op(a, d) % q_field)


def _table_dot_mod_func(it, q_field):
    ret = None
    for _, (x, y) in it:
        if ret is None:
            ret = np.tensordot(x, y, [[], []]) % q_field
        else:
            ret = (ret + np.tensordot(x, y, [[], []])) % q_field
    return ret


def _table_dot_func(it):
    ret = None
    for _, (x, y) in it:
        if ret is None:
            ret = np.tensordot(x, y, [[], []])
        else:
            ret += np.tensordot(x, y, [[], []])
    return ret


def table_dot(a_table, b_table):
    return a_table.join(b_table, lambda x, y: [x, y]) \
        .applyPartitions(lambda it: _table_dot_func(it)) \
        .reduce(lambda x, y: x + y)


def table_dot_mod(a_table, b_table, q_field):
    return a_table.join(b_table, lambda x, y: [x, y]) \
        .applyPartitions(lambda it: _table_dot_mod_func(it, q_field)) \
        .reduce(lambda x, y: x if y is None else y if x is None else x + y)


class ABYFixedPointTensor(ABYTensorBase):
    """
    a table based tensor
    """
    __array_ufunc__ = None

    def __init__(self, value:np.array, q_field, endec, tensor_name: str = None):
        super().__init__(q_field, tensor_name)
        self.value = value
        self.endec = endec


    def dot(self, other: 'ABYFixedPointTensor', target_name=None):
        encoded_vector = self.value
        vec = np.array([x.encoding for x in encoded_vector])
        address = "127.0.0.1"
        port = 7766
        vec_len = len(vec)
        if vec_len <= 0:
            raise ValueError("vec_len <= 0")
        result_vec, result_type = vector_operator_execute(vector_mul_operator_client(), vec, address, port)
        result_vector = np.array([ABYFixedPointNumber(x, encoded_vector[0].exponent) for x in result_vec[:vec_len]])
        return self._boxed(result_vector)

    def dot_local(self, other, target_name=None):
        def _vec_dot(x, y, party_idx, q_field, endec):
            ret = np.dot(x, y) % q_field
            ret = endec.truncate(ret, party_idx)
            if not isinstance(ret, np.ndarray):
                ret = np.array([ret])
            return ret

        if isinstance(other, ABYFixedPointTensor) or isinstance(other, fixedpoint_numpy.ABYFixedPointTensor):
            other = other.value

        if isinstance(other, np.ndarray):
            party_idx = self.get_aby().party_idx
            f = functools.partial(_vec_dot, y=other,
                                  party_idx=party_idx,
                                  q_field=self.q_field,
                                  endec=self.endec)
            ret = self.value.mapValues(f)
            return self._boxed(ret, target_name)

        elif is_table(other):
            ret = table_dot_mod(self.value, other, self.q_field).reshape((1, -1))[0]
            ret = self.endec.truncate(ret, self.get_aby().party_idx)
            return fixedpoint_numpy.ABYFixedPointTensor(ret,
                                                        self.q_field,
                                                        self.endec,
                                                        target_name)
        else:
            raise ValueError(f"type={type(other)}")

    def reduce(self, func, **kwargs):
        ret = self.value.reduce(func)
        return fixedpoint_numpy.ABYFixedPointTensor(ret,
                                                    self.q_field,
                                                    self.endec
                                                    )

    @property
    def shape(self):
        return self.value.count(), len(self.value.first()[1])

    @classmethod
    def from_source(cls, tensor_name, source, **kwargs):
        spdz = cls.get_aby()
        q_field = kwargs['q_field'] if 'q_field' in kwargs else spdz.q_field
        if 'encoder' in kwargs:
            encoder = kwargs['encoder']
        else:
            base = kwargs['base'] if 'base' in kwargs else 10
            frac = kwargs['frac'] if 'frac' in kwargs else 4
            encoder = FixedPointEndec(n=q_field, field=q_field, base=base, precision_fractional=frac)
        if is_table(source):
            source = encoder.encode(source)
            _pre = urand_tensor(q_field, source, use_mix=spdz.use_mix_rand)
            spdz.communicator.remote_share(share=_pre, tensor_name=tensor_name, party=spdz.other_parties[0])
            for _party in spdz.other_parties[1:]:
                r = urand_tensor(q_field, source, use_mix=spdz.use_mix_rand)
                spdz.communicator.remote_share(share=_table_binary_mod_op(r, _pre, q_field, operator.sub),
                                               tensor_name=tensor_name, party=_party)
                _pre = r
            share = _table_binary_mod_op(source, _pre, q_field, operator.sub)
        elif isinstance(source, Party):
            share = spdz.communicator.get_share(tensor_name=tensor_name, party=source)[0]
        else:
            raise ValueError(f"type={type(source)}")
        return ABYFixedPointTensor(share, q_field, encoder, tensor_name)

    def get(self, tensor_name=None, broadcast=True):
        return self.endec.decode(self.rescontruct(tensor_name, broadcast))

    def rescontruct(self, tensor_name=None, broadcast=True):
        from federatedml.ABY.aby_secureprotol.aby import ABY
        aby = ABY.get_instance()
        share_val = self.value.copy()
        name = tensor_name or self.tensor_name

        if name is None:
            raise ValueError("name not specified")

        # remote share to other parties
        if broadcast:
            aby.communicator.broadcast_rescontruct_share(share_val, name)

        # get shares from other parties
        for other_share in aby.communicator.get_rescontruct_shares(name):
            share_val = _table_binary_mod_op(share_val, other_share, self.q_field, operator.add)
        return share_val

    def broadcast_reconstruct_share(self, tensor_name=None):
        from federatedml.ABY.aby_secureprotol.aby import ABY
        aby = ABY.get_instance()
        share_val = self.value.copy()
        name = tensor_name or self.tensor_name
        if name is None:
            raise ValueError("name not specified")
        # remote share to other parties
        aby.communicator.broadcast_rescontruct_share(share_val, name)
        return share_val

    def __str__(self):
        return f"tensor_name={self.tensor_name}, value={self.value}"

    def __repr__(self):
        return self.__str__()

    def as_name(self, tensor_name):
        return self._boxed(value=self.value, tensor_name=tensor_name)

    def __add__(self, other):
        if isinstance(other, ABYPaillierFixedPointTensor):
            z_value = _table_binary_op(self.value, other.value, operator.add)
            return ABYPaillierFixedPointTensor(z_value)
        elif isinstance(other, ABYFixedPointTensor):
            z_value = _table_binary_mod_op(self.value, other.value, self.q_field, operator.add)
        elif is_table(other):
            z_value = _table_binary_mod_op(self.value, other, self.q_field, operator.add)
        else:
            z_value = _table_scalar_mod_op(self.value, other, self.q_field, operator.add)
        return self._boxed(z_value)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, ABYPaillierFixedPointTensor):
            z_value = _table_binary_op(self.value, other.value, operator.sub)
            return ABYPaillierFixedPointTensor(z_value)
        elif isinstance(other, ABYFixedPointTensor):
            z_value = _table_binary_mod_op(self.value, other.value, self.q_field, operator.sub)
        elif is_table(other):
            z_value = _table_binary_mod_op(self.value, other, self.q_field, operator.sub)
        else:
            z_value = _table_scalar_mod_op(self.value, other, self.q_field, operator.sub)

        return self._boxed(z_value)

    def __rsub__(self, other):
        if isinstance(other, (ABYPaillierFixedPointTensor, ABYFixedPointTensor)):
            return other - self
        elif is_table(other):
            z_value = _table_binary_mod_op(other, self.value, self.q_field, operator.sub)
        else:
            z_value = _table_scalar_mod_op(self.value, other, self.q_field, -1 * operator.sub)
        return self._boxed(z_value)

    def __mul__(self, other):
        if isinstance(other, ABYFixedPointTensor):
            z_value = _table_binary_mod_op(self.value, other.value, self.q_field, operator.mul)
        elif isinstance(other, ABYPaillierFixedPointTensor):
            z_value = _table_binary_op(self.value, other.value, operator.mul)
        else:
            z_value = _table_scalar_mod_op(self.value, other, self.q_field, operator.mul)
        z_value = self.endec.truncate(z_value, self.get_aby().party_idx)
        return self._boxed(z_value)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mod__(self, other):
        if not isinstance(other, (int, np.integer)):
            raise NotImplementedError("__mod__ support integer only")
        return self._boxed(_table_scalar_op(self.value, other, operator.mod))

    def _boxed(self, value, tensor_name=None):
        return ABYFixedPointTensor(value=value, q_field=self.q_field, endec=self.endec, tensor_name=tensor_name)


class ABYPaillierFixedPointTensor(ABYTensorBase):
    __array_ufunc__ = None

    def __init__(self, value, tensor_name: str = None, cipher=None):
        super().__init__(q_field=None, tensor_name=tensor_name)
        self.value = value
        self.cipher = cipher

    def dot(self, other, target_name=None):
        def _vec_dot(x, y):
            ret = np.dot(x, y)
            if not isinstance(ret, np.ndarray):
                ret = np.array([ret])
            return ret

        if isinstance(other, (ABYFixedPointTensor, fixedpoint_numpy.ABYFixedPointTensor)):
            other = other.value

        if isinstance(other, np.ndarray):
            ret = self.value.mapValues(lambda x: _vec_dot(x, other))
            return self._boxed(ret, target_name)

        elif is_table(other):
            ret = table_dot(self.value, other).reshape((1, -1))[0]
            return fixedpoint_numpy.PaillierFixedPointTensor(ret, target_name)
        else:
            raise ValueError(f"type={type(other)}")

    def reduce(self, func, **kwargs):
        ret = self.value.reduce(func)
        return fixedpoint_numpy.PaillierFixedPointTensor(ret)

    def __str__(self):
        return f"tensor_name={self.tensor_name}, value={self.value}"

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        if isinstance(other, (ABYPaillierFixedPointTensor, ABYFixedPointTensor)):
            return self._boxed(_table_binary_op(self.value, other.value, operator.add))
        elif is_table(other):
            return self._boxed(_table_binary_op(self.value, other, operator.add))
        else:
            return self._boxed(_table_scalar_op(self.value, other, operator.add))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (ABYPaillierFixedPointTensor, ABYFixedPointTensor)):
            return self._boxed(_table_binary_op(self.value, other.value, operator.sub))
        elif is_table(other):
            return self._boxed(_table_binary_op(self.value, other, operator.sub))
        else:
            return self._boxed(_table_scalar_op(self.value, other, operator.sub))

    def __rsub__(self, other):
        if isinstance(other, (ABYPaillierFixedPointTensor, ABYFixedPointTensor)):
            return self._boxed(_table_binary_op(other.value, self.value, operator.sub))
        elif is_table(other):
            return self._boxed(_table_binary_op(other, self.value, operator.sub))
        else:
            return self._boxed(_table_scalar_op(self.value, other, -1 * operator.sub))

    def __mul__(self, other):
        if isinstance(other, ABYFixedPointTensor):
            z_value = _table_binary_op(self.value, other.value, operator.mul)
        elif is_table(other):
            z_value = _table_binary_op(self.value, other, operator.mul)
        else:
            z_value = _table_scalar_op(self.value, other, operator.mul)
        return self._boxed(z_value)

    def __rmul__(self, other):
        return self.__mul__(other)

    def _boxed(self, value, tensor_name=None):
        return ABYPaillierFixedPointTensor(value=value, tensor_name=tensor_name)

    @classmethod
    def from_source(cls, tensor_name, source, **kwargs):
        spdz = cls.get_aby()
        q_field = kwargs['q_field'] if 'q_field' in kwargs else spdz.q_field

        if 'encoder' in kwargs:
            encoder = kwargs['encoder']
        else:
            base = kwargs['base'] if 'base' in kwargs else 10
            frac = kwargs['frac'] if 'frac' in kwargs else 4
            encoder = FixedPointEndec(n=q_field, field=q_field, base=base, precision_fractional=frac)

        if is_table(source):
            _pre = urand_tensor(q_field, source, use_mix=spdz.use_mix_rand)

            share = _pre

            spdz.communicator.remote_share(share=_table_binary_op(source, encoder.decode(_pre), operator.sub),
                                           tensor_name=tensor_name, party=spdz.other_parties[-1])
            return ABYFixedPointTensor(value=share,
                                       q_field=q_field,
                                       endec=encoder,
                                       tensor_name=tensor_name)

        elif isinstance(source, Party):
            share = spdz.communicator.get_share(tensor_name=tensor_name, party=source)[0]
            is_cipher_source = kwargs['is_cipher_source'] if 'is_cipher_source' in kwargs else True
            if is_cipher_source:
                cipher = kwargs.get("cipher")
                if cipher is None:
                    raise ValueError("Cipher is not provided")

                share = cipher.distribute_decrypt(share)
                share = encoder.encode(share)
            return ABYFixedPointTensor(value=share,
                                       q_field=q_field,
                                       endec=encoder,
                                       tensor_name=tensor_name)
        else:
            raise ValueError(f"type={type(source)}")