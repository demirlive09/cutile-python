# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from math import ceil

import torch
from torch.testing import make_tensor

import cuda.tile as ct
from util import assert_close, assert_equal
from cuda.tile._exception import TileTypeError
from conftest import float_dtypes, int_dtypes, bool_dtypes, dtype_id


@pytest.fixture
def shape():
    return (512, )


@pytest.fixture
def tile():
    return 64


@ct.kernel
def array_astype_to_float32(x, y, TILE: ct.Constant[int], use_method: ct.Constant[bool]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    if use_method:
        ty = tx.astype(np.float32)
    else:
        ty = ct.astype(tx, np.float32)
    ct.store(y, index=(bid,), tile=ty)


@pytest.mark.parametrize("use_method", [True, False])
def test_astype(shape, tile, use_method):
    x = make_tensor(shape, dtype=torch.int32, device='cuda')
    ref = x.to(torch.float32)
    y = torch.zeros_like(ref)
    grid = (ceil(shape[0] / tile), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, array_astype_to_float32, (x, y, tile, use_method))
    assert_equal(y, ref)


@ct.kernel
def array_bitcast(x, y, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ty = ct.bitcast(tx, y.dtype)
    ct.store(y, index=(bid,), tile=ty)


@ct.kernel
def kernel_astype_tf32(x, y, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ty = ct.astype(tx, ct.tfloat32)
    ty = ct.astype(ty, y.dtype)
    ct.store(y, index=(bid,), tile=ty)


@pytest.mark.parametrize("dtype", [torch.float16,
                                   torch.float32,
                                   torch.bfloat16,
                                   torch.float64])
def test_cast_tf32(dtype):
    # Test that tf32 is casted to float32
    x = make_tensor((32, 32), dtype=dtype, device='cuda')
    y = torch.zeros_like(x)

    # emulate TF32 cast in PyTorch by performing a matmul with diag(ones) in TF32 precision
    dummy = torch.eye(32, dtype=dtype, device='cuda')
    torch.set_float32_matmul_precision("high")
    ref = torch.matmul(x, dummy).view(-1)
    torch.set_float32_matmul_precision("highest")
    x = x.view(-1)
    y = y.view(-1)
    grid = (ceil(x.numel() / 32), 1)
    ct.launch(torch.cuda.current_stream(), grid, kernel_astype_tf32, (x, y, 32))
    assert_close(y, ref, atol=1e-6, rtol=1e-3)


@pytest.mark.parametrize("dtype_x, dtype_y", [
    # identities
    (torch.int32, torch.int32),
    (torch.float32, torch.float32),
    (torch.int64, torch.int64),
    (torch.float64, torch.float64),
    (torch.float16, torch.float16),
    # float/int pairs
    (torch.int32, torch.float32),
    (torch.float32, torch.int32),
    (torch.float64, torch.int64),
    (torch.int64, torch.float64),
    # failing pairs with different bitwidths
    (torch.int32, torch.int64),
    (torch.int64, torch.float32),
    (torch.float16, torch.int32),
])
def test_array_bitcast(shape, tile, dtype_x, dtype_y):
    # avoid inputs that could produce nans of infs to not break assert
    if dtype_x in (torch.int32, torch.int64):
        x = torch.randint(0, 100, shape, dtype=dtype_x, device='cuda')
    else:
        x = torch.randn(shape, dtype=dtype_x, device='cuda')
    ref = x.view(dtype=dtype_y)
    y = torch.zeros_like(ref)
    grid = (ceil(shape[0] / tile), 1, 1)
    if dtype_x.itemsize != dtype_y.itemsize:
        with pytest.raises(TileTypeError):
            ct.launch(torch.cuda.current_stream(), grid, array_bitcast, (x, y, tile))

    else:
        ct.launch(torch.cuda.current_stream(), grid, array_bitcast, (x, y, tile))
        assert_equal(y, ref)


@ct.kernel
def array_astype_bool_to_float(y):
    tx = ct.full((1,), True, dtype=ct.bool_)
    ty = ct.astype(tx, np.float32)
    ct.store(y, index=(0,), tile=ty)


def test_astype_bool_to_float():
    x = torch.zeros((1,), dtype=torch.float32, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1,), array_astype_bool_to_float, (x,))
    ref = torch.ones((1,), dtype=torch.float32, device='cuda')
    assert_equal(x, ref)


@ct.kernel
def scalar_astype(scalar, array_out):
    x = ct.astype(scalar, array_out.dtype)
    ct.store(array_out, (0,), x)


def test_astype_scalar():
    x = torch.zeros((1,), dtype=torch.float32, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1,),
              scalar_astype, (5, x,))
    ref = torch.full((1,), 5, dtype=torch.float32, device='cuda')
    assert_equal(x, ref)


def make_array_astype_kernel(to_dtype):
    @ct.kernel
    def kernel(x, y, TILE: ct.Constant[int]):
        bid = ct.bid(0)
        tx = ct.load(x, index=(bid,), shape=(TILE,))
        ty = ct.astype(tx, to_dtype)
        ct.store(y, index=(bid,), tile=ty)
    return kernel


@pytest.mark.parametrize("from_dtype", float_dtypes+int_dtypes+bool_dtypes, ids=dtype_id)
@pytest.mark.parametrize("to_dtype", float_dtypes+int_dtypes+bool_dtypes, ids=dtype_id)
def test_array_astype(shape, tile, from_dtype, to_dtype):
    x = make_tensor(shape, dtype=from_dtype, device='cuda') * 5
    # Make the second half of the array 0 to test truncation
    x[x.numel()//2:] = 0
    y = torch.zeros_like(x, dtype=to_dtype)
    grid = (ceil(x.numel() / tile), 1, 1)

    array_astype = make_array_astype_kernel(to_dtype)
    ct.launch(torch.cuda.current_stream(), grid, array_astype, (x, y, tile))
    assert_equal(y, x.to(y.dtype))
