from bitsandbytes.autograd._functions import *
from bitsandbytes.functional import *

import bitsandbytes.functional
import bitsandbytes.autograd._functions


def pre_call(device):
    return device


def post_call(prev_device):
    return

bitsandbytes.functional.pre_call = pre_call
bitsandbytes.functional.post_call = post_call


def get_colrow_absmax(
    A, row_stats=None, col_stats=None, nnz_block_ptr=None, threshold=0.0
):
    assert A.dtype == torch.float16
    device = A.device

    cols = A.shape[-1]
    if len(A.shape) == 3:
        rows = A.shape[0] * A.shape[1]
    else:
        rows = A.shape[0]

    col_tiles = (cols + 255) // 256
    tiled_rows = ((rows + 15) // 16) * 16
    if row_stats is None:
        row_stats = torch.empty(
            (rows,), dtype=torch.float32, device=device
        ).fill_(-50000.0)
    if col_stats is None:
        col_stats = torch.empty(
            (cols,), dtype=torch.float32, device=device
        ).fill_(-50000.0)

    if nnz_block_ptr is None and threshold > 0.0:
        nnz_block_ptr = torch.zeros(
            ((tiled_rows * col_tiles) + 1,), dtype=torch.int32, device=device
        )

    ptrA = get_ptr(A)
    ptrRowStats = get_ptr(row_stats)
    ptrColStats = get_ptr(col_stats)
    ptrNnzrows = get_ptr(nnz_block_ptr)
    rows = ct.c_int32(rows)
    cols = ct.c_int32(cols)

    prev_device = pre_call(A.device)
    is_on_gpu([A, row_stats, col_stats, nnz_block_ptr])
    lib.cget_col_row_stats(ptrA, ptrRowStats, ptrColStats,
                           ptrNnzrows, ct.c_float(threshold), rows, cols)
    post_call(prev_device)

    if threshold > 0.0:
        # torch.compile needs explicit dtype here
        nnz_block_ptr.cumsum_(0, dtype=nnz_block_ptr.dtype)

    return row_stats, col_stats, nnz_block_ptr

def double_quant(
    A, col_stats=None, row_stats=None, out_col=None, out_row=None, threshold=0.0
):
    device = A.device
    assert A.dtype == torch.half
    assert device.type == "cuda"
    prev_device = pre_call(A.device)

    cols = A.shape[-1]
    if len(A.shape) == 3:
        rows = A.shape[0] * A.shape[1]
    else:
        rows = A.shape[0]

    if row_stats is None or col_stats is None:
        row_stats, col_stats, nnz_row_ptr = get_colrow_absmax(
            A, threshold=threshold
        )

    if out_col is None:
        out_col = torch.zeros(A.shape, device=device, dtype=torch.int8)
    if out_row is None:
        out_row = torch.zeros(A.shape, device=device, dtype=torch.int8)

    coo_tensor = None
    ptrA = get_ptr(A)
    ptrColStats = get_ptr(col_stats)
    ptrRowStats = get_ptr(row_stats)
    ptrOutCol = get_ptr(out_col)
    ptrOutRow = get_ptr(out_row)

    is_on_gpu([A, col_stats, row_stats, out_col, out_row])
    if threshold > 0.0:
        nnz = nnz_row_ptr[-1]# .item()
        if nnz > 0:
        # if (nnz > 0).any():
            coo_tensor = coo_zeros(
                A.shape[0], A.shape[1],
                # nnz_row_ptr[-1].item(),
                nnz_row_ptr[-1],
                device
            )
            ptrRowIdx = get_ptr(coo_tensor.rowidx)
            ptrColIdx = get_ptr(coo_tensor.colidx)
            ptrVal = get_ptr(coo_tensor.values)
            ptrRowPtr = get_ptr(nnz_row_ptr)

            lib.cdouble_rowcol_quant(
                ptrA,
                ptrRowStats,
                ptrColStats,
                ptrOutCol,
                ptrOutRow,
                ptrRowIdx,
                ptrColIdx,
                ptrVal,
                ptrRowPtr,
                ct.c_float(threshold),
                ct.c_int32(rows),
                ct.c_int32(cols),
            )
            val, idx = torch.sort(coo_tensor.rowidx)
            coo_tensor.rowidx = val
            coo_tensor.colidx = coo_tensor.colidx[idx]
            coo_tensor.values = coo_tensor.values[idx]
        else:
            lib.cdouble_rowcol_quant(
                ptrA,
                ptrRowStats,
                ptrColStats,
                ptrOutCol,
                ptrOutRow,
                None,
                None,
                None,
                None,
                ct.c_float(0.0),
                ct.c_int32(rows),
                ct.c_int32(cols),
            )
    else:
        lib.cdouble_rowcol_quant(
            ptrA,
            ptrRowStats,
            ptrColStats,
            ptrOutCol,
            ptrOutRow,
            None,
            None,
            None,
            None,
            ct.c_float(threshold),
            ct.c_int32(rows),
            ct.c_int32(cols),
        )
    post_call(prev_device)

    return out_row, out_col, row_stats, col_stats, coo_tensor

bitsandbytes.functional.get_colrow_absmax = get_colrow_absmax
bitsandbytes.functional.double_quant = double_quant

@staticmethod
def MMforward(ctx, A, B, out=None, bias=None, state=MatmulLtState):
    # using_igemmlt = torch.cuda.get_device_capability(device=A.device) >= (7, 5) and not state.force_no_igemmlt
    using_igemmlt = not state.force_no_igemmlt
    # default of pytorch behavior if inputs are empty
    ctx.is_empty = False
    if prod(A.shape) == 0:
        ctx.is_empty = True
        ctx.A = A
        ctx.B = B
        ctx.bias = bias
        if A.shape[-1] == B.shape[0]:
            return torch.empty(A.shape[:-1] + B.shape[1:], dtype=A.dtype, device=A.device)
        else:
            return torch.empty(A.shape[:-1] + B.shape[:1], dtype=A.dtype, device=A.device)

    # 1. Quantize A
    # 2. Quantize B
    # 3. Matmul
    # 4. Mixed-precision decomposition matmul
    # 5. Save state
    formatB = state.formatB
    input_shape = A.shape
    if state.outlier_pool is None:
        state.outlier_pool = GlobalOutlierPooler.get_instance()

    # Cast A to fp16
    if A.dtype != torch.float16:
        warnings.warn(
            f"MatMul8bitLt: inputs will be cast from {A.dtype} to float16 during quantization")

    # 1. Quantize A
    if len(A.shape) == 3:
        A = A.view(-1, A.shape[-1]).contiguous()
    CA, CAt, SCA, SCAt, coo_tensorA = F.double_quant(
        A.to(torch.float16), threshold=state.threshold)

    if state.threshold > 0.0 and coo_tensorA is not None:
        if state.has_fp16_weights:
            idx = torch.unique(coo_tensorA.colidx).long()
            CA[:, idx] = 0
            CAt[:, idx] = 0
            subA = A[:, idx]
            state.subB = B[:, idx].t().contiguous()
            state.idx = idx
        else:
            if state.CxB is None and using_igemmlt:
                # B in in 8-bit row-major, we can transform it back to 16-bit to extract outlier dimensions
                # we also need to convert it to the turing/ampere format
                state.CxB, state.SB = F.transform(state.CB, to_order=formatB)
    else:
        if not state.has_fp16_weights and state.CxB is None and using_igemmlt:
            state.CxB, state.SB = F.transform(state.CB, to_order=formatB)
        subA = None

    # 2. Quantize B
    if state.has_fp16_weights:
        has_grad = True if (getattr(B, "grad", None) is not None) else False
        is_transposed = not B.is_contiguous() and B.shape[0] == B.stride(1)
        if is_transposed:
            B = B.contiguous()

        if (state.is_training and not has_grad) or state.CxB is None:
            state.reset_grads()
            (
                CB,
                state.CBt,
                state.SCB,
                state.SCBt,
                coo_tensorB,
            ) = F.double_quant(B.to(torch.float16))
            if using_igemmlt:
                state.CxB, state.SB = F.transform(CB, to_order=formatB)
            else:
                state.CB = CB
    else:
        has_grad = False

    if coo_tensorA is not None and not state.has_fp16_weights:
        # extract outliers

        outlier_idx = torch.unique(coo_tensorA.colidx)
        state.idx = outlier_idx
        # state.outlier_pool.add_outliers(outlier_idx, A.shape[-1])
        # if state.use_pool and state.outlier_pool.model_dim == A.shape[-1]:
        #    # do not use pool for 2nd FFN layer
        #    state.idx = state.outlier_pool.get_current_outlier_idx().to(A.device)
        # else:
        #    state.idx = outlier_idx
        if state.CxB is not None:
            outliers = F.extract_outliers(state.CxB, state.SB, state.idx.int())
        else:
            outliers = state.CB[:, state.idx.long()].clone()

        state.subB = (outliers * state.SCB.view(-1, 1) /
                      127.0).t().contiguous().to(A.dtype)
        CA[:, state.idx.long()] = 0
        CAt[:, state.idx.long()] = 0
        subA = A[:, state.idx.long()]

    shapeB = state.SB[0] if state.SB else B.shape

    if len(input_shape) == 3:
        output_shape = (input_shape[0], input_shape[1], shapeB[0])
    else:
        output_shape = (input_shape[0], shapeB[0])

    # 3. Matmul
    if using_igemmlt:
        C32A, SA = F.transform(CA, "col32")
        out32, Sout32 = F.igemmlt(C32A, state.CxB, SA, state.SB)
        if bias is None or bias.dtype == torch.float16:
            # we apply the fused bias here
            output = F.mm_dequant(out32, Sout32, SCA, state.SCB, bias=bias)
            output = output.to(A.dtype)
        else:  # apply bias separately
            output = F.mm_dequant(out32, Sout32, SCA, state.SCB, bias=None)
            output = output.to(A.dtype).add_(bias)

    else:
        A_wo_outliers = A.clone()
        if state.idx is not None:
            A_wo_outliers[:, state.idx.long()] = 0
        output = torch.nn.functional.linear(
            A_wo_outliers, state.CB.to(A.dtype))
        output = output.mul_(state.SCB.unsqueeze(0).mul(1.0 / 127.0))
        if bias is not None:
            output = output.add_(bias)

    # 4. Mixed-precision decomposition matmul
    if coo_tensorA is not None and subA is not None:
        output += torch.matmul(subA, state.subB)

    # 5. Save state
    ctx.state = state

    ctx.formatB = formatB
    ctx.grad_shape = input_shape
    ctx.dtype_A, ctx.dtype_B, ctx.dtype_bias = A.dtype, B.dtype, None if bias is None else bias.dtype

    if any(ctx.needs_input_grad[:2]):
        ctx.tensors = (CAt, subA)
        ctx.tensor_states = (SCAt, state.idx)
    else:
        ctx.tensors = [None, None]
        ctx.tensor_states = (None, None)
        ctx.save_for_backward(None, None)

    clone_func = torch.clone if len(output_shape) == 3 else lambda x: x
    return clone_func(output.view(output_shape))

bitsandbytes.autograd._functions.MatMul8bitLt.forward = MMforward
