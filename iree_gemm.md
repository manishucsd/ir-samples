## IREE GEMM Memory Layouts
IREE GEMMs are always row-row-row or TT cuBLAS/CUTLASS GEMM with row-major (T) output. 

```bash
matrixA (Row-Major, MxK): M rows x K cols (K-dim in matrixA is the fastest moving dimension in the memory)
matrixB (Row-Major, KxN): K rows x N cols (N-dim in matrixB is the fastest moving dimension in the memory)
matrixC (Row-Major, MxN): M rows x N cols (N-dim in matrixC is the fastest moving dimension in the memory)
```

Example:
```mlir
%3 = linalg.matmul ins(%1, %2 : tensor<42x32xf32>, tensor<32x64xf32>) -> outs(%0 : tensor<42x64xf32>)
matrixA: M(42) rows x K(32) cols 
matrixB: K(32) rows x N(64) cols 
matrixC: M(42) rows x N(64) cols 
```

## Now consider the following example
```mlir
#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @m16n8k16_fp16_row_row_row(%arg0: memref<42x32xf16, 3>, %arg1: memref<32x64xf16, 3>, %arg2: memref<42x64xf16, 3>) {
  %cst_0 = arith.constant dense<0.000000e+00> : vector<16x8xf16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %cst = arith.constant 0.000000e+00 : f16
  %A = vector.transfer_read %arg0[%c1, %c3], %cst {in_bounds = [true, true]} : memref<42x32xf16, 3>, vector<16x16xf16>
  %B = vector.transfer_read %arg1[%c3, %c3], %cst {permutation_map = #map0, in_bounds = [true, true]} : memref<32x64xf16, 3>, vector<8x16xf16>
  %C = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<42x64xf16, 3>, vector<16x8xf16>
  %D = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
  vector.transfer_write %D, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<42x64xf16, 3>
  return
}
```

# Fusing transpose with vector.transfer_read
```mlir
%B = vector.transfer_read %arg1[%c3, %c3], %cst {permutation_map = #map0, in_bounds = [true, true]} : memref<32x64xf16, 3>, vector<8x16xf16>
```

The `#map = affine_map<(d0, d1) -> (d1, d0)>` encodes the transposition on the
slice to match the vector shape of <8x16xf16>. Thus, vector is read into a column-major
vector of shape 8x16. The dimension with extent of 16 for matrixB is the fastest/contiguous
dimenstion in the hardware vector registers for vector.contract to be succussfuly lowered to
nvgpu.mma.sync (16816.f16.f16).