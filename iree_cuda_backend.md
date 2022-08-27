# IREE CUDA Backend linalg.matmul as the input IR to PTX

## Input Linalg Matmul
```mlir
module {
 func.func @matmul_f16_f16() {
   %0 = util.unfoldable_constant dense<1.000000e+00> : tensor<3456x2048xf16>
   %1 = util.unfoldable_constant dense<1.000210e-02> : tensor<2048x1024xf16>
   %cst = arith.constant 0.000000e+00 : f16
   %2 = linalg.init_tensor [3456, 1024] : tensor<3456x1024xf16>
   %3 = linalg.fill ins(%cst : f16) outs(%2 : tensor<3456x1024xf16>) -> tensor<3456x1024xf16>
   %4 = linalg.matmul ins(%0, %1 : tensor<3456x2048xf16>, tensor<2048x1024xf16>) outs(%3 : tensor<3456x1024xf16>) -> tensor<3456x1024xf16>
   check.expect_almost_eq_const(%4, dense<2.009380e+01> : tensor<3456x1024xf16>) : tensor<3456x1024xf16>
   return
 }
}
```
Listing 1. Specifically we are interested in lowering of linalg.matmul at line number 11.

Starting with high-level matmul description. The below IR only specifies the gemm problem size (3456x1024x2048), input datatype,
output datatype, and accumulation datatype.

## Tile and Distribute to CTAs
```mlir
// -----// IR Dump After TileAndDistributeToWorkgroups (iree-codegen-tile-and-distribute-to-workgroups) //----- //
…
%3 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_y]
%4 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_count_y]
scf.for %arg0 = %3 to %c3456 step %4 {
  %5 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_x]
  %6 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_count_x]
  scf.for %arg1 = %5 to %c1024 step %6 {
    %7 = flow.dispatch.tensor.load %0, offsets = [%arg0, %c0], sizes = [128, 2048], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:3456x2048xf16> -> tensor<128x2048xf16>
    %8 = flow.dispatch.tensor.load %1, offsets = [%c0, %arg1], sizes = [2048, 128], strides = [%c1, %c1] : !flow.dispatch.tensor<readonly:2048x1024xf16> -> tensor<2048x128xf16>
    %9 = linalg.init_tensor [128, 128] : tensor<128x128xf16>
    %10 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 64]]>} ins(%cst : f16) outs(%9 : tensor<128x128xf16>) -> tensor<128x128xf16>
    %11 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 64]]>} ins(%7, %8 : tensor<128x2048xf16>, tensor<2048x128xf16>) outs(%10 : tensor<128x128xf16>) -> tensor<128x128xf16>
    flow.dispatch.tensor.store %11, %2, offsets = [%arg0, %arg1], sizes = [128, 128], strides = [%c1, %c1] : tensor<128x128xf16> -> !flow.dispatch.tensor<writeonly:3456x1024xf16>
    }
…
```
Listing 2. We are specifically interested in scf.for (structured control flow). 

```mlir
scf.for %arg0 = %3 to %c3456 step %4 {
  scf.for %arg1 = %5 to %c1024 step %6 {
```

**Key takeways from TileAndDistributeToWorkgroups:**
- CTA level tiling and distribute to break into the computation into thread blocks of 128x128 on output C matrix.
- These two `scf.for` loops are tiled using user provided tile size (`128x128`) and distributed accross SMs. 
- Thus, reducing linalg.matmul too tiles of `128x128x2048`. 
- There is a TileK=64 and we will tile accross K dimension as well; however, TileK will not be distributed accrss SMs as split-k-slice=1.
- Note that there is not split-k-slice used in this example, but split-k-slice is supported by IREE.

```mlir
%11 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 64]]>} ins(%7, %8 : tensor<128x2048xf16>, tensor<2048x128xf16>) outs(%10 : tensor<128x128xf16>) -> tensor<128x128xf16>
```

## Bufferize (tensors -> memref)
```mlir
// -----// IR Dump After IREEComprehensiveBufferize (iree-codegen-iree-comprehensive-bufferize) //----- //
module {
  func.func @_matmul_f16_f16_dispatch_0() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c3456 = arith.constant 3456 : index
    %c1024 = arith.constant 1024 : index
    %cst = arith.constant 0.000000e+00 : f16
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : memref<3456x2048xf16>
    memref.assume_alignment %0, 64 : memref<3456x2048xf16>
    %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:3456x2048xf16>
    %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : memref<2048x1024xf16>
    memref.assume_alignment %2, 64 : memref<2048x1024xf16>
    %3 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:2048x1024xf16>
    %4 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : memref<3456x1024xf16>
    memref.assume_alignment %4, 64 : memref<3456x1024xf16>
    %5 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:3456x1024xf16>
    %workgroup_id_x = hal.interface.workgroup.id[0] : index
    %workgroup_count_x = hal.interface.workgroup.count[0] : index
    %workgroup_id_y = hal.interface.workgroup.id[1] : index
    %workgroup_count_y = hal.interface.workgroup.count[1] : index
    %6 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_y]
    %7 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_count_y]
    scf.for %arg0 = %6 to %c3456 step %7 {
      %8 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_x]
      %9 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_count_x]
      scf.for %arg1 = %8 to %c1024 step %9 {
        %10 = memref.subview %4[%arg0, %arg1] [128, 128] [1, 1] : memref<3456x1024xf16> to memref<128x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
        %11 = bufferization.to_tensor %10 : memref<128x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
        %12 = memref.subview %0[%arg0, 0] [128, 2048] [1, 1] : memref<3456x2048xf16> to memref<128x2048xf16, affine_map<(d0, d1)[s0] -> (d0 * 2048 + s0 + d1)>>
        %13 = bufferization.to_tensor %12 : memref<128x2048xf16, affine_map<(d0, d1)[s0] -> (d0 * 2048 + s0 + d1)>>
        %14 = memref.subview %2[0, %arg1] [2048, 128] [1, 1] : memref<2048x1024xf16> to memref<2048x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
        %15 = bufferization.to_tensor %14 : memref<2048x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
        linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 64]]>} ins(%cst : f16) outs(%10 : memref<128x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>)
        %16 = bufferization.to_tensor %10 : memref<128x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
        linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 64]]>} ins(%12, %14 : memref<128x2048xf16, affine_map<(d0, d1)[s0] -> (d0 * 2048 + s0 + d1)>>, memref<2048x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>) outs(%10 : memref<128x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>)
        %17 = bufferization.to_tensor %10 : memref<128x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
        %18 = memref.subview %4[%arg0, %arg1] [128, 128] [%c1, %c1] : memref<3456x1024xf16> to memref<128x128xf16, affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>>
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%10 : memref<128x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>) outs(%18 : memref<128x128xf16, affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>>) {
        ^bb0(%arg2: f16, %arg3: f16):
          linalg.yield %arg2 : f16
        }
      }
    }
    return
  }
}
```
Listing 3. Bufferization pass converts tensors to memref and create memref.subviews.

**Key takeways from IREEComprehensiveBufferize:**
- tensors vs. memref
   - tensors are immutable while memrefs are mutable memory allocations
   - memref.subview creates the view of intput operands A and B and output C for single tiled-iteration 128x128x64.

## Tile and Distribute to Warps
```mlir
// -----// IR Dump After LLVMGPUTileAndDistribute (iree-llvmgpu-tile-and-distribute) //----- //
func.func @_matmul_f16_f16_dispatch_0() {
  ...
  scf.for %arg0 = %6 to %c3456 step %7 { // CtaM tile
    %8 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_id_x]
    %9 = affine.apply affine_map<()[s0] -> (s0 * 128)>()[%workgroup_count_x]
    scf.for %arg1 = %8 to %c1024 step %9 { // CtaN tile
      %10 = memref.subview %5[%arg0, %arg1] [128, 128] [1, 1] : memref<3456x1024xf16> to memref<128x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
      %11 = memref.subview %3[%arg0, 0] [128, 2048] [1, 1] : memref<3456x2048xf16> to memref<128x2048xf16, affine_map<(d0, d1)[s0] -> (d0 * 2048 + s0 + d1)>>
      %12 = memref.subview %4[0, %arg1] [2048, 128] [1, 1] : memref<2048x1024xf16> to memref<2048x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
      %13 = gpu.thread_id  x
      %14 = gpu.thread_id  y
      %15 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%14]
      scf.for %arg2 = %15 to %c128 step %c128 { // 128x128 tile linalg.fill in m-dim
        %16 = affine.apply affine_map<(d0) -> ((d0 floordiv 32) * 64)>(%13)
        scf.for %arg3 = %16 to %c128 step %c128 { // 128x128 tile linalg.fill in n-dim
          %17 = memref.subview %2[%arg2, %arg3] [64, 64] [1, 1] : memref<128x128xf16, 3> to memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3>
          linalg.fill {__internal_linalg_transform__ = "vectorize", lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 64]]>} ins(%cst : f16) outs(%17 : memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3>)
        }
      }
      scf.for %arg2 = %c0 to %c2048 step %c64 { // mainloop gemm_k_iterations
        %16 = memref.subview %11[0, %arg2] [128, 64] [1, 1] : memref<128x2048xf16, affine_map<(d0, d1)[s0] -> (d0 * 2048 + s0 + d1)>> to memref<128x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 2048 + s0 + d1)>>
        %17 = memref.subview %12[%arg2, 0] [64, 128] [1, 1] : memref<2048x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>> to memref<64x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
        gpu.barrier
        memref.copy %16, %1 {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<128x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 2048 + s0 + d1)>> to memref<128x64xf16, 3>
        memref.copy %17, %0 {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<64x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>> to memref<64x128xf16, 3>
        gpu.barrier
        %18 = gpu.thread_id  x
        %19 = gpu.thread_id  y
        %20 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%19]
        scf.for %arg3 = %20 to %c128 step %c128 { // TileAndDistributed over WarpTileM 
          %21 = affine.apply affine_map<(d0) -> ((d0 floordiv 32) * 64)>(%18)
          scf.for %arg4 = %21 to %c128 step %c128 { // TileAndDistributed over WarpTileN
            %22 = memref.subview %1[%arg3, 0] [64, 64] [1, 1] : memref<128x64xf16, 3> to memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>, 3>
            %23 = memref.subview %0[0, %arg4] [64, 64] [1, 1] : memref<64x128xf16, 3> to memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3>
            %24 = memref.subview %2[%arg3, %arg4] [64, 64] [1, 1] : memref<128x128xf16, 3> to memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3>
            linalg.matmul {__internal_linalg_transform__ = "vectorize", lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 64]]>} ins(%22, %23 : memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>, 3>, memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3>) outs(%24 : memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3>)
          }
        }
      }
      gpu.barrier
      memref.copy %2, %10 {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<128x128xf16, 3> to memref<128x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
      gpu.barrier
    }
  }
```
Listing 4. LLVMGPUTileAndDistribute tiles and distribute further to wraps. Notice the introduction of additional `scf.for` loops
tiling and distribution accross warps.


**Key takeaways from **
- Tiles and distributes matmul across warps. See comment "WarpTileM" and "WarpTileN" in the Listing 4.
- These `scf.for` loops are distributed and fully unrolled. These are not seen in the final ptx code.
- Ignorning the `scf.for` on `linalg.fill`, there are 5 `scf.for` loops, 2 distributed/prallelized across CTAs and
two across warps. 
- There is only one `scf.for` loop on the GEMM-K dimension which will be in the IR after the next pass. 
- Notice two `memref.copy` for operandA and operandB that moves the data from global to shared memory.
```mlir
memref.copy %16, %1 {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<128x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 2048 + s0 + d1)>> to memref<128x64xf16, 3>
memref.copy %17, %0 {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<64x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>> to memref<64x128xf16, 3>
```

- `linalg.matmul` is now over a warp tile of 64x64x64, i.e., 2 warps in CtaM and 2 warps in CtaN dimension.
```mlir
linalg.matmul {__internal_linalg_transform__ = "vectorize", lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 64]]>} ins(%22, %23 : memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>, 3>, memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3>) outs(%24 : memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3>)
```
## Remove Distributed (Parallel) `scf.for` Loops over CTAs and Warps within a CTA
```mlir
  // -----// IR Dump After RemoveSingleIterationLoop (iree-codegen-remove-single-iteration-loop) //----- //
  %15 = memref.subview %2[%13, %14] [64, 64] [1, 1] : memref<128x128xf16, 3> to memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3>
  linalg.fill {__internal_linalg_transform__ = "vectorize", lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 64]]>} ins(%cst : f16) outs(%15 : memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3>)
  scf.for %arg0 = %c0 to %c2048 step %c64 {
    %16 = memref.subview %9[0, %arg0] [128, 64] [1, 1] : memref<128x2048xf16, affine_map<(d0, d1)[s0] -> (d0 * 2048 + s0 + d1)>> to memref<128x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 2048 + s0 + d1)>>
    %17 = memref.subview %10[%arg0, 0] [64, 128] [1, 1] : memref<2048x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>> to memref<64x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
    gpu.barrier
    memref.copy %16, %1 {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<128x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 2048 + s0 + d1)>> to memref<128x64xf16, 3>
    memref.copy %17, %0 {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<64x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>> to memref<64x128xf16, 3>
    gpu.barrier
    %18 = gpu.thread_id  x
    %19 = gpu.thread_id  y
    %20 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%19]
    %21 = affine.apply affine_map<(d0) -> ((d0 floordiv 32) * 64)>(%18)
    %22 = memref.subview %1[%20, 0] [64, 64] [1, 1] : memref<128x64xf16, 3> to memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>, 3>
    %23 = memref.subview %0[0, %21] [64, 64] [1, 1] : memref<64x128xf16, 3> to memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3>
    %24 = memref.subview %2[%20, %21] [64, 64] [1, 1] : memref<128x128xf16, 3> to memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3>
    linalg.matmul {__internal_linalg_transform__ = "vectorize", lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 64]]>} ins(%22, %23 : memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>, 3>, memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3>) outs(%24 : memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3>)
  }
```
Listing 5. RemoveSingleIterationLoop keeps only one `scf.for` loop over GEMM-K problem.

## Mutistage Pipelining 
```mlir
// -----// IR Dump After LLVMGPUMultiBuffering (iree-llvmgpu-multi-buffering) //----- //
linalg.fill {__internal_linalg_transform__ = "vectorize", lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 64]]>} ins(%cst : f16) outs(%15 : memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3>)
  scf.for %arg0 = %c0 to %c2048 step %c64 {
    %16 = affine.apply affine_map<(d0, d1, d2) -> (((d0 - d1) floordiv d2) mod 3)>(%arg0, %c0, %c64)
    %17 = memref.subview %1[%16, 0, 0] [1, 128, 64] [1, 1, 1] : memref<3x128x64xf16, 3> to memref<128x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>, 3>
    %18 = affine.apply affine_map<(d0, d1, d2) -> (((d0 - d1) floordiv d2) mod 3)>(%arg0, %c0, %c64)
    %19 = memref.subview %0[%18, 0, 0] [1, 64, 128] [1, 1, 1] : memref<3x64x128xf16, 3> to memref<64x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3>
    %20 = memref.subview %9[0, %arg0] [128, 64] [1, 1] : memref<128x2048xf16, affine_map<(d0, d1)[s0] -> (d0 * 2048 + s0 + d1)>> to memref<128x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 2048 + s0 + d1)>>
    %21 = memref.subview %10[%arg0, 0] [64, 128] [1, 1] : memref<2048x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>> to memref<64x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>
    gpu.barrier
    memref.copy %20, %17 {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<128x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 2048 + s0 + d1)>> to memref<128x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>, 3>
    memref.copy %21, %19 {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<64x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>> to memref<64x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3>
    gpu.barrier
    %22 = gpu.thread_id  x
    %23 = gpu.thread_id  y
    %24 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%23]
    %25 = affine.apply affine_map<(d0) -> ((d0 floordiv 32) * 64)>(%22)
    %26 = memref.subview %17[%24, 0] [64, 64] [1, 1] : memref<128x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>, 3> to memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>, 3>
    %27 = memref.subview %19[0, %25] [64, 64] [1, 1] : memref<64x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3> to memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3>
    %28 = memref.subview %2[%24, %25] [64, 64] [1, 1] : memref<128x128xf16, 3> to memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3>
    linalg.matmul {__internal_linalg_transform__ = "vectorize", lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 64]]>} ins(%26, %27 : memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>, 3>, memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3>) outs(%28 : memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3>)
  }
```
Listing 6. LLVMGPUMultiBuffering stages through shared memory. Notice the 3-dimentional Shared Memory shape 
`memref<3x128x64xf16, 3>`

## MemrefCopyToLinalgPass

`MemrefCopyToLinalgPass` takes the mlir snippet below 
```mlir
    gpu.barrier
    memref.copy %20, %17 {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<128x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 2048 + s0 + d1)>> to memref<128x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>, 3>
    memref.copy %21, %19 {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<64x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>> to memref<64x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3>
    gpu.barrier
```

to

```mlir
    gpu.barrier
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%20 : memref<128x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 2048 + s0 + d1)>>) outs(%17 : memref<128x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>, 3>) attrs =  {__internal_linalg_transform__ = "copy_to_workgroup_memory"} {
    ^bb0(%arg1: f16, %arg2: f16):
      linalg.yield %arg1 : f16
    }
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%21 : memref<64x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>) outs(%19 : memref<64x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3>) attrs =  {__internal_linalg_transform__ = "copy_to_workgroup_memory"} {
    ^bb0(%arg1: f16, %arg2: f16):
      linalg.yield %arg1 : f16
    }
    gpu.barrier
```
Listing 7.


## GPUDistributeSharedMemoryCopy
`GPUDistributeSharedMemoryCopy` takes linalg.generic copy on memeref to vectors distributing the copies over threads. 
Each thread issues vector.transfer_read from Global memory and vector.transfer_write to Global memory.

memref (GMEM) ----vector.transfer_read--> vector (Registers) ---vector.transfer_write--> memref (SMEM)

For SM80 cp.async vector registers will optimized out as cp.async can issue direct copy from GMEM to SMEM.

```mlir
// -----// IR Dump After GPUDistributeSharedMemoryCopy (iree-gpu-distribute-shared-memory-copy) //----- //
    %159 = vector.transfer_read %104[%157, %158], %cst {in_bounds = [true, true]} : memref<64x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>>, vector<1x8xf16>
    ...
    vector.transfer_write %147, %102[%169, %170] {in_bounds = [true, true]} : vector<1x8xf16>, memref<64x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3>
    gpu.barrier
    %185 = gpu.thread_id  x
    %186 = gpu.thread_id  y
    %187 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%186]
    %188 = affine.apply affine_map<(d0) -> ((d0 floordiv 32) * 64)>(%185)
    %189 = memref.subview %100[%187, 0] [64, 64] [1, 1] : memref<128x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>, 3> to memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>, 3>
    %190 = memref.subview %102[0, %188] [64, 64] [1, 1] : memref<64x128xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3> to memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3>
    %191 = memref.subview %5[%187, %188] [64, 64] [1, 1] : memref<128x128xf16, 3> to memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3>
    linalg.matmul {__internal_linalg_transform__ = "vectorize", lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 64]]>} ins(%189, %190 : memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>, 3>, memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3>) outs(%191 : memref<64x64xf16, affine_map<(d0, d1)[s0] -> (d0 * 128 + s0 + d1)>, 3>) 
```
Listing 8. Part of the mainloop after `GPUDistributeSharedMemoryCopy` linalg.matmul above is warp-level matmul.

After the mainloop, there are vector.transfer_read and vector.transfer_write (WHAT ARE THOSE FOR?)

```mlir
gpu.barrier
 %17 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 4 + s2 * 8 + s0 floordiv 16)>()[%0, %1, %2]
 %18 = affine.apply affine_map<()[s0] -> (s0 * 8 - (s0 floordiv 16) * 128)>()[%0]
 %19 = vector.transfer_read %5[%17, %18], %cst {in_bounds = [true, true]} : memref<128x128xf16, 3>, vector<1x8xf16>
 %20 = affine.apply affine_map<()[s0, s1, s2] -> (s1 * 4 + s2 * 8 + s0 floordiv 16 + 8)>()[%0, %1, %2]
    
 ```

## LLVMGPUReduceBankConflicts
`LLVMGPUReduceBankConflicts` only does padding and nother related to Shared Memory swizzle. 

## WorkGroupSwizzle
Threadblock swizzle and L2 reuse.


## LLVMGPUTensorCoreVectorization
`LLVMGPUTensorCoreVectorization` vectorizes `linalg.matmul`. It takes `linalg.matmul` to `vector.contract`.

```mlir
    %254 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %126, %158, %222 : vector<16x8xf16>, vector<8x8xf16> into vector<16x8xf16>
```

```
mainloop:
...
  vector.transfer_reads
  vector.transfer_reads
  vector.contract
  vector.transfer_write // WHY ARE WE WRITING BACK ACCUMULATORS TO SHARED MEMORY inside the MAINLOOP?
```

## OptimizeVectorTransfer
`OptimizeVectorTransfer` does the following:
- Hoist vector.transfer_read on C/Accumulators above the mainloop.
- Hoist vector.transfer_write on C/Accumulators below the mainloop.
- Christopher Bates is having issues and discussion points on hoisting theses and the application of bufferization pass.


## FoldSubViewOps
`FoldSubViewOps` removes subview and folds the arithmetic into the user of the instruction.

## LLVMGPUVectorToGPU
`LLVMGPUVectorToGPU` can be considered as `LLVMGPUVectorToGPU[NVGPU]`. We haven't yet added NVGPU to the pass name yet.
NVGPU dialect starts here and we are transforming from `vector` to `nvgpu` dialect.

```mlir
%479 = nvgpu.ldmatrix %4[%475, %477, %478] {numTiles = 2 : i32, transpose = false} : memref<3x128x72xf16, 3> -> vector<2x2xf16>
%824 = nvgpu.mma.sync(%472, %822, %arg33) {mmaShape = [16, 8, 8]} : (vector<2x2xf16>, vector<1x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
```

## GPUPipelining
`GPUPipelining` applies software pipelining.
- Software pipelining is applied here in GPUPipelining pass.
- The `LLVMGPUMultiBuffering` only creates multiple stages of Shared Memory, but doesn't interleave ldsm, mma.sync, and cp.sync.
- `GPUPipelining` interleaves the instructions and creates optimal instructions schedule for the mainloop. 


## LLVMGPUVectorLowering
Preparing the lowering form `nvgpu` to `nnvm`




