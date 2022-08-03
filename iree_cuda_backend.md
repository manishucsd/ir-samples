**IREE CUDA Backend linalg.matmul as the input IR to PTX**

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
- Note that there is not split-k-slice used in this example, but split-k-slice is supported by IREE.

```mlir
%11 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[128, 128, 64]]>} ins(%7, %8 : tensor<128x2048xf16>, tensor<2048x128xf16>) outs(%10 : tensor<128x128xf16>) -> tensor<128x128xf16>
```

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
- Why tensor to memrefs?
- memref.subview creates the view of intput operands A and B and output C for single tiled-iteration 128x128x64.

TODO: FILL what is the usage of bufferization pass



