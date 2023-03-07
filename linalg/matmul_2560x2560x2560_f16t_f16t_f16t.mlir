// Problem size      : 2560x2560x2560
// matrixA           : F16, RowMajor
// matrixB           : F16, RowMajor
// Accumulation type : F16
// matrixC           : F16, RowMajor

#compilation_trait_tile_128x256_32x3 = #iree_codegen.compilation_info<
  lowering_config = <tile_sizes = [[128, 256, 32]]>,
  translation_info = <LLVMGPUMatmulTensorCore
  pipeline_depth = 3>,
  workgroup_size = [128 : index, 2 : index, 1 : index]
>


func.func @matmul() -> tensor<2560x2560xf16> {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<2560x2560xf16>
  %rhs = util.unfoldable_constant dense<0.4> : tensor<2560x2560xf16>
  %c0 = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<2560x2560xf16>
  %CC = linalg.fill ins(%c0 : f16) outs(%init : tensor<2560x2560xf16>) -> tensor<2560x2560xf16>
  %D = linalg.matmul {compilation_info = #compilation_trait_tile_128x256_32x3}
                    ins(%lhs, %rhs: tensor<2560x2560xf16>, tensor<2560x2560xf16>)
                    outs(%CC: tensor<2560x2560xf16>) -> tensor<2560x2560xf16>
  return %D : tensor<2560x2560xf16>
}