// Problem size      : 3456x1024x2048
// matrixA           : F32, RowMajor
// matrixB           : F32, RowMajor
// Accumulation type : F32
// matrixC           : F32, RowMajor

#compilation_trait = #iree_codegen.compilation_info<
  lowering_config = <tile_sizes = [[1, 128, 128, 16]]>,
  translation_info = <LLVMGPUMatmulTensorCore
  pipeline_depth = 4>,
  workgroup_size = [64 : index, 2 : index, 1 : index]>

func.func @matmul_3456x1024x2048_f32t_f32t_f32t() {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<3456x2048xf32>
  %rhs = util.unfoldable_constant dense<0.4> : tensor<2048x1024xf32>
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<3456x1024xf32>
  %CC = linalg.fill ins(%c0 : f32) outs(%init : tensor<3456x1024xf32>) -> tensor<3456x1024xf32>
  %D = linalg.matmul {compilation_info = #compilation_trait} ins(%lhs, %rhs: tensor<3456x2048xf32>, tensor<2048x1024xf32>) outs(%CC: tensor<3456x1024xf32>) -> tensor<3456x1024xf32>
  check.expect_almost_eq_const(%D, dense<819.000> : tensor<3456x1024xf32>) : tensor<3456x1024xf32>
  return
}
