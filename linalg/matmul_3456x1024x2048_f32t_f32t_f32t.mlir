// Problem size      : 3456x1024x2048
// matrixA           : F32, RowMajor
// matrixB           : F32, RowMajor
// Accumulation type : F32
// matrixC           : F32, RowMajor

// Finename: ./generated/linalg/matmul/matmul_3456x1024x2048_f32t_f32t_f32t/matmul_3456x1024x2048_f32t_f32t_f32t.mlir
// matmul compilation info (tile configuration, translation info, workgroup size)
#tile_config_128x128_16x5_tensorcore_mma_sync = #iree_codegen.compilation_info<
  lowering_config = <tile_sizes = [[128, 128, 16]]>,
  translation_info = <LLVMGPUMatmulTensorCoreMmaSync pipeline_depth = 5>,
  workgroup_size = [64 : index, 2 : index, 1 : index]
>

// Dispatch linalg.matmul row-row layout 
func.func @matmul_3456x1024x2048_f32t_f32t_f32t_tile_config_128x128_16x5_tensorcore_mma_sync(
  %lhs: tensor<3456x2048xf32>,
  %rhs: tensor<2048x1024xf32>) -> tensor<3456x1024xf32>
{
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<3456x1024xf32>
  %inital_result = linalg.fill ins(%c0 : f32) outs(%init : tensor<3456x1024xf32>) -> tensor<3456x1024xf32>
  %result = linalg.matmul {compilation_info = #tile_config_128x128_16x5_tensorcore_mma_sync} 
                     ins(%lhs, %rhs: tensor<3456x2048xf32>, tensor<2048x1024xf32>)
                     outs(%inital_result: tensor<3456x1024xf32>) -> tensor<3456x1024xf32>
  return %result : tensor<3456x1024xf32>
}
