#tile_config_32x32_16x4_tensorcore_mmasync = #iree_codegen.compilation_info<
  lowering_config = <tile_sizes = [[1, 32, 32, 16]]>,
  translation_info = <LLVMGPUMatmulTensorCoreMmaSync pipeline_depth = 4>,
  workgroup_size = [64 : index, 2 : index, 1 : index]
>

// Dispatch linalg.matmul row-row layout 
func.func @transpose_batch_matmul_16x384x64x384_f32t_f32t_f32t(
  %lhs: tensor<16x384x384xf32>,
  %rhs: tensor<16x384x64xf32>) -> tensor<384x16x64xf32>
{
  %c0 = arith.constant 0.0 : f32
  %init           = tensor.empty() : tensor<16x384x64xf32>
  %init_tranposed = tensor.empty() : tensor<384x16x64xf32>
  %inital_result = linalg.fill ins(%c0 : f32) outs(%init : tensor<16x384x64xf32>) -> tensor<16x384x64xf32>
  %result = linalg.batch_matmul {compilation_info = #tile_config_32x32_16x4_tensorcore_mmasync} 
                     ins(%lhs, %rhs: tensor<16x384x384xf32>, tensor<16x384x64xf32>)
                     outs(%inital_result: tensor<16x384x64xf32>) -> tensor<16x384x64xf32>
  %result_transposed = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d1, d0, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%result : tensor<16x384x64xf32>) outs(%init_tranposed : tensor<384x16x64xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<384x16x64xf32>
  return %result_transposed : tensor<384x16x64xf32>
}


