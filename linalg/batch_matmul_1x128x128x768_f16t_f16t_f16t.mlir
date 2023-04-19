#tile_config_64x64_64x5_tensorcore_mmasync = #iree_codegen.compilation_info<
  lowering_config = <tile_sizes = [[1, 64, 64, 64]]>,
  translation_info = <LLVMGPUMatmulTensorCoreMmaSync pipeline_depth = 5>,
  workgroup_size = [64 : index, 2 : index, 1 : index]
>

// Dispatch linalg.matmul row-row layout 
func.func @batch_matmul_1x128x128x768_f16t_f16t_f16t_tile_config_64x64_64x5_tensorcore_mmasync(
  %lhs: tensor<1x128x768xf16>,
  %rhs: tensor<1x768x128xf16>) -> tensor<1x128x128xf16>
{
  %c0 = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<1x128x128xf16>
  %inital_result = linalg.fill ins(%c0 : f16) outs(%init : tensor<1x128x128xf16>) -> tensor<1x128x128xf16>
  %result = linalg.matmul {compilation_info = #tile_config_64x64_64x5_tensorcore_mmasync} 
                     ins(%lhs, %rhs: tensor<1x128x768xf16>, tensor<1x768x128xf16>)
                     outs(%inital_result: tensor<1x128x128xf16>) -> tensor<1x128x128xf16>
  return %result : tensor<1x128x128xf16>
}
