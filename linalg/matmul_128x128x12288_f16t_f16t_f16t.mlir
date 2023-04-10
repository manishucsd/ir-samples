#tile_config_64x64_64x5_tensorcore_mmasync = #iree_codegen.compilation_info<
  lowering_config = <tile_sizes = [[64, 64, 64]]>,
  translation_info = <LLVMGPUMatmulTensorCoreMmaSync pipeline_depth = 5>,
  workgroup_size = [64 : index, 2 : index, 1 : index]
>

func.func @matmul_128x128x12288_f16t_f16t_f16t_tile_config_64x64_64x5_tensorcore_mmasync(
  %lhs: tensor<128x12288xf16>,
  %rhs: tensor<12288x128xf16>) -> tensor<128x128xf16>
{
  %c0 = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<128x128xf16>
  %inital_result = linalg.fill ins(%c0 : f16) outs(%init : tensor<128x128xf16>) -> tensor<128x128xf16>
  %result = linalg.matmul {compilation_info = #tile_config_64x64_64x5_tensorcore_mmasync} 
                     ins(%lhs, %rhs: tensor<128x12288xf16>, tensor<12288x128xf16>)
                     outs(%inital_result: tensor<128x128xf16>) -> tensor<128x128xf16>
  return %result : tensor<128x128xf16>
}
