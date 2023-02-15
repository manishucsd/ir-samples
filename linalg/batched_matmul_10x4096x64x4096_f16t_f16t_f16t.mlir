// Problem size      : 10x4096x64x4096
// Input type        : F16
// Accumulation type : F16 
// Output type       : F16

#compilation_trait_wmma = #iree_codegen.compilation_info<
  lowering_config = <tile_sizes = [[1, 128, 64, 128]]>,
  translation_info = <LLVMGPUMatmulTensorCore
  pipeline_depth = 2>,
  workgroup_size = [64 : index, 2 : index, 1 : index]>

#compilation_trait_mma_sync = #iree_codegen.compilation_info<
  lowering_config = <tile_sizes = [[1, 128, 64, 64]]>,
  translation_info = <LLVMGPUMatmulTensorCore
  pipeline_depth = 4>,
  workgroup_size = [64 : index, 2 : index, 1 : index]>

func.func @batched_matmul_10x4096x64x4096_f16t_f16t_f16t(
    %A : tensor<10x4096x4096xf16>,
    %B : tensor<10x4096x64xf16>) -> tensor<10x4096x64xf16> {
    %c0 = arith.constant 0.0 : f16
    %init = tensor.empty() : tensor<10x4096x64xf16>
    %C = linalg.fill ins(%c0 : f16) outs(%init : tensor<10x4096x64xf16>) -> tensor<10x4096x64xf16>
    %D = linalg.batch_matmul {compilation_info = #compilation_trait_mma_sync} 
            ins(%A, %B : tensor<10x4096x4096xf16>, tensor<10x4096x64xf16>) 
            outs(%C : tensor<10x4096x64xf16>) -> tensor<10x4096x64xf16>
    return %D : tensor<10x4096x64xf16>
}