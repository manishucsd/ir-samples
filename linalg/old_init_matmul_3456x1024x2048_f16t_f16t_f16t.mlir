// old iree-benchmark-module commandline: ./tools/iree-benchmark-module --module_file=mma_sync_old_init_matmul_3456x1024x2048_f16t_f16t_f16t.vmfb --device=cuda --batch_size=100 --entry_function=matmul_3456x1024x2048_f16t_f16t_f16t --benchmark_repetitions=5

// Problem size      : 3456x1024x2048
// matrixA           : F16, RowMajor
// matrixB           : F16, RowMajor
// Accumulation type : F16
// matrixC           : F16, RowMajor

#compilation_trait = #iree_codegen.compilation_info<
  lowering_config = <tile_sizes = [[128, 128, 64]]>,
  translation_info = <LLVMGPUMatmulTensorCore
  pipeline_depth = 4>,
  workgroup_size = [64 : index, 2 : index, 1 : index]
>

func.func @matmul_3456x1024x2048_f16t_f16t_f16t() {
  %lhs = util.unfoldable_constant dense<1.00> : tensor<3456x2048xf16>
  %rhs = util.unfoldable_constant dense<0.01> : tensor<2048x1024xf16>
  %c0 = arith.constant 0.0 : f16
  %init = linalg.init_tensor[3456, 1024] : tensor<3456x1024xf16>
  %CC = linalg.fill ins(%c0 : f16) outs(%init : tensor<3456x1024xf16>) -> tensor<3456x1024xf16>
  %D = linalg.matmul {compilation_info = #compilation_trait} 
                     ins(%lhs, %rhs: tensor<3456x2048xf16>, tensor<2048x1024xf16>)
                     outs(%CC: tensor<3456x1024xf16>) -> tensor<3456x1024xf16>
  check.expect_almost_eq_const(%D, dense<20.2812> : tensor<3456x1024xf16>) : tensor<3456x1024xf16>
  return
}