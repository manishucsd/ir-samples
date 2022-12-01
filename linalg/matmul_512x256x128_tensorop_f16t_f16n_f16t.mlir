#compilation_trait = #iree_codegen.compilation_info<
  lowering_config = <tile_sizes = [[32, 32, 32]]>,
  translation_info = <LLVMGPUMatmulTensorCore
  pipeline_depth = 3>,
  workgroup_size = [64 : index, 1 : index, 1 : index]>

#matmul_trait = {
  compilation_info = #compilation_trait, 
  args_in = 2,
  args_out = 1,
  indexing_maps = [
    affine_map<(m, n, k) -> (m, k)>,
    affine_map<(m, n, k) -> (n, k)>,
    affine_map<(m, n, k) -> (m, n)>
  ],
  iterator_types = ["parallel", "parallel", "reduction"]
}

func.func @matmul_512x256x128_tensorop_f16t_f16n_f16t(
                          %A: tensor<512x128xf16>, 
                          %B: tensor<256x128xf16>,
                          %C: tensor<512x256xf16>) -> tensor<512x256xf16>
{
  %D = linalg.generic #matmul_trait
    ins(%A, %B : tensor<512x128xf16>, tensor<256x128xf16>)
    outs(%C : tensor<512x256xf16>) {
    ^bb(%a: f16, %b: f16, %c: f16) :
      %d = arith.mulf %a, %b: f16
      %e = arith.addf %c, %d: f16
      linalg.yield %e : f16
  } -> tensor<512x256xf16>
  return %D : tensor<512x256xf16>
}