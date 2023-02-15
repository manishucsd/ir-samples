// Problem size      : 2049x2049x2049
// matrixA           : F32, RowMajor
// matrixB           : F32, RowMajor
// Accumulation type : F32
// matrixC           : F32, RowMajor

func.func @matmul_2049x2049x2049_f32t_f32t_f32t() {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<2049x2049xf32>
  %rhs = util.unfoldable_constant dense<0.4> : tensor<2049x2049xf32>
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<2049x2049xf32>
  %CC = linalg.fill ins(%c0 : f32) outs(%init : tensor<2049x2049xf32>) -> tensor<2049x2049xf32>
  %D = linalg.matmul ins(%lhs, %rhs: tensor<2049x2049xf32>, tensor<2049x2049xf32>)
                     outs(%CC: tensor<2049x2049xf32>) -> tensor<2049x2049xf32>
  check.expect_almost_eq_const(%D, dense<204.800> : tensor<2049x2049xf32>) : tensor<2049x2049xf32>
  return
}