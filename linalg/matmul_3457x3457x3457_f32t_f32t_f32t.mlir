// Problem size      : 3457x3457x3457
// matrixA           : F32, RowMajor
// matrixB           : F32, RowMajor
// Accumulation type : F32
// matrixC           : F32, RowMajor

func.func @matmul_3457x3457x3457_f32t_f32t_f32t() {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<3457x3457xf32>
  %rhs = util.unfoldable_constant dense<0.4> : tensor<3457x3457xf32>
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<3457x3457xf32>
  %CC = linalg.fill ins(%c0 : f32) outs(%init : tensor<3457x3457xf32>) -> tensor<3457x3457xf32>
  %D = linalg.matmul ins(%lhs, %rhs: tensor<3457x3457xf32>, tensor<3457x3457xf32>)
                     outs(%CC: tensor<3457x3457xf32>) -> tensor<3457x3457xf32>
  check.expect_almost_eq_const(%D, dense<204.800> : tensor<3457x3457xf32>) : tensor<3457x3457xf32>
  return
}