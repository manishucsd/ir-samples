// Problem size      : 513x513x513
// matrixA           : F32, RowMajor
// matrixB           : F32, RowMajor
// Accumulation type : F32
// matrixC           : F32, RowMajor

func.func @matmul_513x513x513_f32t_f32t_f32t() {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<513x513xf32>
  %rhs = util.unfoldable_constant dense<0.4> : tensor<513x513xf32>
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<513x513xf32>
  %CC = linalg.fill ins(%c0 : f32) outs(%init : tensor<513x513xf32>) -> tensor<513x513xf32>
  %D = linalg.matmul ins(%lhs, %rhs: tensor<513x513xf32>, tensor<513x513xf32>)
                     outs(%CC: tensor<513x513xf32>) -> tensor<513x513xf32>
  check.expect_almost_eq_const(%D, dense<205.200> : tensor<513x513xf32>) : tensor<513x513xf32>
  return
}