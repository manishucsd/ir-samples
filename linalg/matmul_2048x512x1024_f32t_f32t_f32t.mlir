func.func @matmul_f32_f32() {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<2048x1024xf32>
  %rhs = util.unfoldable_constant dense<0.4> : tensor<1024x512xf32>
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<2048x512xf32>
  %CC = linalg.fill ins(%c0 : f32) outs(%init : tensor<2048x512xf32>) -> tensor<2048x512xf32>
  %D = linalg.matmul ins(%lhs, %rhs: tensor<2048x1024xf32>, tensor<1024x512xf32>)
                    outs(%CC: tensor<2048x512xf32>) -> tensor<2048x512xf32>
  check.expect_almost_eq_const(%D, dense<409.596> : tensor<2048x512xf32>) : tensor<2048x512xf32>
  return
}
