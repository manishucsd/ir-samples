// MxNxK = 3456x1024x2048
func.func @matmul_f32_f32() {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<3456x2048xf32>
  %rhs = util.unfoldable_constant dense<0.4> : tensor<2048x1024xf32>
  %c0 = arith.constant 0.0 : f32
  %init = linalg.init_tensor[3456, 1024] : tensor<3456x1024xf32>
  %CC = linalg.fill ins(%c0 : f32) outs(%init : tensor<3456x1024xf32>) -> tensor<3456x1024xf32>
  %D = linalg.matmul ins(%lhs, %rhs: tensor<3456x2048xf32>, tensor<2048x1024xf32>)
                    outs(%CC: tensor<3456x1024xf32>) -> tensor<3456x1024xf32>
  check.expect_almost_eq_const(%D, dense<819.000> : tensor<3456x1024xf32>) : tensor<3456x1024xf32>
  return
}
