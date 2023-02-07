// Problem size      : 3456x1024x2048
// Input type        : F16
// Accumulation type : F16
// Output type       : F16
func.func @matmul_3456x1024x2048_f16_f16() {
  %lhs = util.unfoldable_constant dense<1.00> : tensor<3456x2048xf16>
  %rhs = util.unfoldable_constant dense<0.01> : tensor<2048x1024xf16>
  %c0 = arith.constant 0.0 : f16
  %init = tensor.empty() : tensor<3456x1024xf16>
  %CC = linalg.fill ins(%c0 : f16) outs(%init : tensor<3456x1024xf16>) -> tensor<3456x1024xf16>
  %D = linalg.matmul ins(%lhs, %rhs: tensor<3456x2048xf16>, tensor<2048x1024xf16>)
                    outs(%CC: tensor<3456x1024xf16>) -> tensor<3456x1024xf16>
  check.expect_almost_eq_const(%D, dense<20.2812> : tensor<3456x1024xf16>) : tensor<3456x1024xf16>
  return
}