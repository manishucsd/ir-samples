func.func @vector_dot_product(%vector_a: tensor<4xf32>, %vector_b: tensor<4xf32>, %out: tensor<f32>) -> tensor<f32> {
  %dot = linalg.dot ins(%vector_a, %vector_b : tensor<4xf32>, tensor<4xf32>)
                    outs(%out : tensor<f32>) -> tensor<f32>
  return %dot : tensor<f32>
}