func.func @m16n8k4_tf32(%arg0: vector<2x1xf32>, %arg1: vector<1x1xf32>, %arg2: vector<4x1xf32>) -> vector<4x1xf32> {  
  
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 4]} : (vector<2x1xf32>, vector<1x1xf32>, vector<4x1xf32>) -> vector<4x1xf32>  

  return %d : vector<4x1xf32>
}
