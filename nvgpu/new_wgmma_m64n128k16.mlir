func.func @wgmma_m64n128k16(
      %descA: i64, 
      %descB: i64, 
      %matrixC: vector<16x2x2xf32>) -> vector<16x2x2xf32>
{
  %matrixD1 = nvgpu.warpgroup.mma (%descA, %descB, %matrixC) {typeA = f16, layoutA = #nvgpu.matrix_layout<row>, typeB = f16, layoutB = #nvgpu.matrix_layout<col>, wgmmaShape = [64, 128, 16]} : i64, i64, vector<16x2x2xf32> -> vector<16x2x2xf32>
  %matrixD = nvgpu.warpgroup.mma (%descA, %descB, %matrixD1) {typeA = f16, layoutA = #nvgpu.matrix_layout<row>, typeB = f16, layoutB = #nvgpu.matrix_layout<col>, wgmmaShape = [64, 128, 16]} : i64, i64, vector<16x2x2xf32> -> vector<16x2x2xf32>

  return %matrixD : vector<16x2x2xf32>
}
