func.func @warpgroup_mma_128_128_64(
      %descA: !nvgpu.warpgroup.descriptor<tensor = memref<128x64xf16, 3>>, 
      %descB: !nvgpu.warpgroup.descriptor<tensor = memref<64x128xf16, 3>>, 
      %acc: !nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>>) 
{
  %wgmmaResult = nvgpu.warpgroup.mma %descA, %descB, %acc {transposeB}: 
      !nvgpu.warpgroup.descriptor<tensor = memref<128x64xf16, 3>>, 
      !nvgpu.warpgroup.descriptor<tensor = memref<64x128xf16, 3>>, 
      !nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>> 
      -> 
      !nvgpu.warpgroup.accumulator<fragmented = vector<128x128xf32>>  
  return
}