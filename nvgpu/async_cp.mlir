// -----

func.func @async_cp(
  %src: memref<128x128xf32>, %dst: memref<3x16x128xf32, 3>, %i : index, %loadElements : index) {
  %0 = nvgpu.device_async_copy %src[%i, %i], %dst[%i, %i, %i], 4 : memref<128x128xf32> to memref<3x16x128xf32, 3>
  %1 = nvgpu.device_async_create_group %0
  nvgpu.device_async_wait %1 { numGroups = 1 : i32 }
  %2 = nvgpu.device_async_copy %src[%i, %i], %dst[%i, %i, %i], 4 {bypassL1}: memref<128x128xf32> to memref<3x16x128xf32, 3>
  return
}

// -----
