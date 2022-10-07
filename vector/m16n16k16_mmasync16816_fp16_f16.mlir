#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @m16n16k16_mmasync16816_fp16_f16(%arg0: memref<42x32xf16, 3>, %arg1: memref<32x64xf16, 3>, %arg2: memref<42x64xf16, 3>) {
  %cst_0 = arith.constant dense<0.000000e+00> : vector<16x8xf16>
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %cst = arith.constant 0.000000e+00 : f16
  %A = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<42x32xf16, 3>, vector<16x16xf16>
  %B = vector.transfer_read %arg1[%c0, %c0], %cst {permutation_map = #map0, in_bounds = [true, true]} : memref<32x64xf16, 3>, vector<16x16xf16>
  %C = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<42x64xf16, 3>, vector<16x16xf16>

  // %D0_16x8 = mma.sync16816 %A_16x16_row, %B0_8x16_col, %C0_16x8_row 
  %B0 = vector.extract_strided_slice %B {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
  %C0 = vector.extract_strided_slice %C {offsets = [0, 0], sizes = [16, 8], strides = [1, 1]} : vector<16x16xf16> to vector<16x8xf16>
  %D0 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B0, %C0 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
  vector.transfer_write %D0, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<42x64xf16, 3>

  // %D1_16x8 = mma.sync16816 %A_16x16_row, %B1_8x16_col, %C1_16x8_row 
  %B1 = vector.extract_strided_slice %B {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
  %C1 = vector.extract_strided_slice %C {offsets = [0, 8], sizes = [16, 8], strides = [1, 1]} : vector<16x16xf16> to vector<16x8xf16>
  %D1 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B1, %C1 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
  vector.transfer_write %D1, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<42x64xf16, 3>

  return
}