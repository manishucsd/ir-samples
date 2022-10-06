#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @m16n8k16_mmasync16816_fp16_f16(%arg0: memref<42x32xf16, 3>, %arg1: memref<32x64xf16, 3>, %arg2: memref<42x64xf16, 3>) {
  %cst_0 = arith.constant dense<0.000000e+00> : vector<16x8xf16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %cst = arith.constant 0.000000e+00 : f16
  %A = vector.transfer_read %arg0[%c1, %c3], %cst {in_bounds = [true, true]} : memref<42x32xf16, 3>, vector<16x16xf16>
  %B = vector.transfer_read %arg1[%c3, %c3], %cst {permutation_map = #map0, in_bounds = [true, true]} : memref<32x64xf16, 3>, vector<8x16xf16>
  %C = vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<42x64xf16, 3>, vector<16x8xf16>
  %D = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %A, %B, %C : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf16>
  vector.transfer_write %D, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<16x8xf16>, memref<42x64xf16, 3>
  return
}