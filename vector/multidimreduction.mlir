func.func @multidimreduction(
  %arg0: vector<8x32x16xf32>,%arg1: vector<8x32x16xf32>, %acc: vector<8x16xf32>) -> vector<8x16xf32> {
  %0 = arith.mulf %arg0, %arg1 : vector<8x32x16xf32>
  %1 = vector.multi_reduction <add>, %0, %acc [1] : vector<8x32x16xf32> to vector<8x16xf32>
  return %1 : vector<8x16xf32>
}
