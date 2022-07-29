func.func @m16n8k4_tf32(%arg0: vector<2x1xf32>, %arg1: vector<1x1xf32>, %arg2: vector<2x2xf32>) -> vector<2x2xf32> {  
  // The A, B operand should be bitcast to i32
  // CHECK: llvm.extractvalue
  // CHECK: llvm.bitcast {{.*}} : vector<1xf32> to i32  
  // CHECK: llvm.extractvalue
  // CHECK: llvm.bitcast {{.*}} : vector<1xf32> to i32
  // CHECK: llvm.extractvalue
  // CHECK: llvm.bitcast {{.*}} : vector<1xf32> to i32

  // CHECK: [[d:%.+]] = nvvm.mma.sync A[{{%.+}}, {{%.+}}] B[{{%.+}}] C[{{%.+}}, {{%.+}}, {{%.+}}, {{%.+}}]
  // CHECK-SAME: multiplicandAPtxType = #nvvm.mma_type<tf32>
  // CHECK-SAME: multiplicandBPtxType = #nvvm.mma_type<tf32>
  // CHECK-SAME: shape = #nvvm.shape<m = 16, n = 8, k = 4>
  // CHECK-SAME: -> !llvm.struct<(f32, f32, f32, f32)>  
  %d = nvgpu.mma.sync (%arg0, %arg1, %arg2) {mmaShape = [16, 8, 4]} : (vector<2x1xf32>, vector<1x1xf32>, vector<2x2xf32>) -> vector<2x2xf32>  
  // CHECK: [[el:%.+]] = llvm.extractvalue [[d]][0]
  // CHECK: llvm.bitcast [[el]] : f32 to vector<1xf32>
  // CHECK: [[el:%.+]] = llvm.extractvalue [[d]][1]
  // CHECK: llvm.bitcast [[el]] : f32 to vector<1xf32>
  // CHECK: [[el:%.+]] = llvm.extractvalue [[d]][2]
  // CHECK: llvm.bitcast [[el]] : f32 to vector<1xf32>
  // CHECK: [[el:%.+]] = llvm.extractvalue [[d]][3]
  // CHECK: llvm.bitcast [[el]] : f32 to vector<1xf32>
  // CHECK-COUNT-4: llvm.insertvalue {{.*}} : !llvm.array<4 x vector<1xf32>>
  return %d : vector<2x2xf32>
}
