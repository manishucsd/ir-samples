module {
  func.func @async_cp(%arg0: memref<128x128xf32>, %arg1: memref<3x16x128xf32, 3>, %arg2: index, %arg3: index) {
    %0 = builtin.unrealized_conversion_cast %arg1 : memref<3x16x128xf32, 3> to !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<3 x i64>, array<3 x i64>)>
    %1 = builtin.unrealized_conversion_cast %arg2 : index to i64
    %2 = builtin.unrealized_conversion_cast %arg0 : memref<128x128xf32> to !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %3 = builtin.unrealized_conversion_cast %arg3 : index to i64
    %4 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<3 x i64>, array<3 x i64>)> 
    %5 = llvm.mlir.constant(2048 : index) : i64
    %6 = llvm.mul %1, %5  : i64
    %7 = llvm.mlir.constant(128 : index) : i64
    %8 = llvm.mul %1, %7  : i64
    %9 = llvm.add %6, %8  : i64
    %10 = llvm.add %9, %1  : i64
    %11 = llvm.getelementptr %4[%10] : (!llvm.ptr<f32, 3>, i64) -> !llvm.ptr<f32, 3>
    %12 = llvm.bitcast %11 : !llvm.ptr<f32, 3> to !llvm.ptr<i8, 3>
    %13 = llvm.extractvalue %2[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.mlir.constant(128 : index) : i64
    %15 = llvm.mul %1, %14  : i64
    %16 = llvm.add %15, %1  : i64
    %17 = llvm.getelementptr %13[%16] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %18 = llvm.bitcast %17 : !llvm.ptr<f32> to !llvm.ptr<i8>
    %19 = llvm.addrspacecast %18 : !llvm.ptr<i8> to !llvm.ptr<i8, 1>
    %20 = llvm.mlir.constant(16 : i32) : i32
    %21 = llvm.mlir.constant(3 : i32) : i32
    %22 = llvm.mlir.constant(32 : i32) : i32
    %23 = llvm.trunc %3 : i64 to i32
    %24 = llvm.mul %22, %23  : i32
    %25 = llvm.lshr %24, %21  : i32
    %26 = llvm.inline_asm has_side_effects asm_dialect = att "cp.async.ca.shared.global [%0], [%1], %2, %3;\0A :: r($0), l($1), n($2), r($3));", "" %19, %12, %20, %25 : (!llvm.ptr<i8, 1>, !llvm.ptr<i8, 3>, i32, i32) -> !llvm.void
    %27 = llvm.mlir.constant(0 : i32) : i32
    nvvm.cp.async.commit.group
    %28 = llvm.mlir.constant(0 : i32) : i32
    nvvm.cp.async.wait.group 1
    %29 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<3 x i64>, array<3 x i64>)> 
    %30 = llvm.mlir.constant(2048 : index) : i64
    %31 = llvm.mul %1, %30  : i64
    %32 = llvm.mlir.constant(128 : index) : i64
    %33 = llvm.mul %1, %32  : i64
    %34 = llvm.add %31, %33  : i64
    %35 = llvm.add %34, %1  : i64
    %36 = llvm.getelementptr %29[%35] : (!llvm.ptr<f32, 3>, i64) -> !llvm.ptr<f32, 3>
    %37 = llvm.bitcast %36 : !llvm.ptr<f32, 3> to !llvm.ptr<i8, 3>
    %38 = llvm.extractvalue %2[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.mlir.constant(128 : index) : i64
    %40 = llvm.mul %1, %39  : i64
    %41 = llvm.add %40, %1  : i64
    %42 = llvm.getelementptr %38[%41] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %43 = llvm.bitcast %42 : !llvm.ptr<f32> to !llvm.ptr<i8>
    %44 = llvm.addrspacecast %43 : !llvm.ptr<i8> to !llvm.ptr<i8, 1>
    nvvm.cp.async.shared.global %37, %44, 16 {bypass_l1}
    %45 = llvm.mlir.constant(0 : i32) : i32
    return
  }
}

