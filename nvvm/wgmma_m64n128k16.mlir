func.func @wgmma_m64n128k16(
      %descA: i64, 
      %descB: i64, 
      %matrixC: vector<16x2x2xf32>) -> vector<16x2x2xf32>
{   
    // How are builtin.unrealized_conversion_cast lowered? 
    %0 = builtin.unrealized_conversion_cast %matrixC : vector<16x2x2xf32> to !llvm.array<16 x !llvm.array<2 x vector<2xf16>>>
    %llvm_undef = llvm.mlir.undef : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
    
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64

    %s0 = llvm.extractvalue %0[0] : !llvm.array<16 x !llvm.array<2 x vector<2xf16>>>
    %m0 = llvm.extractvalue %s0[0] : !llvm.array<2 x vector<2xf16>>
    %m1 = llvm.extractvalue %s0[1] : !llvm.array<2 x vector<2xf16>>
    
    %s1 = llvm.extractvalue %0[1] : !llvm.array<16 x !llvm.array<2 x vector<2xf16>>
    %m2 = llvm.extractvalue %s1[0] : !llvm.array<2 x vector<2xf16>>
    %m3 = llvm.extractvalue %s1[1] : !llvm.array<2 x vector<2xf16>>

    %s2 = llvm.extractvalue %0[2] : !llvm.array<16 x !llvm.array<2 x vector<2xf16>>
    %m4 = llvm.extractvalue %s2[0] : !llvm.array<2 x vector<2xf16>>
    %m5 = llvm.extractvalue %s2[1] : !llvm.array<2 x vector<2xf16>>

    %s3 = llvm.extractvalue %0[3] : !llvm.array<16 x !llvm.array<2 x vector<2xf16>>
    %m6 = llvm.extractvalue %s3[0] : !llvm.array<2 x vector<2xf16>>
    %m7 = llvm.extractvalue %s3[1] : !llvm.array<2 x vector<2xf16>>

    %s4 = llvm.extractvalue %0[4] : !llvm.array<16 x !llvm.array<2 x vector<2xf16>>
    %m8 = llvm.extractvalue %s4[0] : !llvm.array<2 x vector<2xf16>>
    %m9 = llvm.extractvalue %s4[1] : !llvm.array<2 x vector<2xf16>>
    
    %s5 = llvm.extractvalue %0[5] : !llvm.array<16 x !llvm.array<2 x vector<2xf16>>
    %m10 = llvm.extractvalue %s5[0] : !llvm.array<2 x vector<2xf16>>
    %m11 = llvm.extractvalue %s5[1] : !llvm.array<2 x vector<2xf16>>

    %s6 = llvm.extractvalue %0[6] : !llvm.array<16 x !llvm.array<2 x vector<2xf16>>
    %m12 = llvm.extractvalue %s6[0] : !llvm.array<2 x vector<2xf16>>
    %m13 = llvm.extractvalue %s6[1] : !llvm.array<2 x vector<2xf16>>

    %s7 = llvm.extractvalue %0[7] : !llvm.array<16 x !llvm.array<2 x vector<2xf16>>
    %m14 = llvm.extractvalue %s7[0] : !llvm.array<2 x vector<2xf16>>
    %m15 = llvm.extractvalue %s7[1] : !llvm.array<2 x vector<2xf16>>

    
    %s8 = llvm.extractvalue %0[8] : !llvm.array<16 x !llvm.array<2 x vector<2xf16>>
    %m16 = llvm.extractvalue %s8[0] : !llvm.array<2 x vector<2xf16>>
    %m17 = llvm.extractvalue %s8[1] : !llvm.array<2 x vector<2xf16>>
    
    %s9 = llvm.extractvalue %0[9] : !llvm.array<16 x !llvm.array<2 x vector<2xf16>>
    %m18 = llvm.extractvalue %s9[0] : !llvm.array<2 x vector<2xf16>>
    %m19 = llvm.extractvalue %s9[1] : !llvm.array<2 x vector<2xf16>>

    %s10 = llvm.extractvalue %0[10] : !llvm.array<16 x !llvm.array<2 x vector<2xf16>>
    %m20 = llvm.extractvalue %s10[0] : !llvm.array<2 x vector<2xf16>>
    %m21 = llvm.extractvalue %s10[1] : !llvm.array<2 x vector<2xf16>>

    
    %s11 = llvm.extractvalue %0[11] : !llvm.array<16 x !llvm.array<2 x vector<2xf16>>
    %m22 = llvm.extractvalue %s11[0] : !llvm.array<2 x vector<2xf16>>
    %m23 = llvm.extractvalue %s11[1] : !llvm.array<2 x vector<2xf16>>
    
    %s12 = llvm.extractvalue %0[12] : !llvm.array<16 x !llvm.array<2 x vector<2xf16>>
    %m24 = llvm.extractvalue %s12[0] : !llvm.array<2 x vector<2xf16>>
    %m25 = llvm.extractvalue %s12[1] : !llvm.array<2 x vector<2xf16>>

    %s13 = llvm.extractvalue %0[13] : !llvm.array<16 x !llvm.array<2 x vector<2xf16>>
    %m26 = llvm.extractvalue %s13[0] : !llvm.array<2 x vector<2xf16>>
    %m27 = llvm.extractvalue %s13[1] : !llvm.array<2 x vector<2xf16>>

    %s14 = llvm.extractvalue %0[14] : !llvm.array<16 x !llvm.array<2 x vector<2xf16>>
    %m28 = llvm.extractvalue %s14[0] : !llvm.array<2 x vector<2xf16>>
    %m29 = llvm.extractvalue %s14[1] : !llvm.array<2 x vector<2xf16>>

    %s15 = llvm.extractvalue %0[15] : !llvm.array<16 x !llvm.array<2 x vector<2xf16>>
    %m30 = llvm.extractvalue %s15[0] : !llvm.array<2 x vector<2xf16>>
    %m31 = llvm.extractvalue %s15[1] : !llvm.array<2 x vector<2xf16>>

    // Flatten 16x2x2xf32 accumulator from vector<16x2x2xf32> to llvm.struct<(...)
    %accum0 = llvm.extractelement %m0[%c0 : i64] : vector<2xf32>
    %accum1 = llvm.extractelement %m0[%c1 : i64] : vector<2xf32>
    %accum2 = llvm.extractelement %m1[%c0 : i64] : vector<2xf32>
    %accum3 = llvm.extractelement %m1[%c1 : i64] : vector<2xf32>
    %accum4 = llvm.extractelement %m2[%c0 : i64] : vector<2xf32>
    %accum5 = llvm.extractelement %m2[%c1 : i64] : vector<2xf32>
    %accum6 = llvm.extractelement %m3[%c0 : i64] : vector<2xf32>
    %accum7 = llvm.extractelement %m3[%c1 : i64] : vector<2xf32>
    %accum8 = llvm.extractelement %m4[%c0 : i64] : vector<2xf32>
    %accum9 = llvm.extractelement %m4[%c1 : i64] : vector<2xf32>
    %accum10 = llvm.extractelement %m5[%c0 : i64] : vector<2xf32>
    %accum11 = llvm.extractelement %m5[%c1 : i64] : vector<2xf32>
    %accum12 = llvm.extractelement %m6[%c0 : i64] : vector<2xf32>
    %accum13 = llvm.extractelement %m6[%c1 : i64] : vector<2xf32>
    %accum14 = llvm.extractelement %m7[%c0 : i64] : vector<2xf32>
    %accum15 = llvm.extractelement %m7[%c1 : i64] : vector<2xf32>
    %accum16 = llvm.extractelement %m8[%c0 : i64] : vector<2xf32>
    %accum17 = llvm.extractelement %m8[%c1 : i64] : vector<2xf32>
    %accum18 = llvm.extractelement %m9[%c0 : i64] : vector<2xf32>
    %accum19 = llvm.extractelement %m9[%c1 : i64] : vector<2xf32>
    %accum20 = llvm.extractelement %m10[%c0 : i64] : vector<2xf32>
    %accum21 = llvm.extractelement %m10[%c1 : i64] : vector<2xf32>
    %accum22 = llvm.extractelement %m11[%c0 : i64] : vector<2xf32>
    %accum23 = llvm.extractelement %m11[%c1 : i64] : vector<2xf32>
    %accum24 = llvm.extractelement %m12[%c0 : i64] : vector<2xf32>
    %accum25 = llvm.extractelement %m12[%c1 : i64] : vector<2xf32>
    %accum26 = llvm.extractelement %m13[%c0 : i64] : vector<2xf32>
    %accum27 = llvm.extractelement %m13[%c1 : i64] : vector<2xf32>
    %accum28 = llvm.extractelement %m14[%c0 : i64] : vector<2xf32>
    %accum29 = llvm.extractelement %m14[%c1 : i64] : vector<2xf32>
    %accum30 = llvm.extractelement %m15[%c0 : i64] : vector<2xf32>
    %accum31 = llvm.extractelement %m15[%c1 : i64] : vector<2xf32>
    %accum32 = llvm.extractelement %m16[%c0 : i64] : vector<2xf32>
    %accum33 = llvm.extractelement %m16[%c1 : i64] : vector<2xf32>
    %accum34 = llvm.extractelement %m17[%c0 : i64] : vector<2xf32>
    %accum35 = llvm.extractelement %m17[%c1 : i64] : vector<2xf32>
    %accum36 = llvm.extractelement %m18[%c0 : i64] : vector<2xf32>
    %accum37 = llvm.extractelement %m18[%c1 : i64] : vector<2xf32>
    %accum38 = llvm.extractelement %m19[%c0 : i64] : vector<2xf32>
    %accum39 = llvm.extractelement %m19[%c1 : i64] : vector<2xf32>
    %accum40 = llvm.extractelement %m20[%c0 : i64] : vector<2xf32>
    %accum41 = llvm.extractelement %m20[%c1 : i64] : vector<2xf32>
    %accum42 = llvm.extractelement %m21[%c0 : i64] : vector<2xf32>
    %accum43 = llvm.extractelement %m21[%c1 : i64] : vector<2xf32>
    %accum44 = llvm.extractelement %m22[%c0 : i64] : vector<2xf32>
    %accum45 = llvm.extractelement %m22[%c1 : i64] : vector<2xf32>
    %accum46 = llvm.extractelement %m23[%c0 : i64] : vector<2xf32>
    %accum47 = llvm.extractelement %m23[%c1 : i64] : vector<2xf32>
    %accum48 = llvm.extractelement %m24[%c0 : i64] : vector<2xf32>
    %accum49 = llvm.extractelement %m24[%c1 : i64] : vector<2xf32>
    %accum50 = llvm.extractelement %m25[%c0 : i64] : vector<2xf32>
    %accum51 = llvm.extractelement %m25[%c1 : i64] : vector<2xf32>
    %accum52 = llvm.extractelement %m26[%c0 : i64] : vector<2xf32>
    %accum53 = llvm.extractelement %m26[%c1 : i64] : vector<2xf32>
    %accum54 = llvm.extractelement %m27[%c0 : i64] : vector<2xf32>
    %accum55 = llvm.extractelement %m27[%c1 : i64] : vector<2xf32>
    %accum56 = llvm.extractelement %m28[%c0 : i64] : vector<2xf32>
    %accum57 = llvm.extractelement %m28[%c1 : i64] : vector<2xf32>
    %accum58 = llvm.extractelement %m29[%c0 : i64] : vector<2xf32>
    %accum59 = llvm.extractelement %m29[%c1 : i64] : vector<2xf32>
    %accum60 = llvm.extractelement %m30[%c0 : i64] : vector<2xf32>
    %accum61 = llvm.extractelement %m30[%c1 : i64] : vector<2xf32>
    %accum62 = llvm.extractelement %m31[%c0 : i64] : vector<2xf32>
    %accum63 = llvm.extractelement %m31[%c1 : i64] : vector<2xf32>



    %const0 = llvm.mlir.constant(0 : i64) : i64
    %acummStruct0 = llvm.insertelement %accum0, %llvm_undef[%const0 : i64] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
    %const1 = llvm.mlir.constant(1 : i64) : i64
    %acummStruct1 = llvm.insertelement %accum1, %acummStruct0[%const1 : i64] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
    %const2 = llvm.mlir.constant(2 : i64) : i64
    %acummStruct2 = llvm.insertelement %accum2, %acummStruct1[%const2 : i64] : !llvm.struct<(

    %5 = nvvm.wgmma.mma_async %0, %1, %4, <m = 64, n = 128, k = 16>, D[<f32>, <one>, <wrapped>], A[<f16>, <one>, <row>], B[<f16>, <one>, <row>] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
    
    return
  }