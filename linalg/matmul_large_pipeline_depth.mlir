#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @mma_fused {
  hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}> {
  hal.executable.export public @_large_aligned_dispatch_0 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [#hal.descriptor_set.layout<0, bindings = [#hal.descriptor_set.binding<0, storage_buffer>, #hal.descriptor_set.binding<1, storage_buffer>, #hal.descriptor_set.binding<2, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @_large_aligned_dispatch_0() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %c2048 = arith.constant 2048 : index
      %c512 = arith.constant 512 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : !flow.dispatch.tensor<readonly:2048x1024xf32>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : !flow.dispatch.tensor<readonly:1024x512xf32>
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<writeonly:2048x512xf32>
      %di = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : !flow.dispatch.tensor<readonly:2048x512xf32>
      %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 1024], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:2048x1024xf32> -> tensor<2048x1024xf32>
      %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1024, 512], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:1024x512xf32> -> tensor<1024x512xf32>
      %d = flow.dispatch.tensor.load %di, offsets = [0, 0], sizes = [2048, 512], strides = [1, 1]
          : !flow.dispatch.tensor<readonly:2048x512xf32> -> tensor<2048x512xf32>
      %init = linalg.init_tensor [2048, 512] : tensor<2048x512xf32>
      %f = linalg.fill ins(%cst : f32) outs(%init : tensor<2048x512xf32>) -> tensor<2048x512xf32>
      %m = linalg.matmul ins(%3, %4 : tensor<2048x1024xf32>, tensor<1024x512xf32>) outs(%f : tensor<2048x512xf32>) -> tensor<2048x512xf32>
      %init2 = linalg.init_tensor [2048, 512] : tensor<2048x512xf32>
      %a = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%m, %d : tensor<2048x512xf32>, tensor<2048x512xf32>) outs(%init2 : tensor<2048x512xf32>) {
        ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
          %19 = arith.addf %arg3, %arg4 : f32
          linalg.yield %19 : f32
        } -> (tensor<2048x512xf32>)
        flow.dispatch.tensor.store %a, %2, offsets = [0, 0], sizes = [2048, 512], strides = [1, 1]
          : tensor<2048x512xf32> -> !flow.dispatch.tensor<writeonly:2048x512xf32>
      return
    }
  }
}
}