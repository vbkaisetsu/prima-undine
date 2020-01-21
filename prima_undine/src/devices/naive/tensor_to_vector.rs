use crate::device_impl::FunctionFwF32Impl;
use crate::Tensor;

define_empty_impl!(TensorToVectorImpl);
impl FunctionFwF32Impl for TensorToVectorImpl {
    fn call(&self, xs: &[&Tensor], _u32data: &[u32], _f32data: &[f32], ys: &mut [f32]) {
        let x = xs[0];
        let y = ys;
        let size = xs[0].shape.size() as usize;
        unsafe {
            let px = const_ptr!(x);
            for i in 0..size {
                y[i] = *px.add(i);
            }
        }
    }
}
