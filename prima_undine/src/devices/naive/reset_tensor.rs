use crate::device_impl::FunctionFwImpl;
use crate::functions::BasicFunctions;
use crate::Tensor;

define_empty_impl!(ResetTensorImpl);
impl FunctionFwImpl for ResetTensorImpl {
    fn call(&self, _xs: &[&Tensor], _u32data: &[u32], f32data: &[f32], ys: &mut [&mut Tensor]) {
        let k = f32data[0];
        let y = &mut ys[0];
        let size = y.shape.size() as usize;
        unsafe {
            let py = mut_ptr!(y);
            for i in 0..size {
                *py.add(i) = k;
            }
        }
    }
}

define_empty_impl!(ResetTensorBySliceImpl);
impl FunctionFwImpl for ResetTensorBySliceImpl {
    fn call(&self, _xs: &[&Tensor], _u32data: &[u32], f32data: &[f32], ys: &mut [&mut Tensor]) {
        let values = &f32data[..];
        let y = &mut ys[0];
        let size = y.shape.size() as usize;
        unsafe {
            let py = mut_ptr!(y);
            for i in 0..size {
                *py.add(i) = values[i];
            }
        }
    }
}

define_empty_impl!(ResetTensorByTensorImpl);
impl FunctionFwImpl for ResetTensorByTensorImpl {
    fn call(&self, xs: &[&Tensor], _u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let y = &mut ys[0];
        let size = y.shape.size() as usize;
        let x_data = x.to_vec();
        unsafe {
            let py = mut_ptr!(y);
            for i in 0..size {
                *py.add(i) = x_data[i];
            }
        }
    }
}
