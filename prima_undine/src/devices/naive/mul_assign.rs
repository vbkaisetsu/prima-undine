use crate::device_impl::FunctionFwImpl;
use crate::Tensor;

define_empty_impl!(MulAssignConstImpl);
impl FunctionFwImpl for MulAssignConstImpl {
    fn call(&self, _xs: &[&Tensor], _u32data: &[u32], f32data: &[f32], ys: &mut [&mut Tensor]) {
        let k = f32data[0];
        let y = &mut ys[0];
        let size = y.shape.size() as usize;
        unsafe {
            let py = mut_ptr!(y);
            for i in 0..size {
                *py.add(i) *= k;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::devices as D;
    use crate::functions::BasicFunctions;

    #[test]
    fn check_mul_assign_const() {
        let x_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.0001];
        let k = 5.;
        let y_data = vec![5000., 500., 50., 5., 0.5, 0.05, 0.005, 0.0005];
        let dev = D::Naive::new();
        let mut x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        dev.call_fw_impl("mul_assign_const_impl", &[], &[], &[k], &mut [&mut x]);
        assert_vector_ulps_eq!(y_data, x.to_vec());
    }
}
