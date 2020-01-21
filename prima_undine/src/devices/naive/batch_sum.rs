use crate::device_impl::FunctionFwImpl;
use crate::Tensor;

define_empty_impl!(BatchSumFwImpl);
impl FunctionFwImpl for BatchSumFwImpl {
    fn call(&self, xs: &[&Tensor], _u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let y = &mut ys[0];
        let bs = x.shape.batch();
        let size = y.shape.size() as usize;
        unsafe {
            let px = const_ptr!(x);
            let py = mut_ptr!(y);
            for i in 0..size {
                let mut temp = 0.0;
                let mut pos = i;
                for _ in 0..bs {
                    temp += *px.add(pos);
                    pos += size;
                }
                *py.add(i) = temp;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::devices as D;
    use crate::functions::BasicFunctions;

    #[test]
    fn batch_sum_fw_test() {
        let x_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., -2., -4., -6., -8., -10., -12., -14., -16.,
        ];
        let y_data = vec![-1., -2., -3., -4., -5., -6., -7., -8.];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![2, 2, 2; 2], &x_data);
        let mut y = dev.new_tensor(shape![2, 2, 2]);
        y.alloc();
        dev.call_fw_impl("batch_sum_fw_impl", &[&x], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }
}
