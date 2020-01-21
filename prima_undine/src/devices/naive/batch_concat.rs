use crate::device_impl::FunctionFwImpl;
use crate::Tensor;

define_empty_impl!(BatchConcatFwImpl);
impl FunctionFwImpl for BatchConcatFwImpl {
    fn call(&self, xs: &[&Tensor], _u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let y = &mut ys[0];
        let mut offset = 0;
        unsafe {
            let py = mut_ptr!(y);
            for x in xs {
                let px = const_ptr!(x);
                let dest = py.add(offset);
                let span = x.shape.size() as usize;
                for i in 0..span {
                    *dest.add(i) = *px.add(i);
                }
                offset += span;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::devices as D;
    use crate::functions::BasicFunctions;

    #[test]
    fn check_batch_concat_fw() {
        let a_data = vec![1., 2., 3., 4., 5., 6.];
        let b_data = vec![7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.];
        let c_data = vec![
            19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35.,
            36.,
        ];
        let y_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36.,
        ];
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_slice(shape![2, 3], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 3; 2], &b_data);
        let c = dev.new_tensor_by_slice(shape![2, 3; 3], &c_data);
        let mut y = dev.new_tensor(shape![2, 3; 6]);
        y.alloc();
        dev.call_fw_impl(
            "batch_concat_fw_impl",
            &[&a, &b, &c],
            &[],
            &[],
            &mut [&mut y],
        );
        assert_vector_ulps_eq!(&y_data, &y.to_vec());
    }
}
