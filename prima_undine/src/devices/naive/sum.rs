use crate::device_impl::FunctionFwImpl;
use crate::Tensor;

define_empty_impl!(SumFwImpl);
impl FunctionFwImpl for SumFwImpl {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let dim = u32data[0];
        let y = &mut ys[0];
        let n = x.shape[dim] as usize;
        let repeat = y.shape.size() as usize;
        let skip1 = y.shape.lower_volume(dim) as usize;
        let skip2 = skip1 * n;
        unsafe {
            let px = const_ptr!(x);
            let py = mut_ptr!(y);
            for i in 0..repeat {
                let mut offset = i % skip1 + (i / skip1) * skip2;
                let mut tmp = 0.;
                for _ in 0..n {
                    tmp += *px.add(offset);
                    offset += skip1;
                }
                *py.add(i) = tmp;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::devices as D;
    use crate::functions::BasicFunctions;

    #[test]
    fn check_sum_fw() {
        let x_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., -1., -2., -3., -4., -5., -6., -7., -8.,
        ];
        let shape = vec![
            shape![1, 2, 2; 2],
            shape![2, 1, 2; 2],
            shape![2, 2; 2],
            shape![2, 2, 2; 2],
        ];
        let y_data = vec![
            vec![3., 7., 11., 15., -3., -7., -11., -15.],
            vec![4., 6., 12., 14., -4., -6., -12., -14.],
            vec![6., 8., 10., 12., -6., -8., -10., -12.],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., -1., -2., -3., -4., -5., -6., -7., -8.,
            ],
        ];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![2, 2, 2; 2], &x_data);
        for i in 0..4 {
            let mut y = dev.new_tensor(shape[i]);
            y.alloc();
            dev.call_fw_impl("sum_fw_impl", &[&x], &[i as u32], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data[i], y.to_vec());
        }
    }

    #[test]
    fn check_sum_fw_2() {
        let ns = vec![
            1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024, 1025, 2047, 2048, 2049, 65535, 65536,
            65537,
        ];
        let dev = D::Naive::new();
        for n in ns {
            let x = dev.new_tensor_by_constant(shape![n], 1.);
            let mut y = dev.new_tensor(shape![]);
            y.alloc();
            dev.call_fw_impl("sum_fw_impl", &[&x], &[0], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(vec![n as f32], y.to_vec());
        }
    }
}
