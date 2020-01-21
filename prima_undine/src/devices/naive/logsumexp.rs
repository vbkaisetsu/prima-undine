use crate::device_impl::FunctionFwImpl;
use crate::Tensor;

define_empty_impl!(LogsumexpFwImpl);
impl FunctionFwImpl for LogsumexpFwImpl {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let dim = u32data[0];
        let y = &mut ys[0];
        let n = x.shape[dim] as usize;
        let repeat = y.shape.size() as usize;
        let skip1 = y.shape.lower_volume(dim) as usize;
        let skip2 = skip1 * n;
        unsafe {
            let src = const_ptr!(x);
            let dest = mut_ptr!(y);
            for i in 0..repeat {
                let mut offset = i % skip1 + (i / skip1) * skip2;
                let mut tmp = *src.add(offset) as f64;
                for _ in 1..n {
                    offset += skip1;
                    let arg = *src.add(offset) as f64;
                    tmp = if tmp > arg {
                        tmp + (1. + (arg - tmp).exp()).ln()
                    } else {
                        arg + (1. + (tmp - arg).exp()).ln()
                    };
                }
                *dest.add(i) = tmp as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::devices as D;
    use crate::functions::BasicFunctions;

    #[test]
    fn check_logsumexp_fw() {
        let x_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., -1., -2., -3., -4., -5., -6., -7., -8.,
        ];
        let y_data = vec![
            vec![
                2.31326169,
                4.31326169,
                6.31326169,
                8.31326169,
                -0.68673831,
                -2.68673831,
                -4.68673831,
                -6.68673831,
            ],
            vec![
                3.12692801,
                4.12692801,
                7.12692801,
                8.12692801,
                -0.87307199,
                -1.87307199,
                -4.87307199,
                -5.87307199,
            ],
            vec![
                5.01814993,
                6.01814993,
                7.01814993,
                8.01814993,
                -0.98185007,
                -1.98185007,
                -2.98185007,
                -3.98185007,
            ],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., -1., -2., -3., -4., -5., -6., -7., -8.,
            ],
        ];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![2, 2, 2; 2], &x_data);
        for i in 0..4 {
            let y_shape = shape![2, 2, 2; 2].resize_dim(i, 1);
            let mut y = dev.new_tensor(y_shape);
            y.alloc();
            dev.call_fw_impl("logsumexp_fw_impl", &[&x], &[i], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data[i as usize], y.to_vec());
        }
    }

    #[test]
    fn check_logsumexp_fw_large() {
        let ns = vec![
            1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024, 1025, 2047, 2048, 2049, 65535, 65536,
            65537,
        ];
        let dev = D::Naive::new();
        for &n in &ns {
            for &k in &[-5., -1., 0., 1., 5.] {
                let x = dev.new_tensor_by_constant(shape![n], k);
                let mut y = dev.new_tensor(shape![]);
                y.alloc();
                dev.call_fw_impl("logsumexp_fw_impl", &[&x], &[0], &[], &mut [&mut y]);
                assert_vector_ulps_eq!(vec![k + (n as f32).ln()], y.to_vec());
            }
        }
    }
}
