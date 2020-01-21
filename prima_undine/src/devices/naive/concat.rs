use crate::device_impl::FunctionFwImpl;
use crate::Tensor;

define_empty_impl!(ConcatFwImpl);
impl FunctionFwImpl for ConcatFwImpl {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let dim = u32data[0];
        let y = &mut ys[0];
        let new_bs = y.shape.batch() as usize;
        let base = y.shape.lower_volume(dim) as usize;
        let skip = base * y.shape[dim] as usize;
        let repeat = y.shape.volume() as usize / skip;
        let mut offset = 0;
        for x in xs {
            let src_dim = x.shape[dim] as usize;
            let span = base * src_dim;
            let b_skip = if x.shape.has_batch() {
                span * repeat
            } else {
                0
            };
            unsafe {
                let src = const_ptr!(*x);
                let mut dest = mut_ptr!(y).add(offset);
                for batch in 0..new_bs {
                    let mut sp = src.add(b_skip * batch);
                    for _ in 0..repeat {
                        let dp = dest;
                        for j in 0..span {
                            *dp.add(j) = *sp;
                            sp = sp.add(1);
                        }
                        dest = dest.add(skip);
                    }
                }
            }
            offset += span;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::devices as D;
    use crate::functions::BasicFunctions;

    #[test]
    fn check_concat_fw_n_3x3() {
        let a_data = vec![1., 1., 1.];
        let b_data = vec![2., 3., 2., 3., 2., 3.];
        let c_data = vec![4., 5., 6., 4., 5., 6., 4., 5., 6.];
        let y_data = vec![
            1., 2., 3., 4., 5., 6., 1., 2., 3., 4., 5., 6., 1., 2., 3., 4., 5., 6.,
        ];
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_slice(shape![1, 3], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 3], &b_data);
        let c = dev.new_tensor_by_slice(shape![3, 3], &c_data);
        let mut y = dev.new_tensor(shape![6, 3]);
        y.alloc();
        dev.call_fw_impl("concat_fw_impl", &[&a, &b, &c], &[0], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_concat_fw_5x4() {
        let shapes = vec![shape![20], shape![5, 4], shape![5, 1, 4]];
        let y_data = vec![
            1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3., 4., 4., 4., 4., 4.,
        ];
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_constant(shape![5], 1.);
        let b = dev.new_tensor_by_constant(shape![5], 2.);
        let c = dev.new_tensor_by_constant(shape![5], 3.);
        let d = dev.new_tensor_by_constant(shape![5], 4.);
        for &i in &[0, 1, 2] {
            let mut y = dev.new_tensor(shapes[i]);
            y.alloc();
            dev.call_fw_impl(
                "concat_fw_impl",
                &[&a, &b, &c, &d],
                &[i as u32],
                &[],
                &mut [&mut y],
            );
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
    }

    #[test]
    fn check_concat_fw_2_2_2x2() {
        let a_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 11., 22., 33., 44., 55., 66., 77., 88.,
        ];
        let b_data = vec![
            -1., -2., -3., -4., -5., -6., -7., -8., -11., -22., -33., -44., -55., -66., -77., -88.,
        ];
        let shapes = vec![
            shape![4, 2, 2; 2],
            shape![2, 4, 2; 2],
            shape![2, 2, 4; 2],
            shape![2, 2, 2, 2; 2],
            shape![2, 2, 2, 1, 2; 2],
        ];
        let y_data = vec![
            vec![
                1., 2., -1., -2., 3., 4., -3., -4., 5., 6., -5., -6., 7., 8., -7., -8., 11., 22.,
                -11., -22., 33., 44., -33., -44., 55., 66., -55., -66., 77., 88., -77., -88.,
            ],
            vec![
                1., 2., 3., 4., -1., -2., -3., -4., 5., 6., 7., 8., -5., -6., -7., -8., 11., 22.,
                33., 44., -11., -22., -33., -44., 55., 66., 77., 88., -55., -66., -77., -88.,
            ],
            vec![
                1., 2., 3., 4., 5., 6., 7., 8., -1., -2., -3., -4., -5., -6., -7., -8., 11., 22.,
                33., 44., 55., 66., 77., 88., -11., -22., -33., -44., -55., -66., -77., -88.,
            ],
        ];
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_slice(shape![2, 2, 2; 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2, 2; 2], &b_data);
        for &i in &[0, 1, 2, 3, 4] {
            let mut y = dev.new_tensor(shapes[i]);
            y.alloc();
            dev.call_fw_impl("concat_fw_impl", &[&a, &b], &[i as u32], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data[if i < 2 { i } else { 2 }], y.to_vec());
        }
    }

    #[test]
    fn check_concat_fw_batch_broadcast() {
        let dev = D::Naive::new();
        {
            let a_data = vec![1., 1., 11., 11.];
            let b_data = vec![2., 2., 2., 2.];
            let c_data = vec![3., 3., 3., 3., 3., 3.];
            let y_data = vec![
                1., 1., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 11., 11., 2., 2., 2., 2., 3., 3.,
                3., 3., 3., 3.,
            ];
            let a = dev.new_tensor_by_slice(shape![2, 1; 2], &a_data);
            let b = dev.new_tensor_by_slice(shape![2, 2], &b_data);
            let c = dev.new_tensor_by_slice(shape![2, 3], &c_data);
            let mut y = dev.new_tensor(shape![2, 6; 2]);
            y.alloc();
            dev.call_fw_impl("concat_fw_impl", &[&a, &b, &c], &[1], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
        {
            let a_data = vec![1., 1., 1., 1., 1., 1.];
            let b_data = vec![2., 2., 2., 2., 22., 22., 22., 22.];
            let c_data = vec![3., 3., 33., 33.];
            let y_data = vec![
                1., 1., 1., 2., 2., 3., 1., 1., 1., 2., 2., 3., 1., 1., 1., 22., 22., 33., 1., 1.,
                1., 22., 22., 33.,
            ];
            let a = dev.new_tensor_by_slice(shape![3, 2], &a_data);
            let b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
            let c = dev.new_tensor_by_slice(shape![1, 2; 2], &c_data);
            let mut y = dev.new_tensor(shape![6, 2; 2]);
            y.alloc();
            dev.call_fw_impl("concat_fw_impl", &[&a, &b, &c], &[0], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
        {
            let a_data = vec![1.];
            let b_data = vec![2.];
            let c_data = vec![3., 33., 333.];
            let y_data = vec![1., 2., 3., 1., 2., 33., 1., 2., 333.];
            let a = dev.new_tensor_by_slice(shape![], &a_data);
            let b = dev.new_tensor_by_slice(shape![], &b_data);
            let c = dev.new_tensor_by_slice(shape![; 3], &c_data);
            let mut y = dev.new_tensor(shape![3; 3]);
            y.alloc();
            dev.call_fw_impl("concat_fw_impl", &[&a, &b, &c], &[0], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
    }
}
