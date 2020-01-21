use crate::device_impl::FunctionBwImpl;
use crate::Tensor;

define_naive_fw_ab_impl!(DivFwImpl, |a: f32, b: f32| { a / b });

define_empty_impl!(DivBwAImpl);
impl FunctionBwImpl for DivBwAImpl {
    fn call(
        &self,
        xs: &[&Tensor],
        _ys: &[&Tensor],
        gys: &[&Tensor],
        _u32data: &[u32],
        _f32data: &[f32],
        gx: &mut Tensor,
    ) {
        let b = xs[1];
        let gy = gys[0];
        let ga = gx;
        let volume = gy.shape.volume() as usize;
        let a_shift = if ga.shape.batch() == 1 { 0 } else { volume };
        let b_shift = if b.shape.batch() == 1 { 0 } else { volume };
        unsafe {
            let mut pb = const_ptr!(b);
            let mut pgy = const_ptr!(gy);
            let mut pga = mut_ptr!(ga);
            for _ in 0..gy.shape.batch() {
                for i in 0..volume {
                    *pga.add(i) += *pgy.add(i) / *pb.add(i);
                }
                pb = pb.add(b_shift);
                pgy = pgy.add(volume);
                pga = pga.add(a_shift);
            }
        }
    }
}

define_empty_impl!(DivBwBImpl);
impl FunctionBwImpl for DivBwBImpl {
    fn call(
        &self,
        xs: &[&Tensor],
        ys: &[&Tensor],
        gys: &[&Tensor],
        _u32data: &[u32],
        _f32data: &[f32],
        gx: &mut Tensor,
    ) {
        let b = xs[1];
        let y = ys[0];
        let gy = gys[0];
        let gb = gx;
        let volume = gy.shape.volume() as usize;
        let b_shift = if gb.shape.batch() == 1 { 0 } else { volume };
        unsafe {
            let mut pb = const_ptr!(b);
            let mut py = const_ptr!(y);
            let mut pgy = const_ptr!(gy);
            let mut pgb = mut_ptr!(gb);
            for _ in 0..gy.shape.batch() {
                for i in 0..volume {
                    *pgb.add(i) -= *pgy.add(i) * *py.add(i) / *pb.add(i);
                }
                pb = pb.add(b_shift);
                py = py.add(volume);
                pgy = pgy.add(volume);
                pgb = pgb.add(b_shift);
            }
        }
    }
}

define_naive_fw_const_impl!(DivConstLFwImpl, |x: f32, k: f32| { k / x });
define_naive_bw_const_impl!(DivConstLBwImpl, |x: f32, y: f32, gy: f32, _k: f32| {
    -gy * y / x
});
define_naive_fw_const_impl!(DivConstRFwImpl, |x: f32, k: f32| { x / k });
define_naive_bw_const_impl!(DivConstRBwImpl, |_x: f32, _y: f32, gy: f32, k: f32| {
    gy / k
});
define_naive_fw_scalar_impl!(DivScalarLFwImpl, |x: f32, k: f32| { k / x });
define_naive_fw_scalar_impl!(DivScalarRFwImpl, |x: f32, k: f32| { x / k });

#[cfg(test)]
mod tests {
    use crate::devices as D;
    use crate::functions::BasicFunctions;

    #[test]
    fn check_div_const_l_fw() {
        let x_data = vec![1000., -100., 10., -1., 0.1, -0.01, 0.001, -0.001];
        let k = 10.;
        let y_data = vec![0.01, -0.1, 1., -10., 100., -1000., 10000., -10000.];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("div_const_l_fw_impl", &[&x], &[], &[k], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_div_const_r_fw() {
        let x_data = vec![1000., -100., 10., -1., 0.1, -0.01, 0.001, -0.001];
        let k = 10.;
        let y_data = vec![100., -10., 1., -0.1, 0.01, -0.001, 0.0001, -0.0001];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("div_const_r_fw_impl", &[&x], &[], &[k], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_div_scalar_l_fw() {
        let x_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.0001];
        let k_data = vec![10., 0.1];
        let y_data = vec![0.01, 0.1, 1., 10., 1., 10., 100., 1000.];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let k = dev.new_tensor_by_slice(shape![; 2], &k_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("div_scalar_l_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_div_scalar_r_fw() {
        let x_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.0001];
        let k_data = vec![10., 0.1];
        let y_data = vec![100., 10., 1., 0.1, 1., 0.1, 0.01, 0.001];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let k = dev.new_tensor_by_slice(shape![; 2], &k_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("div_scalar_r_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_div_scalar_l_fw_batch_broadcast() {
        let dev = D::Naive::new();
        {
            let x_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.001];
            let k_data = vec![10.];
            let y_data = vec![0.01, 0.1, 1., 10., 100., 1000., 10000., 10000.];
            let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
            let k = dev.new_tensor_by_slice(shape![], &k_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("div_scalar_l_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
        {
            let x_data = vec![1000., 100., 10., 1.];
            let k_data = vec![10., 0.1];
            let y_data = vec![0.01, 0.1, 1., 10., 0.0001, 0.001, 0.01, 0.1];
            let x = dev.new_tensor_by_slice(shape![2, 2], &x_data);
            let k = dev.new_tensor_by_slice(shape![; 2], &k_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("div_scalar_l_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
    }

    #[test]
    fn check_div_scalar_r_fw_batch_broadcast() {
        let dev = D::Naive::new();
        {
            let x_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.001];
            let k_data = vec![10.];
            let y_data = vec![100., 10., 1., 0.1, 0.01, 0.001, 0.0001, 0.0001];
            let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
            let k = dev.new_tensor_by_slice(shape![], &k_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("div_scalar_r_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
        {
            let x_data = vec![1000., 100., 10., 1.];
            let k_data = vec![10., 0.1];
            let y_data = vec![100., 10., 1., 0.1, 10000., 1000., 100., 10.];
            let x = dev.new_tensor_by_slice(shape![2, 2], &x_data);
            let k = dev.new_tensor_by_slice(shape![; 2], &k_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("div_scalar_r_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
    }

    #[test]
    fn check_div_fw() {
        let a_data = vec![1000., -100., 10., -1., 0.1, -0.01, 0.001, -0.001];
        let b_data = vec![1., 2., 3., 4., -5., -6., -7., -8.];
        let y1_data = vec![
            1000.,
            -50.,
            10. / 3.,
            -0.25,
            -0.02,
            0.01 / 6.,
            -0.001 / 7.,
            1.25e-4,
        ];
        let y2_data = vec![0.001, -0.02, 0.3, -4., -50., 600., -7000., 8000.];
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_slice(shape![2, 2; 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
        let mut y1 = dev.new_tensor(shape![2, 2; 2]);
        y1.alloc();
        dev.call_fw_impl("div_fw_impl", &[&a, &b], &[], &[], &mut [&mut y1]);
        assert_vector_ulps_eq!(y1_data, y1.to_vec());
        let mut y2 = dev.new_tensor(shape![2, 2; 2]);
        y2.alloc();
        dev.call_fw_impl("div_fw_impl", &[&b, &a], &[], &[], &mut [&mut y2]);
        assert_vector_ulps_eq!(y2_data, y2.to_vec());
    }

    #[test]
    fn check_div_fw_batch_broadcast() {
        let a_data = vec![1., 2., 3., 4.];
        let b_data = vec![1., 1., 1., 1., 1., 2., 3., 4.];
        let y1_data = vec![1., 2., 3., 4., 1., 1., 1., 1.];
        let y2_data = vec![1., 0.5, 1. / 3., 0.25, 1., 1., 1., 1.];
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_slice(shape![2, 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
        let mut y1 = dev.new_tensor(shape![2, 2; 2]);
        y1.alloc();
        dev.call_fw_impl("div_fw_impl", &[&a, &b], &[], &[], &mut [&mut y1]);
        assert_vector_ulps_eq!(y1_data, y1.to_vec());
        let mut y2 = dev.new_tensor(shape![2, 2; 2]);
        y2.alloc();
        dev.call_fw_impl("div_fw_impl", &[&b, &a], &[], &[], &mut [&mut y2]);
        assert_vector_ulps_eq!(y2_data, y2.to_vec());
    }

    #[test]
    fn check_div_const_l_bw() {
        let ks = vec![0.01, 0.1, 1., 10., 100., -0.01, -0.1, -1., -10., -100.];
        let x_data = vec![0.1, 1., 2., 3., -0.1, -1., -2., -3.];
        let gy_data = vec![1., -1., 2., -2., 2., -2., 1., -1.];
        let dev = D::Naive::new();
        for k in ks {
            let gx_data = vec![
                1. - 100. * k,
                1. + k,
                1. - k / 2.,
                1. + 2. * k / 9.,
                1. - 200. * k,
                1. + 2. * k,
                1. - k / 4.,
                1. + k / 9.,
            ];
            let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("div_const_l_fw_impl", &[&x], &[], &[k], &mut [&mut y]);
            let gy = dev.new_tensor_by_slice(y.shape, &gy_data);
            let mut gx = dev.new_tensor_by_constant(x.shape, 1.);
            dev.call_bw_impl(
                "div_const_l_bw_impl",
                &[&x],
                &[&y],
                &[&gy],
                &[],
                &[k],
                &mut gx,
            );
            assert_vector_ulps_eq!(gx_data, gx.to_vec());
        }
    }

    #[test]
    fn check_div_const_r_bw() {
        let ks = vec![0.01, 0.1, 1., 10., 100., -0.01, -0.1, -1., -10., -100.];
        let x_data = vec![0., 1., 2., 3., 0., -1., -2., -3.];
        let gy_data = vec![1., -1., 2., -2., 2., -2., 1., -1.];
        let dev = D::Naive::new();
        for k in ks {
            let gx_data = vec![
                1. + 1. / k,
                1. - 1. / k,
                1. + 2. / k,
                1. - 2. / k,
                1. + 2. / k,
                1. - 2. / k,
                1. + 1. / k,
                1. - 1. / k,
            ];
            let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("div_const_r_fw_impl", &[&x], &[], &[k], &mut [&mut y]);
            let gy = dev.new_tensor_by_slice(y.shape, &gy_data);
            let mut gx = dev.new_tensor_by_constant(x.shape, 1.);
            dev.call_bw_impl(
                "div_const_r_bw_impl",
                &[&x],
                &[&y],
                &[&gy],
                &[],
                &[k],
                &mut gx,
            );
            assert_vector_ulps_eq!(gx_data, gx.to_vec());
        }
    }

    #[test]
    fn check_div_bw_11() {
        let a_data = vec![1., 10.];
        let b_data = vec![10., 1.];
        let gy_data = vec![1., -1.];
        let ga_data = vec![1.1, 0.];
        let gb_data = vec![0.99, 11.];
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_slice(shape![2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2], &b_data);
        let mut y = dev.new_tensor(shape![2]);
        y.alloc();
        dev.call_fw_impl("div_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
        let gy = dev.new_tensor_by_slice(shape![2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2], 1.);
        dev.call_bw_impl("div_bw_a_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut ga);
        dev.call_bw_impl("div_bw_b_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut gb);
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }

    #[test]
    fn check_div_bw_1n() {
        let a_data = vec![1., 10.];
        let b_data = vec![10., 1., -10., -1.];
        let gy_data = vec![1., -1., 2., -2.];
        let ga_data = vec![0.9, 2.];
        let gb_data = vec![0.99, 11., 0.98, 21.];
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_slice(shape![2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2; 2], &b_data);
        let mut y = dev.new_tensor(shape![2; 2]);
        y.alloc();
        dev.call_fw_impl("div_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
        let gy = dev.new_tensor_by_slice(shape![2; 2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2; 2], 1.);
        dev.call_bw_impl("div_bw_a_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut ga);
        dev.call_bw_impl("div_bw_b_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut gb);
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }

    #[test]
    fn check_div_bw_n1() {
        let a_data = vec![1., 10., -1., -10.];
        let b_data = vec![10., 1.];
        let gy_data = vec![1., -1., 2., -2.];
        let ga_data = vec![1.1, 0., 1.2, -1.];
        let gb_data = vec![1.01, -9.];
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_slice(shape![2; 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2], &b_data);
        let mut y = dev.new_tensor(shape![2; 2]);
        y.alloc();
        dev.call_fw_impl("div_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
        let gy = dev.new_tensor_by_slice(shape![2; 2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2; 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2], 1.);
        dev.call_bw_impl("div_bw_a_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut ga);
        dev.call_bw_impl("div_bw_b_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut gb);
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }

    #[test]
    fn check_div_bw_nn() {
        let a_data = vec![1., 10., -1., -10.];
        let b_data = vec![10., 1., -10., -1.];
        let gy_data = vec![1., -1., 2., -2.];
        let ga_data = vec![1.1, 0., 0.8, 3.];
        let gb_data = vec![0.99, 11., 1.02, -19.];
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_slice(shape![2; 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2; 2], &b_data);
        let mut y = dev.new_tensor(shape![2; 2]);
        y.alloc();
        dev.call_fw_impl("div_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
        let gy = dev.new_tensor_by_slice(shape![2; 2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2; 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2; 2], 1.);
        dev.call_bw_impl("div_bw_a_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut ga);
        dev.call_bw_impl("div_bw_b_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut gb);
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }
}
