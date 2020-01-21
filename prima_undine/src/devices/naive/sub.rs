use crate::device_impl::FunctionBwImpl;
use crate::Tensor;

define_naive_fw_ab_impl!(SubFwImpl, |a: f32, b: f32| { a - b });

define_empty_impl!(SubBwAImpl);
impl FunctionBwImpl for SubBwAImpl {
    fn call(
        &self,
        _xs: &[&Tensor],
        _ys: &[&Tensor],
        gys: &[&Tensor],
        _u32data: &[u32],
        _f32data: &[f32],
        gx: &mut Tensor,
    ) {
        let gy = gys[0];
        let ga = gx;
        let volume = gy.shape.volume() as usize;
        let a_shift = if ga.shape.batch() == 1 { 0 } else { volume };
        unsafe {
            let mut pgy = const_ptr!(gy);
            let mut pga = mut_ptr!(ga);
            for _ in 0..gy.shape.batch() {
                for i in 0..volume {
                    *pga.add(i) += *pgy.add(i);
                }
                pgy = pgy.add(volume);
                pga = pga.add(a_shift);
            }
        }
    }
}

define_empty_impl!(SubBwBImpl);
impl FunctionBwImpl for SubBwBImpl {
    fn call(
        &self,
        _xs: &[&Tensor],
        _ys: &[&Tensor],
        gys: &[&Tensor],
        _u32data: &[u32],
        _f32data: &[f32],
        gx: &mut Tensor,
    ) {
        let gy = gys[0];
        let gb = gx;
        let volume = gy.shape.volume() as usize;
        let b_shift = if gb.shape.batch() == 1 { 0 } else { volume };
        unsafe {
            let mut pgy = const_ptr!(gy);
            let mut pgb = mut_ptr!(gb);
            for _ in 0..gy.shape.batch() {
                for i in 0..volume {
                    *pgb.add(i) -= *pgy.add(i);
                }
                pgy = pgy.add(volume);
                pgb = pgb.add(b_shift);
            }
        }
    }
}

define_naive_fw_const_impl!(SubConstLFwImpl, |x: f32, k: f32| { k - x });
define_naive_bw_const_impl!(SubConstLBwImpl, |_x: f32, _y: f32, gy: f32, _k: f32| {
    -gy
});
define_naive_fw_const_impl!(SubConstRFwImpl, |x: f32, k: f32| { x - k });
define_naive_bw_const_impl!(SubConstRBwImpl, |_x: f32, _y: f32, gy: f32, _k: f32| { gy });
define_naive_fw_scalar_impl!(SubScalarLFwImpl, |x: f32, k: f32| { k - x });
define_naive_fw_scalar_impl!(SubScalarRFwImpl, |x: f32, k: f32| { x - k });

#[cfg(test)]
mod tests {
    use crate::devices as D;
    use crate::functions::BasicFunctions;

    #[test]
    fn check_sub_const_l_fw() {
        let x_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.0001];
        let k = 1.;
        let y_data = vec![-999., -99., -9., 0., 0.9, 0.99, 0.999, 0.9999];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("sub_const_l_fw_impl", &[&x], &[], &[k], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_sub_const_r_fw() {
        let x_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.0001];
        let k = 1.;
        let y_data = vec![999., 99., 9., 0., -0.9, -0.99, -0.999, -0.9999];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("sub_const_r_fw_impl", &[&x], &[], &[k], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_sub_scalar_l_fw() {
        let x_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.0001];
        let k_data = vec![10., 1.];
        let y_data = vec![-990., -90., 0., 9., 0.9, 0.99, 0.999, 0.9999];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let k = dev.new_tensor_by_slice(shape![; 2], &k_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("sub_scalar_l_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_sub_scalar_r_fw() {
        let x_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.0001];
        let k_data = vec![10., 1.];
        let y_data = vec![990., 90., 0., -9., -0.9, -0.99, -0.999, -0.9999];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let k = dev.new_tensor_by_slice(shape![; 2], &k_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("sub_scalar_r_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_sub_scalar_l_fw_batch_broadcast() {
        let dev = D::Naive::new();
        {
            let x_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.0001];
            let k_data = vec![1.];
            let y_data = vec![-999., -99., -9., 0., 0.9, 0.99, 0.999, 0.9999];
            let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
            let k = dev.new_tensor_by_slice(shape![], &k_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("sub_scalar_l_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
        {
            let x_data = vec![1000., 100., 10., 1.];
            let k_data = vec![10., 1.];
            let y_data = vec![-990., -90., 0., 9., -999., -99., -9., 0.];
            let x = dev.new_tensor_by_slice(shape![2, 2], &x_data);
            let k = dev.new_tensor_by_slice(shape![; 2], &k_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("sub_scalar_l_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
    }

    #[test]
    fn check_sub_scalar_r_fw_batch_broadcast() {
        let dev = D::Naive::new();
        {
            let x_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.0001];
            let k_data = vec![1.];
            let y_data = vec![999., 99., 9., 0., -0.9, -0.99, -0.999, -0.9999];
            let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
            let k = dev.new_tensor_by_slice(shape![], &k_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("sub_scalar_r_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
        {
            let x_data = vec![1000., 100., 10., 1.];
            let k_data = vec![10., 1.];
            let y_data = vec![990., 90., 0., -9., 999., 99., 9., 0.];
            let x = dev.new_tensor_by_slice(shape![2, 2], &x_data);
            let k = dev.new_tensor_by_slice(shape![; 2], &k_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("sub_scalar_r_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
    }

    #[test]
    fn check_sub_fw() {
        let a_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.0001];
        let b_data = vec![0., 100., 20., 3., 0.4, 0.05, 0.006, 0.0007];
        let y1_data = vec![1000., 0., -10., -2., -0.3, -0.04, -0.005, -0.0006];
        let y2_data = vec![-1000., 0., 10., 2., 0.3, 0.04, 0.005, 0.0006];
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_slice(shape![2, 2; 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
        let mut y1 = dev.new_tensor(shape![2, 2; 2]);
        y1.alloc();
        dev.call_fw_impl("sub_fw_impl", &[&a, &b], &[], &[], &mut [&mut y1]);
        assert_vector_ulps_eq!(y1_data, y1.to_vec());
        let mut y2 = dev.new_tensor(shape![2, 2; 2]);
        y2.alloc();
        dev.call_fw_impl("sub_fw_impl", &[&b, &a], &[], &[], &mut [&mut y2]);
        assert_vector_ulps_eq!(y2_data, y2.to_vec());
    }

    #[test]
    fn check_sub_fw_batch_broadcast() {
        let a_data = vec![0., 1., 2., 3.];
        let b_data = vec![0., 0., 0., 0., 4., 4., 4., 4.];
        let y1_data = vec![0., 1., 2., 3., -4., -3., -2., -1.];
        let y2_data = vec![0., -1., -2., -3., 4., 3., 2., 1.];
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_slice(shape![2, 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
        let mut y1 = dev.new_tensor(shape![2, 2; 2]);
        y1.alloc();
        dev.call_fw_impl("sub_fw_impl", &[&a, &b], &[], &[], &mut [&mut y1]);
        assert_vector_ulps_eq!(y1_data, y1.to_vec());
        let mut y2 = dev.new_tensor(shape![2, 2; 2]);
        y2.alloc();
        dev.call_fw_impl("sub_fw_impl", &[&b, &a], &[], &[], &mut [&mut y2]);
        assert_vector_ulps_eq!(y2_data, y2.to_vec());
    }

    #[test]
    fn check_sub_const_l_bw() {
        let gy_data = vec![1., -1., 2., -2., 2., -2., 1., -1.];
        let gx_data = vec![0., 2., -1., 3., -1., 3., 0., 2.];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_constant(shape![2, 2; 2], std::f32::NAN);
        let y = dev.new_tensor_by_constant(shape![2, 2; 2], std::f32::NAN);
        let gy = dev.new_tensor_by_slice(y.shape, &gy_data);
        let mut gx = dev.new_tensor_by_constant(x.shape, 1.);
        dev.call_bw_impl(
            "sub_const_l_bw_impl",
            &[&x],
            &[&y],
            &[&gy],
            &[],
            &[std::f32::NAN],
            &mut gx,
        );
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_sub_const_r_bw() {
        let gy_data = vec![1., -1., 2., -2., 2., -2., 1., -1.];
        let gx_data = vec![2., 0., 3., -1., 3., -1., 2., 0.];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_constant(shape![2, 2; 2], std::f32::NAN);
        let y = dev.new_tensor_by_constant(shape![2, 2; 2], std::f32::NAN);
        let gy = dev.new_tensor_by_slice(y.shape, &gy_data);
        let mut gx = dev.new_tensor_by_constant(x.shape, 1.);
        dev.call_bw_impl(
            "sub_const_r_bw_impl",
            &[&x],
            &[&y],
            &[&gy],
            &[],
            &[std::f32::NAN],
            &mut gx,
        );
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_sub_bw_11() {
        let gy_data = vec![1., -1., 2., -2.];
        let ga_data = vec![2., 0., 3., -1.];
        let gb_data = vec![0., 2., -1., 3.];
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_constant(shape![2; 2], std::f32::NAN);
        let b = dev.new_tensor_by_constant(shape![2; 2], std::f32::NAN);
        let y = dev.new_tensor_by_constant(shape![2; 2], std::f32::NAN);
        let gy = dev.new_tensor_by_slice(shape![2; 2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2; 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2; 2], 1.);
        dev.call_bw_impl("sub_bw_a_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut ga);
        dev.call_bw_impl("sub_bw_b_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut gb);
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }

    #[test]
    fn check_sub_bw_1n() {
        let gy_data = vec![1., -1., 2., -2.];
        let ga_data = vec![4., -2.];
        let gb_data = vec![0., 2., -1., 3.];
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_constant(shape![2], std::f32::NAN);
        let b = dev.new_tensor_by_constant(shape![2; 2], std::f32::NAN);
        let y = dev.new_tensor_by_constant(shape![2; 2], std::f32::NAN);
        let gy = dev.new_tensor_by_slice(shape![2; 2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2; 2], 1.);
        dev.call_bw_impl("sub_bw_a_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut ga);
        dev.call_bw_impl("sub_bw_b_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut gb);
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }

    #[test]
    fn check_sub_bw_n1() {
        let gy_data = vec![1., -1., 2., -2.];
        let ga_data = vec![2., 0., 3., -1.];
        let gb_data = vec![-2., 4.];
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_constant(shape![2; 2], std::f32::NAN);
        let b = dev.new_tensor_by_constant(shape![2], std::f32::NAN);
        let y = dev.new_tensor_by_constant(shape![2; 2], std::f32::NAN);
        let gy = dev.new_tensor_by_slice(shape![2; 2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2; 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2], 1.);
        dev.call_bw_impl("sub_bw_a_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut ga);
        dev.call_bw_impl("sub_bw_b_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut gb);
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }

    #[test]
    fn check_sub_bw_nn() {
        let gy_data = vec![1., -1., 2., -2.];
        let ga_data = vec![2., 0., 3., -1.];
        let gb_data = vec![0., 2., -1., 3.];
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_constant(shape![2; 2], std::f32::NAN);
        let b = dev.new_tensor_by_constant(shape![2; 2], std::f32::NAN);
        let y = dev.new_tensor_by_constant(shape![2; 2], std::f32::NAN);
        let gy = dev.new_tensor_by_slice(shape![2; 2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2; 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2; 2], 1.);
        dev.call_bw_impl("sub_bw_a_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut ga);
        dev.call_bw_impl("sub_bw_b_impl", &[&a, &b], &[&y], &[&gy], &[], &[], &mut gb);
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }
}
