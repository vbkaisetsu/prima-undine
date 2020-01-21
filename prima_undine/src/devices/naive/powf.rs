use crate::device_impl::FunctionBwImpl;
use crate::Tensor;

define_naive_fw_ab_impl!(PowfFwImpl, |a: f32, b: f32| { a.powf(b) });

define_empty_impl!(PowfBwAImpl);
impl FunctionBwImpl for PowfBwAImpl {
    fn call(
        &self,
        xs: &[&Tensor],
        _ys: &[&Tensor],
        gys: &[&Tensor],
        _u32data: &[u32],
        _f32data: &[f32],
        gx: &mut Tensor,
    ) {
        let a = xs[0];
        let b = xs[1];
        let gy = gys[0];
        let ga = gx;
        let volume = gy.shape.volume() as usize;
        let a_shift = if a.shape.batch() == 1 { 0 } else { volume };
        let b_shift = if b.shape.batch() == 1 { 0 } else { volume };
        unsafe {
            let mut pa = const_ptr!(a);
            let mut pb = const_ptr!(b);
            let mut pgy = const_ptr!(gy);
            let mut pga = mut_ptr!(ga);
            for _ in 0..gy.shape.batch() {
                for i in 0..volume {
                    *pga.add(i) += *pgy.add(i) * *pb.add(i) * (*pa.add(i)).powf(*pb.add(i) - 1.);
                }
                pa = pa.add(a_shift);
                pb = pb.add(b_shift);
                pgy = pgy.add(volume);
                pga = pga.add(a_shift);
            }
        }
    }
}

define_empty_impl!(PowfBwBImpl);
impl FunctionBwImpl for PowfBwBImpl {
    fn call(
        &self,
        xs: &[&Tensor],
        ys: &[&Tensor],
        gys: &[&Tensor],
        _u32data: &[u32],
        _f32data: &[f32],
        gx: &mut Tensor,
    ) {
        let a = xs[0];
        let y = ys[0];
        let gy = gys[0];
        let gb = gx;
        let volume = y.shape.volume() as usize;
        let a_shift = if a.shape.batch() == 1 { 0 } else { volume };
        let b_shift = if gb.shape.batch() == 1 { 0 } else { volume };
        unsafe {
            let mut pa = const_ptr!(a);
            let mut py = const_ptr!(y);
            let mut pgy = const_ptr!(gy);
            let mut pgb = mut_ptr!(gb);
            for _ in 0..y.shape.batch() {
                for i in 0..volume {
                    *pgb.add(i) += *pgy.add(i) * (*pa.add(i)).ln() * *py.add(i);
                }
                pa = pa.add(a_shift);
                py = py.add(volume);
                pgy = pgy.add(volume);
                pgb = pgb.add(b_shift);
            }
        }
    }
}

define_naive_fw_const_impl!(PowfConstLFwImpl, |x: f32, k: f32| { k.powf(x) });
define_naive_bw_const_impl!(PowfConstLBwImpl, |_x: f32, y: f32, gy: f32, k: f32| {
    gy * k.ln() * y
});
define_naive_fw_const_impl!(PowfConstRFwImpl, |x: f32, k: f32| { x.powf(k) });
define_naive_bw_const_impl!(PowfConstRBwImpl, |x: f32, _y: f32, gy: f32, k: f32| {
    gy * k * x.powf(k - 1.)
});
define_naive_fw_scalar_impl!(PowfScalarLFwImpl, |x: f32, k: f32| { k.powf(x) });
define_naive_fw_scalar_impl!(PowfScalarRFwImpl, |x: f32, k: f32| { x.powf(k) });

#[cfg(test)]
mod tests {
    use crate::devices as D;
    use crate::functions::BasicFunctions;

    #[test]
    fn check_powf_const_r_fw() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let k = 3.;
        let y_data = vec![1., 8., 27., 64., 125., 216., 343., 512.];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("powf_const_r_fw_impl", &[&x], &[], &[k], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_powf_const_l_fw() {
        let x_data = vec![3., 2., 1., 0., -1., -2., -3., -4.];
        let k = 3.;
        let y_data = vec![27., 9., 3., 1., 1. / 3., 1. / 9., 1. / 27., 1. / 81.];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("powf_const_l_fw_impl", &[&x], &[], &[k], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_powf_scalar_r_fw() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let k_data = vec![3., -3.];
        let y_data = vec![1., 8., 27., 64., 1. / 125., 1. / 216., 1. / 343., 1. / 512.];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let k = dev.new_tensor_by_slice(shape![; 2], &k_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("powf_scalar_r_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_powf_scalar_l_fw() {
        let x_data = vec![3., 2., 1., 0., -1., -2., -3., -4.];
        let k_data = vec![2., 3.];
        let y_data = vec![8., 4., 2., 1., 1. / 3., 1. / 9., 1. / 27., 1. / 81.];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let k = dev.new_tensor_by_slice(shape![; 2], &k_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("powf_scalar_l_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_powf_scalar_r_fw_batch_broadcast() {
        let dev = D::Naive::new();
        {
            let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8.];
            let k_data = vec![3.];
            let y_data = vec![1., 8., 27., 64., 125., 216., 343., 512.];
            let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
            let k = dev.new_tensor_by_slice(shape![], &k_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("powf_scalar_r_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
        {
            let x_data = vec![1., 2., 3., 4.];
            let k_data = vec![3., -3.];
            let y_data = vec![1., 8., 27., 64., 1., 1. / 8., 1. / 27., 1. / 64.];
            let x = dev.new_tensor_by_slice(shape![2, 2], &x_data);
            let k = dev.new_tensor_by_slice(shape![; 2], &k_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("powf_scalar_r_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
    }

    #[test]
    fn check_powf_const_l_fw_batch_broadcast() {
        let dev = D::Naive::new();
        {
            let x_data = vec![3., 2., 1., 0., -1., -2., -3., -4.];
            let k_data = vec![3.];
            let y_data = vec![27., 9., 3., 1., 1. / 3., 1. / 9., 1. / 27., 1. / 81.];
            let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
            let k = dev.new_tensor_by_slice(shape![], &k_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("powf_scalar_l_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
        {
            let x_data = vec![3., 2., 1., 0.];
            let k_data = vec![2., 3.];
            let y_data = vec![8., 4., 2., 1., 27., 9., 3., 1.];
            let x = dev.new_tensor_by_slice(shape![2, 2], &x_data);
            let k = dev.new_tensor_by_slice(shape![; 2], &k_data);
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("powf_scalar_l_fw_impl", &[&x, &k], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
    }

    #[test]
    fn check_powf_fw() {
        let a_data = vec![0., 1., 2., 3., 0., 1., 2., 3.];
        let b_data = vec![2., 2., 2., 2., 3., 3., 3., 3.];
        let y1_data = vec![0., 1., 4., 9., 0., 1., 8., 27.];
        let y2_data = vec![1., 2., 4., 8., 1., 3., 9., 27.];
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_slice(shape![2, 2; 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
        let mut y1 = dev.new_tensor(shape![2, 2; 2]);
        y1.alloc();
        dev.call_fw_impl("powf_fw_impl", &[&a, &b], &[], &[], &mut [&mut y1]);
        assert_vector_ulps_eq!(y1_data, y1.to_vec());
        let mut y2 = dev.new_tensor(shape![2, 2; 2]);
        y2.alloc();
        dev.call_fw_impl("powf_fw_impl", &[&b, &a], &[], &[], &mut [&mut y2]);
        assert_vector_ulps_eq!(y2_data, y2.to_vec());
    }

    #[test]
    fn check_powf_fw_batch_broadcast() {
        let a_data = vec![0., 1., 2., 3.];
        let b_data = vec![2., 2., 2., 2., 3., 3., 3., 3.];
        let y1_data = vec![0., 1., 4., 9., 0., 1., 8., 27.];
        let y2_data = vec![1., 2., 4., 8., 1., 3., 9., 27.];
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_slice(shape![2, 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
        let mut y1 = dev.new_tensor(shape![2, 2; 2]);
        y1.alloc();
        dev.call_fw_impl("powf_fw_impl", &[&a, &b], &[], &[], &mut [&mut y1]);
        assert_vector_ulps_eq!(y1_data, y1.to_vec());
        let mut y2 = dev.new_tensor(shape![2, 2; 2]);
        y2.alloc();
        dev.call_fw_impl("powf_fw_impl", &[&b, &a], &[], &[], &mut [&mut y2]);
        assert_vector_ulps_eq!(y2_data, y2.to_vec());
    }

    #[test]
    fn check_powf_const_r_bw() {
        let x_data = vec![1., 2., 4., 8., 16., 32., 64., 128.];
        let ks = vec![1., 0.5, 0.25, 0. - 0.125, -0.25, -0.5, -0.1];
        let gy_data = vec![1., -1., 2., -2., 1., -1., 2., -2.];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let gy = dev.new_tensor_by_slice(shape![2, 2; 2], &gy_data);
        for &k in &ks {
            let y_data = x_data.iter().map(|&x| x.powf(k)).collect::<Vec<f32>>();
            let gx_data = x_data
                .iter()
                .zip(&gy_data)
                .map(|(&x, &gy)| 1. + gy * k * x.powf(k - 1.))
                .collect::<Vec<f32>>();
            let y = dev.new_tensor_by_slice(shape![2, 2; 2], &y_data);
            let mut gx = dev.new_tensor_by_constant(shape![2, 2; 2], 1.);
            dev.call_bw_impl(
                "powf_const_r_bw_impl",
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
    fn check_powf_const_l_bw() {
        let x_data = vec![1., 0.5, 0.25, 0., -0.125, -0.25, -0.5, -1.];
        let ks: Vec<f32> = vec![1., 2., 4., 8., 16., 32., 64., 128.];
        let gy_data = vec![1., -1., 2., -2., 1., -1., 2., -2.];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let gy = dev.new_tensor_by_slice(shape![2, 2; 2], &gy_data);
        for &k in &ks {
            let y_data = x_data.iter().map(|&x| k.powf(x)).collect::<Vec<f32>>();
            let gx_data = x_data
                .iter()
                .zip(&gy_data)
                .map(|(&x, &gy)| 1. + gy * k.ln() * k.powf(x))
                .collect::<Vec<f32>>();
            let y = dev.new_tensor_by_slice(shape![2, 2; 2], &y_data);
            let mut gx = dev.new_tensor_by_constant(shape![2, 2; 2], 1.);
            dev.call_bw_impl(
                "powf_const_l_bw_impl",
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
    fn check_powf_bw_11() {
        let a_data = vec![1., 2., 4., 8.];
        let b_data = vec![2., 1., 0., -1.];
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_slice(shape![2, 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2], &b_data);
        let y_data = a_data
            .iter()
            .zip(&b_data)
            .map(|(&a, &b)| a.powf(b))
            .collect::<Vec<f32>>();
        let y = dev.new_tensor_by_slice(shape![2, 2], &y_data);
        let gy_data = vec![1., -1., 2., -2.];
        let gy = dev.new_tensor_by_slice(shape![2, 2], &gy_data);
        let ga_data = a_data
            .iter()
            .zip(&b_data)
            .zip(&gy_data)
            .map(|((&a, &b), &gy)| 1. + gy * b * a.powf(b - 1.))
            .collect::<Vec<f32>>();
        let gb_data = a_data
            .iter()
            .zip(&b_data)
            .zip(&gy_data)
            .map(|((&a, &b), &gy)| 1. + gy * a.ln() * a.powf(b))
            .collect::<Vec<f32>>();
        let mut ga = dev.new_tensor_by_constant(shape![2, 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2, 2], 1.);
        dev.call_bw_impl(
            "powf_bw_a_impl",
            &[&a, &b],
            &[&y],
            &[&gy],
            &[],
            &[],
            &mut ga,
        );
        dev.call_bw_impl(
            "powf_bw_b_impl",
            &[&a, &b],
            &[&y],
            &[&gy],
            &[],
            &[],
            &mut gb,
        );
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }

    #[test]
    fn check_powf_bw_1n() {
        let a_data: Vec<f32> = vec![1., 2., 4., 8.];
        let b_data = vec![3., 2., 1., 0., -1., -2., -3., -4.];
        let mut y_data = vec![0.; 8];
        let gy_data = vec![1., -1., 2., -2., 1., -1., 2., -2.];
        let mut ga_data = vec![1.; 4];
        let mut gb_data = vec![1.; 8];
        for ib in 0..b_data.len() {
            let ia = ib % a_data.len();
            y_data[ib] = a_data[ia].powf(b_data[ib]);
            ga_data[ia] += gy_data[ib] * b_data[ib] * a_data[ia].powf(b_data[ib] - 1.);
            gb_data[ib] += gy_data[ib] * a_data[ia].ln() * a_data[ia].powf(b_data[ib]);
        }
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_slice(shape![2, 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
        let y = dev.new_tensor_by_slice(shape![2, 2; 2], &y_data);
        let gy = dev.new_tensor_by_slice(shape![2, 2; 2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2, 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2, 2; 2], 1.);
        dev.call_bw_impl(
            "powf_bw_a_impl",
            &[&a, &b],
            &[&y],
            &[&gy],
            &[],
            &[],
            &mut ga,
        );
        dev.call_bw_impl(
            "powf_bw_b_impl",
            &[&a, &b],
            &[&y],
            &[&gy],
            &[],
            &[],
            &mut gb,
        );
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }

    #[test]
    fn check_powf_bw_n1() {
        let a_data: Vec<f32> = vec![1., 2., 4., 8., 16., 32., 64., 128.];
        let b_data = vec![2., 1., 0., -1.];
        let mut y_data = vec![0.; 8];
        let gy_data = vec![1., -1., 2., -2., 1., -1., 2., -2.];
        let mut ga_data = vec![1.; 8];
        let mut gb_data = vec![1.; 4];
        for ia in 0..a_data.len() {
            let ib = ia % b_data.len();
            y_data[ia] = a_data[ia].powf(b_data[ib]);
            ga_data[ia] += gy_data[ia] * b_data[ib] * a_data[ia].powf(b_data[ib] - 1.);
            gb_data[ib] += gy_data[ia] * a_data[ia].ln() * a_data[ia].powf(b_data[ib]);
        }
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_slice(shape![2, 2; 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2], &b_data);
        let y = dev.new_tensor_by_slice(shape![2, 2; 2], &y_data);
        let gy = dev.new_tensor_by_slice(shape![2, 2; 2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2, 2; 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2, 2], 1.);
        dev.call_bw_impl(
            "powf_bw_a_impl",
            &[&a, &b],
            &[&y],
            &[&gy],
            &[],
            &[],
            &mut ga,
        );
        dev.call_bw_impl(
            "powf_bw_b_impl",
            &[&a, &b],
            &[&y],
            &[&gy],
            &[],
            &[],
            &mut gb,
        );
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }

    #[test]
    fn check_powf_bw_nn() {
        let a_data: Vec<f32> = vec![1., 2., 4., 8., 16., 32., 64., 128.];
        let b_data = vec![1., 0.5, 0.25, 0., -0.125, -0.25, -0.5, -1.];
        let y_data = a_data
            .iter()
            .zip(&b_data)
            .map(|(&a, &b)| a.powf(b))
            .collect::<Vec<f32>>();
        let gy_data = vec![1., -1., 2., -2., 1., -1., 2., -2.];
        let ga_data = a_data
            .iter()
            .zip(&b_data)
            .zip(&gy_data)
            .map(|((&a, &b), &gy)| 1. + gy * b * a.powf(b - 1.))
            .collect::<Vec<f32>>();
        let gb_data = a_data
            .iter()
            .zip(&b_data)
            .zip(&gy_data)
            .map(|((&a, &b), &gy)| 1. + gy * a.ln() * a.powf(b))
            .collect::<Vec<f32>>();
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_slice(shape![2, 2; 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
        let y = dev.new_tensor_by_slice(shape![2, 2; 2], &y_data);
        let gy = dev.new_tensor_by_slice(shape![2, 2; 2], &gy_data);
        let mut ga = dev.new_tensor_by_constant(shape![2, 2; 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2, 2; 2], 1.);
        dev.call_bw_impl(
            "powf_bw_a_impl",
            &[&a, &b],
            &[&y],
            &[&gy],
            &[],
            &[],
            &mut ga,
        );
        dev.call_bw_impl(
            "powf_bw_b_impl",
            &[&a, &b],
            &[&y],
            &[&gy],
            &[],
            &[],
            &mut gb,
        );
        assert_vector_ulps_eq!(ga_data, ga.to_vec());
        assert_vector_ulps_eq!(gb_data, gb.to_vec());
    }
}
