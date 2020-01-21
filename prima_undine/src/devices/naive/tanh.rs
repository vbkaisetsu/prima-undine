define_naive_fw_x_impl!(TanhFwImpl, |x: f32| x.tanh());
define_naive_bw_x_impl!(TanhBwImpl, |_x: f32, y: f32, gy: f32| gy * (1. - y * y));

#[cfg(test)]
mod tests {
    use crate::devices as D;
    use crate::functions::BasicFunctions;

    #[test]
    fn check_tanh_fw() {
        let y_f = |x: f64| x.tanh();
        let x_data = vec![0., 0.5, 1., 2., 4., 8., 0., -0.5, -1., -2., -4., -8.];
        let y_data = generate_fw_testset!(x_data, y_f);
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![2, 3; 2], &x_data);
        let mut y = dev.new_tensor(shape![2, 3; 2]);
        y.alloc();
        dev.call_fw_impl("tanh_fw_impl", &[&x], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_tanh_bw() {
        let y_f = |x: f64| x.tanh();
        let gx_f = |_x: f64, y: f64, gy: f64| 1. + gy * (1. - y * y);
        let x_data = vec![0., 1., 2., 3., 0., -1., -2., -3.];
        let gy_data = vec![1., -1., 2., -2., 2., -2., 1., -1.];
        let (y_data, gx_data) = generate_bw_testset!(x_data, gy_data, y_f, gx_f);
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let y = dev.new_tensor_by_slice(shape![2, 2; 2], &y_data);
        let gy = dev.new_tensor_by_slice(shape![2, 2; 2], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![2, 2; 2], 1.);
        dev.call_bw_impl("tanh_bw_impl", &[&x], &[&y], &[&gy], &[], &[], &mut gx);
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }
}
