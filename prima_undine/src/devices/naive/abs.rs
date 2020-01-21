define_naive_fw_x_impl!(AbsFwImpl, |x: f32| x.abs());
define_naive_bw_x_impl!(AbsBwImpl, |x: f32, _y: f32, gy: f32| gy
    * ((x > 0.) as i32 - (x < 0.) as i32) as f32);

#[cfg(test)]
mod tests {
    use crate::devices as D;
    use crate::functions::BasicFunctions;

    #[test]
    fn check_abs_fw() {
        let x_data = vec![0.25, 0.5, 0., 1., 2., 4., -0.25, -0.5, -0., -1., -2., -4.];
        let y_data = vec![0.25, 0.5, 0., 1., 2., 4., 0.25, 0.5, 0., 1., 2., 4.];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![2, 3; 2], &x_data);
        let mut y = dev.new_tensor(shape![2, 3; 2]);
        y.alloc();
        dev.call_fw_impl("abs_fw_impl", &[&x], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(&y_data, &y.to_vec());
    }

    #[test]
    fn check_abs_bw() {
        let x_data = vec![0.01, 0., 1., 2.5, -0.01, 0., -1., -2.5];
        let gy_data = vec![1., -1., 2., -2., 2., -2., 1., -1.];
        let gx_data = vec![2., 1., 3., -1., -1., 1., 0., 2.];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let y = dev.new_tensor_by_constant(shape![2, 2; 2], std::f32::NAN);
        let gy = dev.new_tensor_by_slice(shape![2, 2; 2], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![2, 2; 2], 1.);
        dev.call_bw_impl("abs_bw_impl", &[&x], &[&y], &[&gy], &[], &[], &mut gx);
        assert_vector_ulps_eq!(&gx_data, &gx.to_vec());
    }
}
