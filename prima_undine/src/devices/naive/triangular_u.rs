use crate::device_impl::{FunctionBwImpl, FunctionFwImpl};
use crate::functions::BasicFunctions;
use crate::Tensor;

define_empty_impl!(TriangularUFwImpl);
impl FunctionFwImpl for TriangularUFwImpl {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let y = &mut ys[0];
        let k = u32data[0];
        let size = x.shape[0];
        let bs = y.shape.batch();
        unsafe {
            let px = const_ptr!(x);
            let py = mut_ptr!(y);
            let mut p = 0;
            for _ in 0..bs {
                for j in 0..size {
                    for i in 0..size {
                        *py.add(p) = if j >= i + k { *px.add(p) } else { 0. };
                        p += 1;
                    }
                }
            }
        }
    }
}

define_empty_impl!(TriangularUBwImpl);
impl FunctionBwImpl for TriangularUBwImpl {
    fn call(
        &self,
        _xs: &[&Tensor],
        _ys: &[&Tensor],
        gys: &[&Tensor],
        u32data: &[u32],
        _f32data: &[f32],
        gx: &mut Tensor,
    ) {
        let gy = gys[0];
        let k = u32data[0];
        *gx += &gy.triangular_u(k);
    }
}

#[cfg(test)]
mod tests {
    use crate::devices as D;
    use crate::functions::BasicFunctions;

    #[test]
    fn check_triangular_u_fw_11() {
        let x_data = vec![42.];
        let y_data = vec![42.];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![], &x_data);
        let mut y = dev.new_tensor(shape![]);
        y.alloc();
        dev.call_fw_impl("triangular_u_fw_impl", &[&x], &[0], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_triangular_u_fw_nn() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
        let y_data = vec![1., 0., 0., 4., 5., 0., 7., 8., 9.];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![3, 3], &x_data);
        let mut y = dev.new_tensor(shape![3, 3]);
        y.alloc();
        dev.call_fw_impl("triangular_u_fw_impl", &[&x], &[0], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_triangular_u_fw_nn_k1() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
        let y_data = vec![0., 0., 0., 4., 0., 0., 7., 8., 0.];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![3, 3], &x_data);
        let mut y = dev.new_tensor(shape![3, 3]);
        y.alloc();
        dev.call_fw_impl("triangular_u_fw_impl", &[&x], &[1], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_triangular_u_fw_nn_k2() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
        let y_data = vec![0., 0., 0., 0., 0., 0., 7., 0., 0.];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![3, 3], &x_data);
        let mut y = dev.new_tensor(shape![3, 3]);
        y.alloc();
        dev.call_fw_impl("triangular_u_fw_impl", &[&x], &[2], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_triangular_u_fw_batch_nn() {
        let x_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
        ];
        let y_data = vec![
            1., 0., 0., 4., 5., 0., 7., 8., 9., 10., 0., 0., 13., 14., 0., 16., 17., 18.,
        ];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![3, 3; 2], &x_data);
        let mut y = dev.new_tensor(shape![3, 3; 2]);
        y.alloc();
        dev.call_fw_impl("triangular_u_fw_impl", &[&x], &[0], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_triangular_u_fw_batch_nn_k1() {
        let x_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
        ];
        let y_data = vec![
            0., 0., 0., 4., 0., 0., 7., 8., 0., 0., 0., 0., 13., 0., 0., 16., 17., 0.,
        ];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![3, 3; 2], &x_data);
        let mut y = dev.new_tensor(shape![3, 3; 2]);
        y.alloc();
        dev.call_fw_impl("triangular_u_fw_impl", &[&x], &[1], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_triangular_u_bw_11() {
        let gy_data = vec![42.];
        let gx_data = vec![43.];
        let dev = D::Naive::new();
        let gy = dev.new_tensor_by_slice(shape![], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![], 1.);
        dev.call_bw_impl("triangular_u_bw_impl", &[], &[], &[&gy], &[0], &[], &mut gx);
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_triangular_u_bw_nn() {
        let gy_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
        let gx_data = vec![2., 1., 1., 5., 6., 1., 8., 9., 10.];
        let dev = D::Naive::new();
        let gy = dev.new_tensor_by_slice(shape![3, 3], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![3, 3], 1.);
        dev.call_bw_impl("triangular_u_bw_impl", &[], &[], &[&gy], &[0], &[], &mut gx);
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_triangular_u_bw_nn_k1() {
        let gy_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
        let gx_data = vec![1., 1., 1., 5., 1., 1., 8., 9., 1.];
        let dev = D::Naive::new();
        let gy = dev.new_tensor_by_slice(shape![3, 3], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![3, 3], 1.);
        dev.call_bw_impl("triangular_u_bw_impl", &[], &[], &[&gy], &[1], &[], &mut gx);
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_triangular_u_bw_nn_k2() {
        let gy_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9.];
        let gx_data = vec![1., 1., 1., 1., 1., 1., 8., 1., 1.];
        let dev = D::Naive::new();
        let gy = dev.new_tensor_by_slice(shape![3, 3], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![3, 3], 1.);
        dev.call_bw_impl("triangular_u_bw_impl", &[], &[], &[&gy], &[2], &[], &mut gx);
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_triangular_u_bw_batch_nn() {
        let gy_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
        ];
        let gx_data = vec![
            2., 1., 1., 5., 6., 1., 8., 9., 10., 11., 1., 1., 14., 15., 1., 17., 18., 19.,
        ];
        let dev = D::Naive::new();
        let gy = dev.new_tensor_by_slice(shape![3, 3; 2], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![3, 3; 2], 1.);
        dev.call_bw_impl("triangular_u_bw_impl", &[], &[], &[&gy], &[0], &[], &mut gx);
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_triangular_u_bw_batch_nn_k1() {
        let gy_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
        ];
        let gx_data = vec![
            1., 1., 1., 5., 1., 1., 8., 9., 1., 1., 1., 1., 14., 1., 1., 17., 18., 1.,
        ];
        let dev = D::Naive::new();
        let gy = dev.new_tensor_by_slice(shape![3, 3; 2], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![3, 3; 2], 1.);
        dev.call_bw_impl("triangular_u_bw_impl", &[], &[], &[&gy], &[1], &[], &mut gx);
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }
}
