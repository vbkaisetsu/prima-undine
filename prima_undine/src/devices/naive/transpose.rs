use crate::device_impl::{FunctionBwImpl, FunctionFwImpl};
use crate::functions::BasicFunctions;
use crate::Tensor;

define_empty_impl!(TransposeFwImpl);
impl FunctionFwImpl for TransposeFwImpl {
    fn call(&self, xs: &[&Tensor], _u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let y = &mut ys[0];
        let d1 = x.shape[0] as usize;
        let d2 = x.shape[1] as usize;
        let ms = d1 * d2 as usize;
        let bs = y.shape.batch();
        unsafe {
            let px = const_ptr!(x);
            let py = mut_ptr!(y);
            let mut shift_y = 0;
            let mut k = 0;
            for _ in 0..bs {
                let mut p = shift_y;
                for _ in 0..d2 {
                    let mut pp = p;
                    for _ in 0..d1 {
                        *py.add(pp) = *px.add(k);
                        pp += d2;
                        k += 1;
                    }
                    p += 1;
                }
                shift_y += ms;
            }
        }
    }
}

define_empty_impl!(TransposeBwImpl);
impl FunctionBwImpl for TransposeBwImpl {
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
        *gx += &gy.transpose();
    }
}

#[cfg(test)]
mod tests {
    use crate::devices as D;
    use crate::functions::BasicFunctions;

    #[test]
    fn check_transpose_fw_11() {
        let x_data = vec![42.];
        let y_data = vec![42.];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![], &x_data);
        let mut y = dev.new_tensor(shape![]);
        y.alloc();
        dev.call_fw_impl("transpose_fw_impl", &[&x], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_transpose_fw_n1() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let y_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![12], &x_data);
        let mut y = dev.new_tensor(shape![1, 12]);
        y.alloc();
        dev.call_fw_impl("transpose_fw_impl", &[&x], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_transpose_fw_1n() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let y_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![1, 3; 4], &x_data);
        let mut y = dev.new_tensor(shape![3; 4]);
        y.alloc();
        dev.call_fw_impl("transpose_fw_impl", &[&x], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_transpose_fw_nn() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let y_data = vec![1., 3., 2., 4., 5., 7., 6., 8., 9., 11., 10., 12.];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![2, 2; 3], &x_data);
        let mut y = dev.new_tensor(shape![2, 2; 3]);
        y.alloc();
        dev.call_fw_impl("transpose_fw_impl", &[&x], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_transpose_fw_mn() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let y_data = vec![1., 3., 5., 2., 4., 6., 7., 9., 11., 8., 10., 12.];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![2, 3; 2], &x_data);
        let mut y = dev.new_tensor(shape![3, 2; 2]);
        y.alloc();
        dev.call_fw_impl("transpose_fw_impl", &[&x], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_transpose_bw_11() {
        let gy_data = vec![42.];
        let gx_data = vec![43.];
        let dev = D::Naive::new();
        let gy = dev.new_tensor_by_slice(shape![], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![], 1.);
        dev.call_bw_impl("transpose_bw_impl", &[], &[], &[&gy], &[], &[], &mut gx);
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_transpose_bw_n1() {
        let gy_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let gx_data = vec![2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.];
        let dev = D::Naive::new();
        let gy = dev.new_tensor_by_slice(shape![12], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![1, 12], 1.);
        dev.call_bw_impl("transpose_bw_impl", &[], &[], &[&gy], &[], &[], &mut gx);
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_transpose_bw_1n() {
        let gy_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let gx_data = vec![2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.];
        let dev = D::Naive::new();
        let gy = dev.new_tensor_by_slice(shape![1, 12], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![12], 1.);
        dev.call_bw_impl("transpose_bw_impl", &[], &[], &[&gy], &[], &[], &mut gx);
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_transpose_bw_nn() {
        let gy_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let gx_data = vec![2., 4., 3., 5., 6., 8., 7., 9., 10., 12., 11., 13.];
        let dev = D::Naive::new();
        let gy = dev.new_tensor_by_slice(shape![2, 2; 3], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![2, 2; 3], 1.);
        dev.call_bw_impl("transpose_bw_impl", &[], &[], &[&gy], &[], &[], &mut gx);
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_transpose_bw_mn() {
        let gy_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let gx_data = vec![2., 4., 6., 3., 5., 7., 8., 10., 12., 9., 11., 13.];
        let dev = D::Naive::new();
        let gy = dev.new_tensor_by_slice(shape![2, 3; 2], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![3, 2; 2], 1.);
        dev.call_bw_impl("transpose_bw_impl", &[], &[], &[&gy], &[], &[], &mut gx);
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }
}
