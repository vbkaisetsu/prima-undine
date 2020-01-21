use crate::device_impl::{FunctionBwImpl, FunctionFwImpl};
use crate::Tensor;

define_empty_impl!(PermuteDimsFwImpl);
impl FunctionFwImpl for PermuteDimsFwImpl {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let perm = u32data;
        let y = &mut ys[0];
        let volume = x.shape.volume() as usize;
        let bs = x.shape.batch();
        let ndims = perm.len();
        let mut x_strides = vec![0; ndims];
        let mut y_strides = vec![0; ndims];
        let mut x_stride_tmp = 1;
        let mut y_stride_tmp = 1;
        for i in 0..ndims {
            x_strides[ndims - i - 1] = x_stride_tmp as usize;
            y_strides[ndims - perm[i] as usize - 1] = y_stride_tmp as usize;
            x_stride_tmp *= x.shape[i as u32] as usize;
            y_stride_tmp *= y.shape[i as u32] as usize;
        }
        unsafe {
            let mut src = const_ptr!(x);
            let mut dest = mut_ptr!(y);
            for _ in 0..bs {
                for i in 0..volume {
                    let mut tmp = i;
                    let mut j = 0;
                    for d in 0..ndims {
                        let p = tmp / x_strides[d];
                        tmp -= p * x_strides[d];
                        j += p * y_strides[d];
                    }
                    *dest.add(j) = *src.add(i);
                }
                src = src.add(volume);
                dest = dest.add(volume);
            }
        }
    }
}

define_empty_impl!(PermuteDimsBwImpl);
impl FunctionBwImpl for PermuteDimsBwImpl {
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
        let perm = u32data;
        let volume = gx.shape.volume() as usize;
        let bs = gx.shape.batch();
        let ndims = perm.len();
        let mut x_strides = vec![0; ndims];
        let mut y_strides = vec![0; ndims];
        let mut x_stride_tmp = 1;
        let mut y_stride_tmp = 1;
        for i in 0..ndims {
            x_strides[ndims - i - 1] = x_stride_tmp;
            y_strides[ndims - perm[i] as usize - 1] = y_stride_tmp as usize;
            x_stride_tmp *= gx.shape[i as u32] as usize;
            y_stride_tmp *= gy.shape[i as u32] as usize;
        }
        unsafe {
            let mut src = const_ptr!(gy);
            let mut dest = mut_ptr!(gx);
            for _ in 0..bs {
                for i in 0..volume {
                    let mut tmp = i;
                    let mut j = 0;
                    for d in 0..ndims {
                        let p = tmp / x_strides[d];
                        tmp -= p * x_strides[d];
                        j += p * y_strides[d];
                    }
                    *dest.add(i) += *src.add(j);
                }
                src = src.add(volume);
                dest = dest.add(volume);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::devices as D;
    use crate::functions::BasicFunctions;

    #[test]
    fn check_permute_dims_fw_111() {
        let x_data = vec![42., 43.];
        let y_data = vec![42., 43.];
        let perm = vec![];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![; 2], &x_data);
        let mut y = dev.new_tensor(shape![; 2]);
        y.alloc();
        dev.call_fw_impl("permute_dims_fw_impl", &[&x], &perm, &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_permute_dims_fw_n11() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let y_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let perm = vec![1, 2, 0];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![6; 2], &x_data);
        let mut y = dev.new_tensor(shape![1, 1, 6; 2]);
        y.alloc();
        dev.call_fw_impl("permute_dims_fw_impl", &[&x], &perm, &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_permute_dims_fw_1n1() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let y_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let perm = vec![0, 2, 1];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![1, 4; 3], &x_data);
        let mut y = dev.new_tensor(shape![1, 1, 4; 3]);
        y.alloc();
        dev.call_fw_impl("permute_dims_fw_impl", &[&x], &perm, &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_permute_dims_fw_11n() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let y_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let perm = vec![2, 0, 1];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![1, 1, 4; 3], &x_data);
        let mut y = dev.new_tensor(shape![4; 3]);
        y.alloc();
        dev.call_fw_impl("permute_dims_fw_impl", &[&x], &perm, &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_permute_dims_fw_mn1() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let y_data = vec![1., 3., 5., 2., 4., 6., 7., 9., 11., 8., 10., 12.];
        let perm = vec![1, 2, 0];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![2, 3; 2], &x_data);
        let mut y = dev.new_tensor(shape![3, 1, 2; 2]);
        y.alloc();
        dev.call_fw_impl("permute_dims_fw_impl", &[&x], &perm, &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_permute_dims_fw_m1n() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let y_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let perm = vec![0, 2, 1];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![3, 1, 2; 2], &x_data);
        let mut y = dev.new_tensor(shape![3, 2; 2]);
        y.alloc();
        dev.call_fw_impl("permute_dims_fw_impl", &[&x], &perm, &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_permute_dims_fw_1mn() {
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let y_data = vec![1., 4., 2., 5., 3., 6., 7., 10., 8., 11., 9., 12.];
        let perm = vec![2, 0, 1];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![1, 3, 2; 2], &x_data);
        let mut y = dev.new_tensor(shape![2, 1, 3; 2]);
        y.alloc();
        dev.call_fw_impl("permute_dims_fw_impl", &[&x], &perm, &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_permute_dims_fw_lmn() {
        let x_data = vec![
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
            20., 21., 22., 23., 24.,
        ];
        let y_data = vec![
            1., 7., 2., 8., 3., 9., 4., 10., 5., 11., 6., 12., 13., 19., 14., 20., 15., 21., 16.,
            22., 17., 23., 18., 24.,
        ];
        let perm = vec![2, 0, 1];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![2, 3, 2; 2], &x_data);
        let mut y = dev.new_tensor(shape![2, 2, 3; 2]);
        y.alloc();
        dev.call_fw_impl("permute_dims_fw_impl", &[&x], &perm, &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_permute_dims_bw_111() {
        let gy_data = vec![42., 43.];
        let gx_data = vec![43., 44.];
        let perm = vec![];
        let dev = D::Naive::new();
        let gy = dev.new_tensor_by_slice(shape![; 2], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![; 2], 1.);
        dev.call_bw_impl(
            "permute_dims_bw_impl",
            &[],
            &[],
            &[&gy],
            &perm,
            &[],
            &mut gx,
        );
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_permute_dims_bw_n11() {
        let gy_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let gx_data = vec![2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.];
        let perm = vec![1, 0];
        let dev = D::Naive::new();
        let gy = dev.new_tensor_by_slice(shape![1, 6; 2], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![6; 2], 1.);
        dev.call_bw_impl(
            "permute_dims_bw_impl",
            &[],
            &[],
            &[&gy],
            &perm,
            &[],
            &mut gx,
        );
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_permute_dims_bw_1n1() {
        let gy_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let gx_data = vec![2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.];
        let perm = vec![0, 2, 1];
        let dev = D::Naive::new();
        let gy = dev.new_tensor_by_slice(shape![1, 1, 4; 3], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![1, 4; 3], 1.);
        dev.call_bw_impl(
            "permute_dims_bw_impl",
            &[],
            &[],
            &[&gy],
            &perm,
            &[],
            &mut gx,
        );
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_permute_dims_bw_11n() {
        let gy_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let gx_data = vec![2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.];
        let perm = vec![2, 0, 1];
        let dev = D::Naive::new();
        let gy = dev.new_tensor_by_slice(shape![1, 1, 4; 3], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![4; 3], 1.);
        dev.call_bw_impl(
            "permute_dims_bw_impl",
            &[],
            &[],
            &[&gy],
            &perm,
            &[],
            &mut gx,
        );
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_permute_dims_bw_mn1() {
        let gy_data = vec![1., 3., 5., 2., 4., 6., 7., 9., 11., 8., 10., 12.];
        let gx_data = vec![2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.];
        let perm = vec![1, 2, 0];
        let dev = D::Naive::new();
        let gy = dev.new_tensor_by_slice(shape![3, 1, 2; 2], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![2, 3; 2], 1.);
        dev.call_bw_impl(
            "permute_dims_bw_impl",
            &[],
            &[],
            &[&gy],
            &perm,
            &[],
            &mut gx,
        );
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_permute_dims_bw_m1n() {
        let gy_data = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let gx_data = vec![2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.];
        let perm = vec![0, 2, 1];
        let dev = D::Naive::new();
        let gy = dev.new_tensor_by_slice(shape![3, 2; 2], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![3, 1, 2; 2], 1.);
        dev.call_bw_impl(
            "permute_dims_bw_impl",
            &[],
            &[],
            &[&gy],
            &perm,
            &[],
            &mut gx,
        );
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_permute_dims_bw_1mn() {
        let gy_data = vec![1., 4., 2., 5., 3., 6., 7., 10., 8., 11., 9., 12.];
        let gx_data = vec![2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.];
        let perm = vec![2, 0, 1];
        let dev = D::Naive::new();
        let gy = dev.new_tensor_by_slice(shape![2, 1, 3; 2], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![1, 3, 2; 2], 1.);
        dev.call_bw_impl(
            "permute_dims_bw_impl",
            &[],
            &[],
            &[&gy],
            &perm,
            &[],
            &mut gx,
        );
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }

    #[test]
    fn check_permute_dims_bw_lmn() {
        let gy_data = vec![
            1., 7., 2., 8., 3., 9., 4., 10., 5., 11., 6., 12., 13., 19., 14., 20., 15., 21., 16.,
            22., 17., 23., 18., 24.,
        ];
        let gx_data = vec![
            2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.,
            21., 22., 23., 24., 25.,
        ];
        let perm = vec![2, 0, 1];
        let dev = D::Naive::new();
        let gy = dev.new_tensor_by_slice(shape![2, 2, 3; 2], &gy_data);
        let mut gx = dev.new_tensor_by_constant(shape![2, 3, 2; 2], 1.);
        dev.call_bw_impl(
            "permute_dims_bw_impl",
            &[],
            &[],
            &[&gy],
            &perm,
            &[],
            &mut gx,
        );
        assert_vector_ulps_eq!(gx_data, gx.to_vec());
    }
}
