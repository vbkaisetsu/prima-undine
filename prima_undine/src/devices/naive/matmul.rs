use std::cmp;

use crate::device_impl::{FunctionBwImpl, FunctionFwImpl};
use crate::functions::BasicDeviceFunctions;
use crate::Tensor;

define_empty_impl!(MatmulFwImpl);
impl FunctionFwImpl for MatmulFwImpl {
    fn call(&self, xs: &[&Tensor], _u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let a = xs[0];
        let b = xs[1];
        let y = &mut ys[0];
        let d1 = a.shape[0] as usize;
        let d2 = a.shape[1] as usize;
        let d3 = b.shape[1] as usize;
        let bs = y.shape.batch();
        let size = d1 * d3;
        let skip_a = if a.shape.has_batch() { d1 * d2 } else { 0 };
        let skip_b = if b.shape.has_batch() { d2 * d3 } else { 0 };
        unsafe {
            let mut pa = const_ptr!(a);
            let mut pb = const_ptr!(b);
            let mut py = mut_ptr!(y);
            for _ in 0..bs {
                for n in 0..size {
                    *py.add(n) = 0.0;
                }
                for k in (0..d3).step_by(8) {
                    let ek = cmp::min(k + 8, d3);
                    for i in (0..d1).step_by(8) {
                        let ei = cmp::min(i + 8, d1);
                        for j in (0..d2).step_by(8) {
                            let ej = cmp::min(j + 8, d2);
                            for kk in k..ek {
                                let kk_d1 = kk * d1;
                                let kk_d2 = kk * d2;
                                for ii in i..ei {
                                    let mut tmp = 0.0;
                                    for jj in j..ej {
                                        tmp += *pa.add(ii + jj * d1) * *pb.add(jj + kk_d2);
                                    }
                                    *py.add(ii + kk_d1) += tmp;
                                }
                            }
                        }
                    }
                }
                py = py.add(size);
                pa = pa.add(skip_a);
                pb = pb.add(skip_b);
            }
        }
    }
}

define_empty_impl!(MatmulBwAImpl);
impl FunctionBwImpl for MatmulBwAImpl {
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
        *ga += &gy.device().matmul_fw(gy, &b.device().transpose_fw(b));
    }
}

define_empty_impl!(MatmulBwBImpl);
impl FunctionBwImpl for MatmulBwBImpl {
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
        let gy = gys[0];
        let gb = gx;
        *gb += &a.device().matmul_fw(&a.device().transpose_fw(a), gy);
    }
}

#[cfg(test)]
mod tests {
    use crate::devices as D;
    use crate::functions::BasicFunctions;

    #[test]
    fn check_matmul_fw_aa() {
        let x_data = vec![1., 2., 3., 4., 1., 0., 0., 1., 0., 2., 3., 0.];
        let y_data = vec![7., 10., 15., 22., 1., 0., 0., 1., 6., 0., 0., 6.];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![2, 2; 3], &x_data);
        let mut y = dev.new_tensor(shape![2, 2; 3]);
        y.alloc();
        dev.call_fw_impl("matmul_fw_impl", &[&x, &x], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_matmul_fw_ab() {
        let a_data = vec![
            1., 1000., 1., 10., 100., 10., 100., 10., 100., 1000., 1., 1000.,
        ];
        let b_data = vec![
            0., 2., 4., 6., 1., 3., 5., 7., 8., 6., 4., 2., 9., 7., 5., 3., 2., 3., 5., 7., 9., 4.,
            1., 0.,
        ];
        let y_data = vec![
            6420., 246., 6420., 7531., 1357., 7531., 2468., 8642., 2468., 3579., 9753., 3579.,
            7532., 2357., 7532., 149., 9410., 149.,
        ];
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_slice(shape![3, 4], &a_data);
        let b = dev.new_tensor_by_slice(shape![4, 6], &b_data);
        let mut y = dev.new_tensor(shape![3, 6]);
        y.alloc();
        dev.call_fw_impl("matmul_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_matmul_fw_batch_broadcast_1n() {
        let a_data = vec![10., 1000., 1., 100.];
        let b_data = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let y_data = vec![12., 1200., 34., 3400., 56., 5600., 78., 7800.];
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_slice(shape![2, 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("matmul_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_matmul_fw_batch_broadcast_n1() {
        let a_data = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let b_data = vec![10., 1., 1000., 100.];
        let y_data = vec![13., 24., 1300., 2400., 57., 68., 5700., 6800.];
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_slice(shape![2, 2; 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2], &b_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("matmul_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
        assert_vector_ulps_eq!(y_data, y.to_vec());
    }

    #[test]
    fn check_matmul_fw_large() {
        let n = 123;
        let mut a_data = vec![0.; n * n];
        let mut b_data = vec![0.; n * n];
        let mut y1_data = vec![0.; n * n];
        let mut y2_data = vec![0.; n * n];
        let mut k = 0;
        for i in 0..n {
            k += i * i;
        }
        for i in 0..n {
            for j in 0..n {
                a_data[i + j * n] = i as f32 / 16.;
                b_data[i + j * n] = j as f32 / 16.;
                y1_data[i + j * n] = (n * i * j) as f32 / 256.;
                y2_data[i + j * n] = k as f32 / 256.;
            }
        }
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_slice(shape![n as u32, n as u32], &a_data);
        let b = dev.new_tensor_by_slice(shape![n as u32, n as u32], &b_data);
        let mut y1 = dev.new_tensor(shape![n as u32, n as u32]);
        y1.alloc();
        dev.call_fw_impl("matmul_fw_impl", &[&a, &b], &[], &[], &mut [&mut y1]);
        assert_vector_ulps_eq!(y1_data, y1.to_vec());
        let mut y2 = dev.new_tensor(shape![n as u32, n as u32]);
        y2.alloc();
        dev.call_fw_impl("matmul_fw_impl", &[&b, &a], &[], &[], &mut [&mut y2]);
        assert_vector_ulps_eq!(y2_data, y2.to_vec());
    }

    #[test]
    fn check_matmul_bw_11() {
        let a_data = vec![1., 2., 3., 4.];
        let b_data = vec![1., 0., 0., 2.];
        let gy_data = vec![1., -1., 2., -2.];
        let ga_data = vec![2., 0., 5., -3.];
        let gb_data = vec![0., 0., -1., -1.];
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_slice(shape![2, 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2], &b_data);
        let gy = dev.new_tensor_by_slice(shape![2, 2], &gy_data);
        let mut y = dev.new_tensor(shape![2, 2]);
        y.alloc();
        dev.call_fw_impl("matmul_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
        let mut ga = dev.new_tensor_by_constant(shape![2, 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2, 2], 1.);
        dev.call_bw_impl(
            "matmul_bw_a_impl",
            &[&a, &b],
            &[&y],
            &[&gy],
            &[],
            &[],
            &mut ga,
        );
        dev.call_bw_impl(
            "matmul_bw_b_impl",
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
    fn check_matmul_bw_1n() {
        let a_data = vec![1., 2., 3., 4.];
        let b_data = vec![1., 0., 0., 2., 0., 1., 2., 0.];
        let gy_data = vec![1., -1., 2., -2., 2., -2., 1., -1.];
        let ga_data = vec![4., -2., 7., -5.];
        let gb_data = vec![0., 0., -1., -1., -1., -1., 0., 0.];
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_slice(shape![2, 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
        let gy = dev.new_tensor_by_slice(shape![2, 2; 2], &gy_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("matmul_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
        let mut ga = dev.new_tensor_by_constant(shape![2, 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2, 2; 2], 1.);
        dev.call_bw_impl(
            "matmul_bw_a_impl",
            &[&a, &b],
            &[&y],
            &[&gy],
            &[],
            &[],
            &mut ga,
        );
        dev.call_bw_impl(
            "matmul_bw_b_impl",
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
    fn check_matmul_bw_n1() {
        let a_data = vec![1., 2., 3., 4., -1., -2., -3., -4.];
        let b_data = vec![1., 0., 0., 2.];
        let gy_data = vec![1., -1., 2., -2., 2., -2., 1., -1.];
        let ga_data = vec![2., 0., 5., -3., 3., -1., 3., -1.];
        let gb_data = vec![2., 2., 0., 0.];
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_slice(shape![2, 2; 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2], &b_data);
        let gy = dev.new_tensor_by_slice(shape![2, 2; 2], &gy_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("matmul_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
        let mut ga = dev.new_tensor_by_constant(shape![2, 2; 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2, 2], 1.);
        dev.call_bw_impl(
            "matmul_bw_a_impl",
            &[&a, &b],
            &[&y],
            &[&gy],
            &[],
            &[],
            &mut ga,
        );
        dev.call_bw_impl(
            "matmul_bw_b_impl",
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
    fn check_matmul_bw_nn() {
        let a_data = vec![1., 2., 3., 4., -1., -2., -3., -4.];
        let b_data = vec![1., 0., 0., 2., 0., 1., 2., 0.];
        let gy_data = vec![1., -1., 2., -2., 2., -2., 1., -1.];
        let ga_data = vec![2., 0., 5., -3., 3., -1., 3., -1.];
        let gb_data = vec![0., 0., -1., -1., 3., 3., 2., 2.];
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_slice(shape![2, 2; 2], &a_data);
        let b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
        let gy = dev.new_tensor_by_slice(shape![2, 2; 2], &gy_data);
        let mut y = dev.new_tensor(shape![2, 2; 2]);
        y.alloc();
        dev.call_fw_impl("matmul_fw_impl", &[&a, &b], &[], &[], &mut [&mut y]);
        let mut ga = dev.new_tensor_by_constant(shape![2, 2; 2], 1.);
        let mut gb = dev.new_tensor_by_constant(shape![2, 2; 2], 1.);
        dev.call_bw_impl(
            "matmul_bw_a_impl",
            &[&a, &b],
            &[&y],
            &[&gy],
            &[],
            &[],
            &mut ga,
        );
        dev.call_bw_impl(
            "matmul_bw_b_impl",
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
