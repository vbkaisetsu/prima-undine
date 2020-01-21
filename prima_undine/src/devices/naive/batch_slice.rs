use crate::device_impl::{FunctionBwImpl, FunctionFwImpl};
use crate::Tensor;

define_empty_impl!(BatchSliceFwImpl);
impl FunctionFwImpl for BatchSliceFwImpl {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let offset = u32data[0] as usize;
        let y = &mut ys[0];
        let volume = y.shape.volume() as usize;
        let repeat = y.shape.batch() as usize;
        unsafe {
            let src = const_ptr!(x).add(volume * offset);
            let py = mut_ptr!(y);
            for i in 0..volume * repeat {
                *py.add(i) = *src.add(i);
            }
        }
    }
}

define_empty_impl!(BatchSliceBwImpl);
impl FunctionBwImpl for BatchSliceBwImpl {
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
        let offset = u32data[0] as usize;
        let volume = gy.shape.volume() as usize;
        let repeat = gy.shape.batch() as usize;
        unsafe {
            let pgy = const_ptr!(gy);
            let dest = mut_ptr!(gx).add(volume * offset);
            for i in 0..volume * repeat {
                *dest.add(i) += *pgy.add(i);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::functions::BasicFunctions;
    use crate::{devices as D, Shape};

    #[test]
    fn check_batch_slice_fw() {
        let x_data = (0..18).map(|x| x as f32).collect::<Vec<f32>>();
        struct TestCase(u32, u32, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(0, 1, shape![3, 2], vec![0., 1., 2., 3., 4., 5.]),
            TestCase(1, 2, shape![3, 2], vec![6., 7., 8., 9., 10., 11.]),
            TestCase(2, 3, shape![3, 2], vec![12., 13., 14., 15., 16., 17.]),
            TestCase(
                0,
                2,
                shape![3, 2; 2],
                vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
            ),
            TestCase(
                1,
                3,
                shape![3, 2; 2],
                vec![6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.],
            ),
            TestCase(0, 3, shape![3, 2; 3], x_data.clone()),
        ];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![3, 2; 3], &x_data);
        for tc in &test_cases {
            let mut y = dev.new_tensor(tc.2);
            y.alloc();
            dev.call_fw_impl(
                "batch_slice_fw_impl",
                &[&x],
                &[tc.0, tc.1],
                &[],
                &mut [&mut y],
            );
            assert_vector_ulps_eq!(tc.3, &y.to_vec());
        }
    }

    #[test]
    fn check_batch_slice_bw_nn_1() {
        let a_data = vec![0., 1., 2., 3., 0., 1., 2., 3., 0., 1., 2., 3.];
        let gy_data = vec![1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.];
        let gx_data = vec![1., 2., 3., 4., 2., 3., 4., 5., 3., 4., 5., 6.];
        let dev = D::Naive::new();
        let mut gx = dev.new_tensor_by_slice(shape![2, 2; 3], &a_data);
        let gy = dev.new_tensor_by_slice(shape![2, 2; 3], &gy_data);
        dev.call_bw_impl("batch_slice_bw_impl", &[], &[], &[&gy], &[0], &[], &mut gx);
        assert_vector_ulps_eq!(gx_data, &gx.to_vec());
    }

    #[test]
    fn check_batch_slice_bw_nn_2() {
        let a_data = vec![0., 1., 2., 3., 0., 1., 2., 3., 0., 1., 2., 3.];
        let gy_data = vec![1., 1., 2., 2., 3., 3., 4., 4.];
        struct TestCase(Shape, u32, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 2; 2],
                0,
                vec![1., 2., 4., 5., 3., 4., 6., 7., 0., 1., 2., 3.],
            ),
            TestCase(
                shape![2, 2; 2],
                1,
                vec![0., 1., 2., 3., 1., 2., 4., 5., 3., 4., 6., 7.],
            ),
        ];
        let dev = D::Naive::new();
        for tc in &test_cases {
            let gy = dev.new_tensor_by_slice(tc.0, &gy_data);
            let mut gx = dev.new_tensor_by_slice(shape![2, 2; 3], &a_data);
            dev.call_bw_impl(
                "batch_slice_bw_impl",
                &[],
                &[],
                &[&gy],
                &[tc.1],
                &[],
                &mut gx,
            );
            assert_vector_ulps_eq!(tc.2, &gx.to_vec());
        }
    }

    #[test]
    fn check_batch_slice_bw_n1_1() {
        let a_data = vec![1., 2., 3., 4., 2., 3., 4., 5., 3., 4., 5., 6.];
        let gy_data = vec![-1., -2., -3., -4.];
        let gx_data = vec![0., 0., 0., 0., 2., 3., 4., 5., 3., 4., 5., 6.];
        let dev = D::Naive::new();
        let gy = dev.new_tensor_by_slice(shape![2, 2], &gy_data);
        let mut gx = dev.new_tensor_by_slice(shape![2, 2; 3], &a_data);
        dev.call_bw_impl("batch_slice_bw_impl", &[], &[], &[&gy], &[0], &[], &mut gx);
        assert_vector_ulps_eq!(gx_data, &gx.to_vec());
    }

    #[test]
    fn check_batch_slice_bw_n1_2() {
        let a_data = vec![1., 2., 3., 4., 2., 3., 4., 5., 3., 4., 5., 6.];
        let gy_data = vec![-1., -2., -3., -4.];
        struct TestCase(Shape, u32, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 2],
                0,
                vec![0., 0., 0., 0., 2., 3., 4., 5., 3., 4., 5., 6.],
            ),
            TestCase(
                shape![2, 2],
                1,
                vec![1., 2., 3., 4., 1., 1., 1., 1., 3., 4., 5., 6.],
            ),
            TestCase(
                shape![2, 2],
                2,
                vec![1., 2., 3., 4., 2., 3., 4., 5., 2., 2., 2., 2.],
            ),
        ];
        let dev = D::Naive::new();
        for tc in &test_cases {
            let gy = dev.new_tensor_by_slice(tc.0, &gy_data);
            let mut gx = dev.new_tensor_by_slice(shape![2, 2; 3], &a_data);
            dev.call_bw_impl(
                "batch_slice_bw_impl",
                &[],
                &[],
                &[&gy],
                &[tc.1],
                &[],
                &mut gx,
            );
            assert_vector_ulps_eq!(tc.2, &gx.to_vec());
        }
    }
}
