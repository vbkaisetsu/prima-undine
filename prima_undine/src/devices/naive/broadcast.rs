use crate::device_impl::FunctionFwImpl;
use crate::Tensor;

define_empty_impl!(BroadcastFwImpl);
impl FunctionFwImpl for BroadcastFwImpl {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let dim = u32data[0];
        let size = u32data[1];
        let y = &mut ys[0];
        let repeat = x.shape.size() as usize;
        let skip1 = y.shape.lower_volume(dim) as usize;
        let skip2 = skip1 * size as usize;
        unsafe {
            let src = const_ptr!(x);
            let dest = mut_ptr!(y);
            for i in 0..repeat {
                let mut offset = i % skip1 + (i / skip1) * skip2;
                let tmp = *src.add(i);
                for _ in 0..size {
                    *dest.add(offset) = tmp;
                    offset += skip1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::functions::BasicFunctions;
    use crate::{devices as D, Shape};

    #[test]
    fn check_broadcast_fw() {
        struct TestCase(u32, u32, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(0, 1, shape![], vec![1.]),
            TestCase(0, 20, shape![20], vec![1.; 20]),
            TestCase(1, 50, shape![1, 50], vec![1.; 50]),
            TestCase(2, 100, shape![1, 1, 100], vec![1.; 100]),
        ];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_constant(shape![], 1.);
        for tc in &test_cases {
            let mut y = dev.new_tensor(tc.2);
            y.alloc();
            dev.call_fw_impl(
                "broadcast_fw_impl",
                &[&x],
                &[tc.0, tc.1],
                &[],
                &mut [&mut y],
            );
            assert_vector_ulps_eq!(tc.3, y.to_vec());
        }
    }

    #[test]
    fn check_broadcast_fw_2() {
        struct TestCase(u32, u32, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(1, 1, shape![2; 3], vec![1., 2., 3., 4., 5., 6.]),
            TestCase(2, 1, shape![2; 3], vec![1., 2., 3., 4., 5., 6.]),
            TestCase(
                1,
                2,
                shape![2, 2; 3],
                vec![1., 2., 1., 2., 3., 4., 3., 4., 5., 6., 5., 6.],
            ),
            TestCase(
                2,
                2,
                shape![2, 1, 2; 3],
                vec![1., 2., 1., 2., 3., 4., 3., 4., 5., 6., 5., 6.],
            ),
        ];
        let x_data = vec![1., 2., 3., 4., 5., 6.];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![2; 3], &x_data);
        for tc in &test_cases {
            let mut y = dev.new_tensor(tc.2);
            y.alloc();
            dev.call_fw_impl(
                "broadcast_fw_impl",
                &[&x],
                &[tc.0, tc.1],
                &[],
                &mut [&mut y],
            );
            assert_vector_ulps_eq!(tc.3, y.to_vec());
        }
    }

    #[test]
    fn check_broadcast_fw_3() {
        struct TestCase(u32, u32, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(
                0,
                1,
                shape![1, 2, 1, 2; 2],
                vec![1., 2., 3., 4., 5., 6., 7., 8.],
            ),
            TestCase(
                2,
                1,
                shape![1, 2, 1, 2; 2],
                vec![1., 2., 3., 4., 5., 6., 7., 8.],
            ),
            TestCase(
                4,
                1,
                shape![1, 2, 1, 2; 2],
                vec![1., 2., 3., 4., 5., 6., 7., 8.],
            ),
            TestCase(
                0,
                2,
                shape![2, 2, 1, 2; 2],
                vec![
                    1., 1., 2., 2., 3., 3., 4., 4., 5., 5., 6., 6., 7., 7., 8., 8.,
                ],
            ),
            TestCase(
                2,
                2,
                shape![1, 2, 2, 2; 2],
                vec![
                    1., 2., 1., 2., 3., 4., 3., 4., 5., 6., 5., 6., 7., 8., 7., 8.,
                ],
            ),
            TestCase(
                4,
                2,
                shape![1, 2, 1, 2, 2; 2],
                vec![
                    1., 2., 3., 4., 1., 2., 3., 4., 5., 6., 7., 8., 5., 6., 7., 8.,
                ],
            ),
        ];
        let x_data = vec![1., 2., 3., 4., 5., 6., 7., 8.];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![1, 2, 1, 2; 2], &x_data);
        for tc in &test_cases {
            let mut y = dev.new_tensor(tc.2);
            y.alloc();
            dev.call_fw_impl(
                "broadcast_fw_impl",
                &[&x],
                &[tc.0, tc.1],
                &[],
                &mut [&mut y],
            );
            assert_vector_ulps_eq!(tc.3, y.to_vec());
        }
    }
}
