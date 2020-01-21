use std::cmp;

use crate::device_impl::{FunctionBwImpl, FunctionFwImpl};
use crate::Tensor;

define_empty_impl!(SliceFwImpl);
impl FunctionFwImpl for SliceFwImpl {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let dim = u32data[0];
        let offset = u32data[1] as usize;
        let y = &mut ys[0];
        let base = y.shape.lower_volume(dim) as usize;
        let span = base * y.shape[dim] as usize;
        let skip = base * x.shape[dim] as usize;
        let repeat = y.shape.size() as usize / span;
        unsafe {
            let src = const_ptr!(x).add(base * offset as usize);
            let mut dest = mut_ptr!(y);
            for i in 0..repeat {
                let sp = src.add(skip * i);
                for j in 0..span {
                    *dest = *sp.add(j);
                    dest = dest.add(1);
                }
            }
        }
    }
}

define_empty_impl!(SliceBwImpl);
impl FunctionBwImpl for SliceBwImpl {
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
        let dim = u32data[0];
        let offset = u32data[1];
        let sy = gy.shape;
        let sx = gx.shape;
        let base = sx.lower_volume(dim) as usize;
        let span = base * sy[dim] as usize;
        let skip = base * sx[dim] as usize;
        let repeat = sx.volume() as usize / skip;
        let bs = cmp::max(sx.batch(), sy.batch()) as usize;
        let b_skip_d = if sx.has_batch() {
            sx.volume() as usize
        } else {
            0
        };
        let b_skip_s = if sy.has_batch() {
            sy.volume() as usize
        } else {
            0
        };
        unsafe {
            let src = const_ptr!(gy);
            let dest = mut_ptr!(gx).add(base * offset as usize);
            for batch in 0..bs {
                let mut sp = src.add(b_skip_s * batch);
                let dp = dest.add(b_skip_d * batch);
                for i in 0..repeat {
                    let ddp = dp.add(skip * i);
                    for j in 0..span {
                        *ddp.add(j) += *sp;
                        sp = sp.add(1);
                    }
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
    fn check_slice_fw() {
        let x_data = (0..3 * 3 * 2 * 4).map(|x| x as f32).collect::<Vec<f32>>();
        struct TestCase(u32, u32, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(
                0,
                0,
                shape![1, 3, 2; 4],
                vec![
                    0., 3., 6., 9., 12., 15., 18., 21., 24., 27., 30., 33., 36., 39., 42., 45.,
                    48., 51., 54., 57., 60., 63., 66., 69.,
                ],
            ),
            TestCase(
                1,
                0,
                shape![3, 1, 2; 4],
                vec![
                    0., 1., 2., 9., 10., 11., 18., 19., 20., 27., 28., 29., 36., 37., 38., 45.,
                    46., 47., 54., 55., 56., 63., 64., 65.,
                ],
            ),
            TestCase(
                2,
                0,
                shape![3, 3, 1; 4],
                vec![
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 18., 19., 20., 21., 22., 23., 24., 25.,
                    26., 36., 37., 38., 39., 40., 41., 42., 43., 44., 54., 55., 56., 57., 58., 59.,
                    60., 61., 62.,
                ],
            ),
            // middle
            TestCase(
                0,
                1,
                shape![1, 3, 2; 4],
                vec![
                    1., 4., 7., 10., 13., 16., 19., 22., 25., 28., 31., 34., 37., 40., 43., 46.,
                    49., 52., 55., 58., 61., 64., 67., 70.,
                ],
            ),
            TestCase(
                1,
                1,
                shape![3, 1, 2; 4],
                vec![
                    3., 4., 5., 12., 13., 14., 21., 22., 23., 30., 31., 32., 39., 40., 41., 48.,
                    49., 50., 57., 58., 59., 66., 67., 68.,
                ],
            ),
            TestCase(
                2,
                1,
                shape![3, 3, 1; 4],
                vec![
                    9., 10., 11., 12., 13., 14., 15., 16., 17., 27., 28., 29., 30., 31., 32., 33.,
                    34., 35., 45., 46., 47., 48., 49., 50., 51., 52., 53., 63., 64., 65., 66., 67.,
                    68., 69., 70., 71.,
                ],
            ),
            // rightmost
            TestCase(
                0,
                2,
                shape![1, 3, 2; 4],
                vec![
                    2., 5., 8., 11., 14., 17., 20., 23., 26., 29., 32., 35., 38., 41., 44., 47.,
                    50., 53., 56., 59., 62., 65., 68., 71.,
                ],
            ),
            TestCase(
                1,
                2,
                shape![3, 1, 2; 4],
                vec![
                    6., 7., 8., 15., 16., 17., 24., 25., 26., 33., 34., 35., 42., 43., 44., 51.,
                    52., 53., 60., 61., 62., 69., 70., 71.,
                ],
            ),
            // higher dim
            TestCase(3, 0, shape![3, 3, 2; 4], x_data.clone()),
        ];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![3, 3, 2; 4], &x_data);
        for tc in &test_cases {
            let mut y = dev.new_tensor(tc.2);
            y.alloc();
            dev.call_fw_impl("slice_fw_impl", &[&x], &[tc.0, tc.1], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(tc.3, y.to_vec());
        }
    }

    #[test]
    fn check_slice_bw_nn_1() {
        struct TestCase(Shape, u32, u32, Vec<f32>);
        let gy_data = vec![1., 1., 2., 2., 3., 3.];
        let test_cases = vec![
            TestCase(
                shape![1, 2; 3],
                0,
                0,
                vec![2., 1., 2., 1., 3., 1., 3., 1., 4., 1., 4., 1.],
            ),
            TestCase(
                shape![1, 2; 3],
                0,
                1,
                vec![1., 2., 1., 2., 1., 3., 1., 3., 1., 4., 1., 4.],
            ),
            TestCase(
                shape![2; 3],
                1,
                0,
                vec![2., 2., 1., 1., 3., 3., 1., 1., 4., 4., 1., 1.],
            ),
            TestCase(
                shape![2; 3],
                1,
                1,
                vec![1., 1., 2., 2., 1., 1., 3., 3., 1., 1., 4., 4.],
            ),
        ];
        let dev = D::Naive::new();
        for tc in &test_cases {
            let gy = dev.new_tensor_by_slice(tc.0, &gy_data);
            let mut gx = dev.new_tensor_by_constant(shape![2, 2; 3], 1.);
            dev.call_bw_impl(
                "slice_bw_impl",
                &[],
                &[],
                &[&gy],
                &[tc.1, tc.2],
                &[],
                &mut gx,
            );
            assert_vector_ulps_eq!(tc.3, gx.to_vec());
        }
    }

    #[test]
    fn check_slice_bw_nn_2() {
        struct TestCase(Shape, u32, u32, Vec<f32>);
        let gy_data = vec![1., 1., 2., 2., 3., 3.];
        let test_cases = vec![
            TestCase(
                shape![1, 2; 3],
                0,
                0,
                vec![2., 1., 2., 1., 3., 1., 3., 1., 4., 1., 4., 1.],
            ),
            TestCase(
                shape![1, 2; 3],
                0,
                1,
                vec![1., 2., 1., 2., 1., 3., 1., 3., 1., 4., 1., 4.],
            ),
            TestCase(
                shape![2; 3],
                1,
                0,
                vec![2., 2., 1., 1., 3., 3., 1., 1., 4., 4., 1., 1.],
            ),
            TestCase(
                shape![2; 3],
                1,
                1,
                vec![1., 1., 2., 2., 1., 1., 3., 3., 1., 1., 4., 4.],
            ),
        ];
        let dev = D::Naive::new();
        for tc in &test_cases {
            let gy = dev.new_tensor_by_slice(tc.0, &gy_data);
            let mut gx = dev.new_tensor_by_constant(shape![2, 2; 3], 1.);
            dev.call_bw_impl(
                "slice_bw_impl",
                &[],
                &[],
                &[&gy],
                &[tc.1, tc.2],
                &[],
                &mut gx,
            );
            assert_vector_ulps_eq!(tc.3, gx.to_vec());
        }
    }

    #[test]
    fn check_slice_bw_1n_1() {
        let gy_data = vec![1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.];
        let gx_data = vec![7., 7., 7., 7.];
        let dev = D::Naive::new();
        for &i in &[0, 1, 2, 5, 10] {
            let gy = dev.new_tensor_by_slice(shape![2, 2; 3], &gy_data);
            let mut gx = dev.new_tensor_by_constant(shape![2, 2], 1.);
            dev.call_bw_impl("slice_bw_impl", &[], &[], &[&gy], &[i, 0], &[], &mut gx);
            assert_vector_ulps_eq!(gx_data, gx.to_vec());
        }
    }

    #[test]
    fn check_slice_bw_1n_2() {
        struct TestCase(Shape, u32, u32, Vec<f32>);
        let gy_data = vec![1., 1., 2., 2., 3., 3.];
        let test_cases = vec![
            TestCase(shape![1, 2; 3], 0, 0, vec![7., 1., 7., 1.]),
            TestCase(shape![1, 2; 3], 0, 1, vec![1., 7., 1., 7.]),
            TestCase(shape![2; 3], 1, 0, vec![7., 7., 1., 1.]),
            TestCase(shape![2; 3], 1, 1, vec![1., 1., 7., 7.]),
        ];
        let dev = D::Naive::new();
        for tc in &test_cases {
            let gy = dev.new_tensor_by_slice(tc.0, &gy_data);
            let mut gx = dev.new_tensor_by_constant(shape![2, 2], 1.);
            dev.call_bw_impl(
                "slice_bw_impl",
                &[],
                &[],
                &[&gy],
                &[tc.1, tc.2],
                &[],
                &mut gx,
            );
            assert_vector_ulps_eq!(tc.3, gx.to_vec());
        }
    }

    #[test]
    fn check_slice_bw_n1_1() {
        let gy_data = vec![-1., -2., -3., -4.];
        let gx_data = vec![0., -1., -2., -3., 0., -1., -2., -3., 0., -1., -2., -3.];
        let dev = D::Naive::new();
        for &i in &[0, 1, 2, 5, 10] {
            let gy = dev.new_tensor_by_slice(shape![2, 2], &gy_data);
            let mut gx = dev.new_tensor_by_constant(shape![2, 2; 3], 1.);
            dev.call_bw_impl("slice_bw_impl", &[], &[], &[&gy], &[i, 0], &[], &mut gx);
            assert_vector_ulps_eq!(gx_data, gx.to_vec());
        }
    }

    #[test]
    fn check_slice_bw_n1_2() {
        struct TestCase(Shape, u32, u32, Vec<f32>);
        let gy_data = vec![-1., -2.];
        let test_cases = vec![
            TestCase(
                shape![1, 2],
                0,
                0,
                vec![0., 1., -1., 1., 0., 1., -1., 1., 0., 1., -1., 1.],
            ),
            TestCase(
                shape![1, 2],
                0,
                1,
                vec![1., 0., 1., -1., 1., 0., 1., -1., 1., 0., 1., -1.],
            ),
            TestCase(
                shape![2],
                1,
                0,
                vec![0., -1., 1., 1., 0., -1., 1., 1., 0., -1., 1., 1.],
            ),
            TestCase(
                shape![2],
                1,
                1,
                vec![1., 1., 0., -1., 1., 1., 0., -1., 1., 1., 0., -1.],
            ),
        ];
        let dev = D::Naive::new();
        for tc in &test_cases {
            let gy = dev.new_tensor_by_slice(tc.0, &gy_data);
            let mut gx = dev.new_tensor_by_constant(shape![2, 2; 3], 1.);
            dev.call_bw_impl(
                "slice_bw_impl",
                &[],
                &[],
                &[&gy],
                &[tc.1, tc.2],
                &[],
                &mut gx,
            );
            assert_vector_ulps_eq!(tc.3, gx.to_vec());
        }
    }
}
