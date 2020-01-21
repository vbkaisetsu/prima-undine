use crate::device_impl::{FunctionBwImpl, FunctionFwImpl};
use crate::Tensor;

define_empty_impl!(MinFwImpl);
impl FunctionFwImpl for MinFwImpl {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let dim = u32data[0];
        let y = &mut ys[0];
        let n = x.shape[dim] as usize;
        let repeat = y.shape.size() as usize;
        let skip1 = y.shape.lower_volume(dim) as usize;
        let skip2 = skip1 * n;
        unsafe {
            let px = const_ptr!(x);
            let py = mut_ptr!(y);
            for i in 0..repeat {
                let mut offset = i % skip1 + (i / skip1) * skip2;
                let mut tmp = *px.add(offset);
                for _ in 0..n {
                    if *px.add(offset) < tmp {
                        tmp = *px.add(offset);
                    }
                    offset += skip1;
                }
                *py.add(i) = tmp;
            }
        }
    }
}

define_empty_impl!(MinBwImpl);
impl FunctionBwImpl for MinBwImpl {
    fn call(
        &self,
        xs: &[&Tensor],
        ys: &[&Tensor],
        gys: &[&Tensor],
        u32data: &[u32],
        _f32data: &[f32],
        gx: &mut Tensor,
    ) {
        let x = xs[0];
        let y = ys[0];
        let gy = gys[0];
        let dim = u32data[0];
        let n = x.shape[dim] as usize;
        let repeat = y.shape.size() as usize;
        let skip1 = y.shape.lower_volume(dim) as usize;
        let skip2 = skip1 * n;
        unsafe {
            let py = const_ptr!(y);
            let px = const_ptr!(x);
            let pgy = const_ptr!(gy);
            let pgx = mut_ptr!(gx);
            for i in 0..repeat {
                let minval = *py.add(i);
                let mut offset = i % skip1 + (i / skip1) * skip2;
                for _ in 0..n {
                    if *px.add(offset) == minval {
                        *pgx.add(offset) += *pgy.add(i);
                        break;
                    }
                    offset += skip1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cmp;

    use crate::functions::BasicFunctions;
    use crate::{devices as D, Shape};
    use rand::seq::SliceRandom;

    #[test]
    fn check_min_fw_dims() {
        struct TestCase(u32, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(0, shape![1, 3; 2], vec![3., 0., 6., -2., -8., -5.]),
            TestCase(1, shape![3, 1; 2], vec![0., 1., 2., -6., -7., -8.]),
            TestCase(
                2,
                shape![3, 3; 2],
                vec![
                    3., 4., 5., 0., 1., 2., 6., 7., 8., 0., -1., -2., -6., -7., -8., -3., -4., -5.,
                ],
            ),
        ];
        let x_data = vec![
            3., 4., 5., 0., 1., 2., 6., 7., 8., 0., -1., -2., -6., -7., -8., -3., -4., -5.,
        ];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![3, 3; 2], &x_data);
        for tc in &test_cases {
            let mut y = dev.new_tensor(tc.1);
            y.alloc();
            dev.call_fw_impl("min_fw_impl", &[&x], &[tc.0], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(tc.2, y.to_vec());
        }
    }

    #[test]
    fn check_min_fw_large() {
        let ns = vec![
            1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024, 1025, 2047, 2048, 2049, 65535, 65536,
            65537,
        ];
        let mut rng = rand::thread_rng();
        let dev = D::Naive::new();
        for &n in &ns {
            let mut x_data = (0..n).map(|x| x as f32).collect::<Vec<f32>>();
            x_data.shuffle(&mut rng);
            let x = dev.new_tensor_by_slice(shape![n], &x_data);
            let mut y = dev.new_tensor(shape![]);
            y.alloc();
            dev.call_fw_impl("min_fw_impl", &[&x], &[0], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(vec![0.], y.to_vec());
        }
    }

    #[test]
    fn check_min_bw_dims() {
        struct TestCase(u32, Vec<f32>, Vec<f32>, Vec<f32>);
        let test_cases = vec![
            TestCase(
                0,
                vec![3., 0., 6., -2., -8., -5.],
                vec![1., 2., 6., 5., 3., 4.],
                vec![
                    2., 1., 1., 3., 1., 1., 7., 1., 1., 1., 1., 6., 1., 1., 4., 1., 1., 5.,
                ],
            ),
            TestCase(
                1,
                vec![0., 1., 2., -6., -7., -8.],
                vec![-1., 1., -2., 2., -3., 3.],
                vec![
                    1., 1., 1., 0., 2., -1., 1., 1., 1., 1., 1., 1., 3., -2., 4., 1., 1., 1.,
                ],
            ),
            TestCase(
                2,
                vec![
                    3., 4., 5., 0., 1., 2., 6., 7., 8., 0., -1., -2., -6., -7., -8., -3., -4., -5.,
                ],
                vec![
                    0., 1., 0., -1., 0., 1., 0., -1., 2., 1., 0., -1., 0., 1., 2., 3., 4., 6.,
                ],
                vec![
                    1., 2., 1., 0., 1., 2., 1., 0., 3., 2., 1., 0., 1., 2., 3., 4., 5., 7.,
                ],
            ),
        ];
        let x_data = vec![
            3., 4., 5., 0., 1., 2., 6., 7., 8., 0., -1., -2., -6., -7., -8., -3., -4., -5.,
        ];
        let dev = D::Naive::new();
        let r = shape![3, 3; 2];
        let x = dev.new_tensor_by_slice(r, &x_data);
        for tc in &test_cases {
            let s = r.resize_dim(tc.0, 1);
            let y = dev.new_tensor_by_slice(s, &tc.1);
            let gy = dev.new_tensor_by_slice(s, &tc.2);
            let mut gx = dev.new_tensor_by_constant(r, 1.);
            dev.call_bw_impl("min_bw_impl", &[&x], &[&y], &[&gy], &[tc.0], &[], &mut gx);
            assert_vector_ulps_eq!(tc.3, gx.to_vec());
        }
    }

    #[test]
    fn check_min_bw_large() {
        let ns = vec![
            1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024, 1025, 2047, 2048,
            2049,
            //65534, 65535, 65536,
        ];
        let mut rng = rand::thread_rng();
        let dev = D::Naive::new();
        for &n in &ns {
            let mut x_data = (0..n).map(|x| x as f32).collect::<Vec<f32>>();
            let mut gx_data = vec![1.; n as usize];
            x_data.shuffle(&mut rng);
            let pos = x_data.iter().position(|&x| x == 0.).unwrap();
            gx_data[pos] = 2.;
            let x = dev.new_tensor_by_slice(shape![n], &x_data);
            let y = dev.new_tensor_by_constant(shape![], 0.);
            let gy = dev.new_tensor_by_constant(shape![], 1.);
            let mut gx = dev.new_tensor_by_constant(shape![n], 1.);
            dev.call_bw_impl("min_bw_impl", &[&x], &[&y], &[&gy], &[0], &[], &mut gx);
            assert_vector_ulps_eq!(gx_data, gx.to_vec());
        }
    }

    #[test]
    fn check_min_multiple_large_bw() {
        let ns = vec![
            1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024, 1025, 2047, 2048,
            2049,
            //65534, 65535, 65536,
        ];
        let mut rng = rand::thread_rng();
        let dev = D::Naive::new();
        for &n in &ns {
            let mut x_data = (0..n).map(|x| x as f32).collect::<Vec<f32>>();
            for i in 0..cmp::min(10, n as usize) {
                x_data[i] = 0.;
            }
            let mut gx_data = vec![1.; n as usize];
            x_data.shuffle(&mut rng);
            let pos = x_data.iter().position(|&x| x == 0.).unwrap();
            gx_data[pos] = 2.;
            let x = dev.new_tensor_by_slice(shape![n], &x_data);
            let y = dev.new_tensor_by_constant(shape![], 0.);
            let gy = dev.new_tensor_by_constant(shape![], 1.);
            let mut gx = dev.new_tensor_by_constant(shape![n], 1.);
            dev.call_bw_impl("min_bw_impl", &[&x], &[&y], &[&gy], &[0], &[], &mut gx);
            assert_vector_ulps_eq!(gx_data, gx.to_vec());
        }
    }
}
