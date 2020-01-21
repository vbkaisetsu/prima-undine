use crate::device_impl::{FunctionBwImpl, FunctionFwImpl};
use crate::Tensor;

define_empty_impl!(PickFwImpl);
impl FunctionFwImpl for PickFwImpl {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let dim = u32data[0];
        let ids = &u32data[1..];
        let y = &mut ys[0];
        let bs = y.shape.batch() as usize;
        let skip_x = if x.shape.has_batch() {
            x.shape.volume() as usize
        } else {
            0
        };
        let skip_i = if ids.len() > 1 { 1 } else { 0 };
        let base = y.shape.lower_volume(dim) as usize;
        let skip = base * x.shape[dim] as usize;
        let repeat = y.shape.volume() as usize / base;
        unsafe {
            let px = const_ptr!(x);
            let mut dest = mut_ptr!(y);
            for batch in 0..bs {
                let src = px.add(batch * skip_x + base * ids[batch * skip_i] as usize);
                for i in 0..repeat {
                    let sp = src.add(skip * i);
                    for j in 0..base {
                        *dest = *sp.add(j);
                        dest = dest.add(1);
                    }
                }
            }
        }
    }
}

define_empty_impl!(PickBwImpl);
impl FunctionBwImpl for PickBwImpl {
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
        let ids = &u32data[1..];
        let bs = gy.shape.batch() as usize;
        let skip_x = if gx.shape.has_batch() {
            gx.shape.volume() as usize
        } else {
            0
        };
        let skip_i = if ids.len() > 1 { 1 } else { 0 };
        let base = gy.shape.lower_volume(dim) as usize;
        let skip = base * gx.shape[dim] as usize;
        let repeat = gy.shape.volume() as usize / base;
        unsafe {
            let mut src = const_ptr!(gy);
            let pgx = mut_ptr!(gx);
            for batch in 0..bs {
                let dest = pgx.add(batch * skip_x + base * ids[batch * skip_i] as usize);
                for i in 0..repeat {
                    let dp = dest.add(skip * i);
                    for j in 0..base {
                        *dp.add(j) += *src;
                        src = src.add(1);
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
    fn check_pick_fw_nn() {
        struct TestCase(Shape, u32, Vec<u32>, Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(
                shape![2, 2, 2; 3],
                0,
                vec![0, 0, 0],
                shape![1, 2, 2; 3],
                vec![0., 2., 4., 6., 8., 10., 12., 14., 16., 18., 20., 22.],
            ),
            TestCase(
                shape![2, 2, 2; 3],
                0,
                vec![1, 0, 1],
                shape![1, 2, 2; 3],
                vec![1., 3., 5., 7., 8., 10., 12., 14., 17., 19., 21., 23.],
            ),
            TestCase(
                shape![2, 2, 2; 3],
                0,
                vec![0],
                shape![1, 2, 2; 3],
                vec![0., 2., 4., 6., 8., 10., 12., 14., 16., 18., 20., 22.],
            ),
            TestCase(
                shape![2, 2, 2],
                0,
                vec![0, 1, 0],
                shape![1, 2, 2; 3],
                vec![0., 2., 4., 6., 1., 3., 5., 7., 0., 2., 4., 6.],
            ),
            TestCase(
                shape![2, 2, 2; 3],
                1,
                vec![0, 0, 0],
                shape![2, 1, 2; 3],
                vec![0., 1., 4., 5., 8., 9., 12., 13., 16., 17., 20., 21.],
            ),
            TestCase(
                shape![2, 2, 2; 3],
                2,
                vec![0, 0, 0],
                shape![2, 2, 1; 3],
                vec![0., 1., 2., 3., 8., 9., 10., 11., 16., 17., 18., 19.],
            ),
        ];
        let dev = D::Naive::new();
        for tc in &test_cases {
            let x_data = (0..tc.0.size()).map(|x| x as f32).collect::<Vec<f32>>();
            let x = dev.new_tensor_by_slice(tc.0, &x_data);
            let mut y = dev.new_tensor(tc.3);
            y.alloc();
            let mut u32data = vec![tc.1];
            u32data.append(&mut tc.2.clone());
            dev.call_fw_impl("pick_fw_impl", &[&x], &u32data, &[], &mut [&mut y]);
            assert_vector_ulps_eq!(tc.4, y.to_vec());
        }
    }

    #[test]
    fn check_pick_bw_n1() {
        struct TestCase(Shape, Vec<f32>, u32, Vec<u32>, Vec<f32>);
        let gx_data = vec![0., 1., 2., 3., 0., 1., 2., 3., 0., 1., 2., 3.];
        let test_cases = vec![
            TestCase(
                shape![1, 2; 3],
                vec![1., 1., 2., 2., 3., 3.],
                0,
                vec![0],
                vec![1., 1., 3., 3., 2., 1., 4., 3., 3., 1., 5., 3.],
            ),
            TestCase(
                shape![1, 2; 3],
                vec![1., 1., 2., 2., 3., 3.],
                0,
                vec![1],
                vec![0., 2., 2., 4., 0., 3., 2., 5., 0., 4., 2., 6.],
            ),
            TestCase(
                shape![2; 3],
                vec![1., 1., 2., 2., 3., 3.],
                1,
                vec![0],
                vec![1., 2., 2., 3., 2., 3., 2., 3., 3., 4., 2., 3.],
            ),
            TestCase(
                shape![2; 3],
                vec![1., 1., 2., 2., 3., 3.],
                1,
                vec![1],
                vec![0., 1., 3., 4., 0., 1., 4., 5., 0., 1., 5., 6.],
            ),
            TestCase(
                shape![2, 2; 3],
                vec![1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.],
                2,
                vec![0],
                vec![1., 2., 3., 4., 2., 3., 4., 5., 3., 4., 5., 6.],
            ),
        ];
        let dev = D::Naive::new();
        for tc in &test_cases {
            let gy = dev.new_tensor_by_slice(tc.0, &tc.1);
            let mut gx = dev.new_tensor_by_slice(shape![2, 2; 3], &gx_data);
            let mut u32data = vec![tc.2];
            u32data.append(&mut tc.3.clone());
            dev.call_bw_impl("pick_bw_impl", &[], &[], &[&gy], &u32data, &[], &mut gx);
            assert_vector_ulps_eq!(tc.4, gx.to_vec());
        }
    }

    #[test]
    fn check_pick_bw_1n() {
        struct TestCase(Shape, Vec<f32>, u32, Vec<u32>, Vec<f32>);
        let gx_data = vec![0., 1., 2., 3.];
        let test_cases = vec![
            TestCase(
                shape![1, 2; 3],
                vec![1., 1., 2., 2., 3., 3.],
                0,
                vec![0, 0, 0],
                vec![6., 1., 8., 3.],
            ),
            TestCase(
                shape![1, 2; 3],
                vec![1., 1., 2., 2., 3., 3.],
                0,
                vec![1, 1, 1],
                vec![0., 7., 2., 9.],
            ),
            TestCase(
                shape![1, 2; 3],
                vec![1., 1., 2., 2., 3., 3.],
                0,
                vec![0, 1, 0],
                vec![4., 3., 6., 5.],
            ),
            TestCase(
                shape![1, 2; 3],
                vec![1., 1., 2., 2., 3., 3.],
                0,
                vec![1, 0, 1],
                vec![2., 5., 4., 7.],
            ),
            TestCase(
                shape![2; 3],
                vec![1., 1., 2., 2., 3., 3.],
                1,
                vec![0, 0, 0],
                vec![6., 7., 2., 3.],
            ),
            TestCase(
                shape![2; 3],
                vec![1., 1., 2., 2., 3., 3.],
                1,
                vec![1, 1, 1],
                vec![0., 1., 8., 9.],
            ),
            TestCase(
                shape![2; 3],
                vec![1., 1., 2., 2., 3., 3.],
                1,
                vec![0, 1, 0],
                vec![4., 5., 4., 5.],
            ),
            TestCase(
                shape![2; 3],
                vec![1., 1., 2., 2., 3., 3.],
                1,
                vec![1, 0, 1],
                vec![2., 3., 6., 7.],
            ),
            TestCase(
                shape![2, 2; 3],
                vec![1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.],
                2,
                vec![0, 0, 0],
                vec![6., 7., 8., 9.],
            ),
        ];
        let dev = D::Naive::new();
        for tc in &test_cases {
            let gy = dev.new_tensor_by_slice(tc.0, &tc.1);
            let mut gx = dev.new_tensor_by_slice(shape![2, 2], &gx_data);
            let mut u32data = vec![tc.2];
            u32data.append(&mut tc.3.clone());
            dev.call_bw_impl("pick_bw_impl", &[], &[], &[&gy], &u32data, &[], &mut gx);
            assert_vector_ulps_eq!(tc.4, gx.to_vec());
        }
    }
}
