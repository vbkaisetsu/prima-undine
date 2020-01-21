use crate::device_impl::FunctionFwImpl;
use crate::Tensor;

define_empty_impl!(IdentityImpl);
impl FunctionFwImpl for IdentityImpl {
    fn call(&self, _xs: &[&Tensor], _u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let y = &mut ys[0];
        y.reset(0.);
        let size = y.shape[0] as usize;
        unsafe {
            let py = mut_ptr!(y);
            for i in 0..size {
                *py.add(i * (size + 1)) = 1.;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::functions::BasicFunctions;
    use crate::{devices as D, Shape};

    #[test]
    fn check_identity() {
        struct TestCase(Shape, Vec<f32>);
        let test_cases = vec![
            TestCase(shape![], vec![1.]),
            TestCase(shape![2, 2], vec![1., 0., 0., 1.]),
            TestCase(shape![3, 3], vec![1., 0., 0., 0., 1., 0., 0., 0., 1.]),
            TestCase(
                shape![4, 4],
                vec![
                    1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
                ],
            ),
        ];
        let dev = D::Naive::new();
        for tc in &test_cases {
            let mut y = dev.new_tensor(tc.0);
            y.alloc();
            dev.call_fw_impl("identity_impl", &[], &[], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(tc.1, y.to_vec());
        }
    }
}
