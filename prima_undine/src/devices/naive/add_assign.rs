use std::cmp;

use crate::device_impl::FunctionFwImpl;
use crate::Tensor;

define_empty_impl!(AddAssignImpl);
impl FunctionFwImpl for AddAssignImpl {
    fn call(&self, xs: &[&Tensor], _u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let y = &mut ys[0];
        let volume = x.shape.volume() as usize;
        let x_skip = if x.shape.has_batch() { volume } else { 0 };
        let y_skip = if y.shape.has_batch() { volume } else { 0 };
        let batch = cmp::max(x.shape.batch(), y.shape.batch()) as usize;
        unsafe {
            let mut px = const_ptr!(x);
            let mut py = mut_ptr!(y);
            for _ in 0..batch {
                for i in 0..volume {
                    *py.add(i) += *px.add(i);
                }
                px = px.add(x_skip);
                py = py.add(y_skip);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::devices as D;
    use crate::functions::BasicFunctions;

    #[test]
    fn check_add_assign() {
        let a_data = vec![1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.0001];
        let b_data = vec![0., 100., 20., 3., 0.4, 0.05, 0.006, 0.0007];
        let y_data = vec![1000., 200., 30., 4., 0.5, 0.06, 0.007, 0.0008];
        let dev = D::Naive::new();
        let a = dev.new_tensor_by_slice(shape![2, 2; 2], &a_data);
        let mut b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
        dev.call_fw_impl("add_assign_impl", &[&a], &[], &[], &mut [&mut b]);
        assert_vector_ulps_eq!(y_data, b.to_vec());
    }

    #[test]
    fn check_add_assign_batch_broadcast() {
        let a_data = vec![0., 1., 2., 3.];
        let b_data = vec![-2., -2., -2., -2., 4., 4., 4., 4.];
        let y1_data = vec![-2., -1., 0., 1., 4., 5., 6., 7.];
        let y2_data = vec![2., 3., 4., 5.];
        let dev = D::Naive::new();
        {
            let a = dev.new_tensor_by_slice(shape![2, 2], &a_data);
            let mut b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
            dev.call_fw_impl("add_assign_impl", &[&a], &[], &[], &mut [&mut b]);
            assert_vector_ulps_eq!(y1_data, b.to_vec());
        }
        {
            let b = dev.new_tensor_by_slice(shape![2, 2; 2], &b_data);
            let mut a = dev.new_tensor_by_slice(shape![2, 2], &a_data);
            dev.call_fw_impl("add_assign_impl", &[&b], &[], &[], &mut [&mut a]);
            assert_vector_ulps_eq!(y2_data, a.to_vec());
        }
    }
}
