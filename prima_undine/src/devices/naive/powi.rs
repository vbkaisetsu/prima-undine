use crate::device_impl::{FunctionBwImpl, FunctionFwImpl};
use crate::Tensor;

define_empty_impl!(PowiFwImpl);
impl FunctionFwImpl for PowiFwImpl {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], _f32data: &[f32], ys: &mut [&mut Tensor]) {
        let x = xs[0];
        let n = u32data[0] as i32;
        let y = &mut ys[0];
        let size = y.shape.size() as usize;
        unsafe {
            let px = const_ptr!(x);
            let py = mut_ptr!(y);
            for i in 0..size {
                *py.add(i) = (*px.add(i)).powi(n);
            }
        }
    }
}

define_empty_impl!(PowiBwImpl);
impl FunctionBwImpl for PowiBwImpl {
    fn call(
        &self,
        xs: &[&Tensor],
        _ys: &[&Tensor],
        gys: &[&Tensor],
        u32data: &[u32],
        _f32data: &[f32],
        gx: &mut Tensor,
    ) {
        let x = xs[0];
        let gy = gys[0];
        let n = u32data[0] as i32;
        let size = gy.shape.size() as usize;
        unsafe {
            let px = const_ptr!(x);
            let pgy = const_ptr!(gy);
            let pgx = mut_ptr!(gx);
            for i in 0..size {
                *pgx.add(i) += *pgy.add(i) * n as f32 * (*px.add(i)).powi(n - 1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::devices as D;
    use crate::functions::BasicFunctions;

    #[test]
    fn check_powi_fw() {
        let ns = vec![-8, -4, -3, -2, -1, 0, 1, 2, 3, 4, 8];
        let x_data = vec![0.1, 2., 1., 2.5, -0.1, 2., -1., -2.5];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        for &n in &ns {
            let y_data = x_data.iter().map(|&x| x.powi(n)).collect::<Vec<f32>>();
            let mut y = dev.new_tensor(shape![2, 2; 2]);
            y.alloc();
            dev.call_fw_impl("powi_fw_impl", &[&x], &[n as u32], &[], &mut [&mut y]);
            assert_vector_ulps_eq!(y_data, y.to_vec());
        }
    }

    #[test]
    fn check_powi_bw() {
        let ns = vec![-8, -4, -3, -2, -1, 0, 1, 2, 3, 4, 8];
        let x_data = vec![0.1, 2., 1., 2.5, -0.1, 2., -1., -2.5];
        let gy_data = vec![1., -1., 2., -2., 2., -2., 1., -1.];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![2, 2; 2], &x_data);
        let gy = dev.new_tensor_by_slice(shape![2, 2; 2], &gy_data);
        for &n in &ns {
            let y_data = x_data.iter().map(|&x| x.powi(n)).collect::<Vec<f32>>();
            let gx_data = x_data
                .iter()
                .zip(&gy_data)
                .map(|(&x, &gy)| 1. + gy * n as f32 * x.powi(n - 1))
                .collect::<Vec<f32>>();
            let y = dev.new_tensor_by_slice(shape![2, 2; 2], &y_data);
            let mut gx = dev.new_tensor_by_constant(shape![2, 2; 2], 1.);
            dev.call_bw_impl(
                "powi_bw_impl",
                &[&x],
                &[&y],
                &[&gy],
                &[n as u32],
                &[],
                &mut gx,
            );
            assert_vector_ulps_eq!(gx_data, gx.to_vec());
        }
    }
}
