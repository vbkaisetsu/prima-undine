use crate::device_impl::FunctionFwU32Impl;
use crate::Tensor;

define_empty_impl!(ArgmaxImpl);
impl FunctionFwU32Impl for ArgmaxImpl {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], _f32data: &[f32], ys: &mut [u32]) {
        let x = xs[0];
        let dim = u32data[0];
        let y = ys;
        let s = x.shape;
        let n = s[dim] as usize;
        let repeat = s.size() as usize / n;
        let skip1 = s.lower_volume(dim) as usize;
        let skip2 = skip1 * n;
        unsafe {
            let src = const_ptr!(x);
            for i in 0..repeat {
                let mut offset = i % skip1 + (i / skip1) * skip2;
                let mut max_val = *src.add(offset);
                let mut argmax_val = 0;
                for j in 1..n {
                    offset += skip1;
                    if *src.add(offset) > max_val {
                        max_val = *src.add(offset);
                        argmax_val = j;
                    }
                }
                y[i] = argmax_val as u32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cmp;

    use crate::devices as D;
    use rand::seq::SliceRandom;

    #[test]
    fn check_argmax_dims() {
        let x_data = vec![
            0., 1., 2., 6., 7., 8., 3., 4., 5., -3., -4., -5., 0., -1., -2., -6., -7., -8.,
        ];
        let expected = vec![
            vec![2, 2, 2, 0, 0, 0],
            vec![1, 1, 1, 1, 1, 1],
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![3, 3; 2], &x_data);
        for &i in &[0, 1, 2] {
            let mut result = vec![0; (x.shape.size() / x.shape[i]) as usize];
            dev.call_fw_u32_impl("argmax_impl", &[&x], &[i], &[], &mut result);
            assert_eq!(expected[i as usize], result);
        }
    }

    #[test]
    fn check_argmax_large() {
        let ns = vec![
            1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024, 1025, 2047, 2048,
            2049,
            //65534, 65535, 65536,
        ];
        let mut rng = rand::thread_rng();
        let dev = D::Naive::new();
        for &n in &ns {
            let mut x_data = (0..n).map(|x| x as f32).collect::<Vec<f32>>();
            x_data.shuffle(&mut rng);
            let pos = x_data.iter().position(|&x| x == (n - 1) as f32).unwrap() as u32;
            let expected = vec![pos];
            let mut result = vec![0];
            let x = dev.new_tensor_by_slice(shape![n], &x_data);
            dev.call_fw_u32_impl("argmax_impl", &[&x], &[0], &[], &mut result);
            assert_eq!(expected, result);
        }
    }

    #[test]
    fn check_argmax_multiple_large() {
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
                x_data[i] = (n - 1) as f32;
            }
            x_data.shuffle(&mut rng);
            let pos = x_data.iter().position(|&x| x == (n - 1) as f32).unwrap() as u32;
            let expected = vec![pos];
            let mut result = vec![0];
            let x = dev.new_tensor_by_slice(shape![n], &x_data);
            dev.call_fw_u32_impl("argmax_impl", &[&x], &[0], &[], &mut result);
            assert_eq!(expected, result);
        }
    }
}
