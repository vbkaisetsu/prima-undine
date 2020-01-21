use crate::device_impl::FunctionFwU32Impl;
use crate::Tensor;

define_empty_impl!(ArgsortImpl);
impl FunctionFwU32Impl for ArgsortImpl {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], _f32data: &[f32], ys: &mut [u32]) {
        let x = xs[0];
        let dim = u32data[0];
        let y = ys;
        let s = x.shape;
        let size = s.size() as usize;
        let skip = s.lower_volume(dim) as usize;
        let len = s[dim] as usize;
        let mut indices = vec![0; size];
        for offset in (0..size).step_by(skip * len) {
            for j in 0..skip {
                for i in 0..len {
                    indices[offset + j * len + i] = offset + j + skip * i;
                }
            }
        }
        unsafe {
            let src = const_ptr!(x);
            for offset in (0..size).step_by(len) {
                indices[offset..offset + len]
                    .sort_unstable_by(|&i, &j| (*src.add(i)).partial_cmp(&(*src.add(j))).unwrap());
            }
        }
        for offset in (0..size).step_by(skip * len) {
            for j in 0..skip {
                for i in 0..len {
                    y[offset + j + skip * i] = indices[offset + j * len + i] as u32;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::devices as D;
    use crate::functions::BasicFunctions;
    use crate::shape;
    use rand::seq::SliceRandom;

    #[test]
    fn check_argsort_dims() {
        let x_data = vec![
            10., 17., 6., 8., 16., 18., 14., 15., 4., 3., 13., 11., 9., 12., 2., 1., 5., 7.,
        ];
        let expected = vec![
            vec![
                6., 10., 17., 8., 16., 18., 4., 14., 15., 3., 11., 13., 2., 9., 12., 1., 5., 7.,
            ],
            vec![
                8., 15., 4., 10., 16., 6., 14., 17., 18., 1., 5., 2., 3., 12., 7., 9., 13., 11.,
            ],
            vec![
                10., 17., 6., 8., 16., 18., 14., 15., 4., 3., 13., 11., 9., 12., 2., 1., 5., 7.,
            ],
        ];
        let dev = D::Naive::new();
        let x = dev.new_tensor_by_slice(shape![3, 3; 2], &x_data);
        for &i in &[0, 1, 2] {
            let mut result = vec![0; x.shape().size() as usize];
            dev.call_fw_u32_impl("argsort_impl", &[&x], &[i], &[], &mut result);
            let sorted = result
                .iter()
                .map(|&i| x_data[i as usize])
                .collect::<Vec<f32>>();
            assert_eq!(expected[i as usize], sorted);
        }
    }

    #[test]
    fn check_index_sort_large() {
        let ns = vec![
            1, 2, 3, 15, 16, 17, 255, 256, 257, 1023, 1024, 1025, 2047, 2048, 2049, 65534, 65535,
            65536,
        ];
        let mut rng = rand::thread_rng();
        let dev = D::Naive::new();
        for &n in &ns {
            let mut x_data = (0..n).map(|x| x as f32).collect::<Vec<f32>>();
            let expected = x_data.clone();
            x_data.shuffle(&mut rng);
            let x = dev.new_tensor_by_slice(shape![n], &x_data);
            let mut result = vec![0; n as usize];
            dev.call_fw_u32_impl("argsort_impl", &[&x], &[0], &[], &mut result);
            let sorted = result
                .iter()
                .map(|&i| x_data[i as usize])
                .collect::<Vec<f32>>();
            assert_eq!(expected, sorted);
        }
    }
}
