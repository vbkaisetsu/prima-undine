use crate::{shape_ops, Device, Tensor};

pub trait BasicDeviceFunctions {
    // utility

    fn argmax(&self, x: &Tensor, dim: u32) -> Vec<u32>;
    fn argmin(&self, x: &Tensor, dim: u32) -> Vec<u32>;
    fn argsort(&self, x: &Tensor, dim: u32) -> Vec<u32>;

    // basic

    fn powf_fw(&self, a: &Tensor, b: &Tensor) -> Tensor;
    fn powf_const_l_fw(&self, x: &Tensor, k: f32) -> Tensor;
    fn powf_const_r_fw(&self, x: &Tensor, k: f32) -> Tensor;
    fn powf_scalar_l_fw(&self, x: &Tensor, k: &Tensor) -> Tensor;
    fn powf_scalar_r_fw(&self, x: &Tensor, k: &Tensor) -> Tensor;
    fn powi_fw(&self, x: &Tensor, k: i32) -> Tensor;
    fn sqrt_fw(&self, x: &Tensor) -> Tensor;
    fn abs_fw(&self, x: &Tensor) -> Tensor;

    fn powf_bw_a(&self, a: &Tensor, b: &Tensor, y: &Tensor, gy: &Tensor, ga: &mut Tensor);
    fn powf_bw_b(&self, a: &Tensor, b: &Tensor, y: &Tensor, gy: &Tensor, gb: &mut Tensor);
    fn powf_const_l_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, k: f32, gx: &mut Tensor);
    fn powf_const_r_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, k: f32, gx: &mut Tensor);
    fn powi_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, k: i32, gx: &mut Tensor);
    fn sqrt_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, gx: &mut Tensor);
    fn abs_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, gx: &mut Tensor);

    // trigonometric

    fn sin_fw(&self, x: &Tensor) -> Tensor;
    fn cos_fw(&self, x: &Tensor) -> Tensor;
    fn tan_fw(&self, x: &Tensor) -> Tensor;

    fn sin_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, gx: &mut Tensor);
    fn cos_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, gx: &mut Tensor);
    fn tan_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, gx: &mut Tensor);

    // exp

    fn exp_fw(&self, x: &Tensor) -> Tensor;
    fn ln_fw(&self, x: &Tensor) -> Tensor;
    fn tanh_fw(&self, x: &Tensor) -> Tensor;
    fn sigmoid_fw(&self, x: &Tensor) -> Tensor;
    fn softplus_fw(&self, x: &Tensor) -> Tensor;
    //fn softmax(&self, x: &Tensor, dim: u32) -> Tensor;
    //fn ln_softmax(&self, x: &Tensor, dim: u32) -> Tensor;
    //fn softmax_cross_entropy(&self, x: &Tensor, t: &Tensor, dim: u32) -> Tensor;

    fn exp_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, gx: &mut Tensor);
    fn ln_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, gx: &mut Tensor);
    fn tanh_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, gx: &mut Tensor);
    fn sigmoid_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, gx: &mut Tensor);

    // reduction

    fn sum_fw(&self, x: &Tensor, dim: u32) -> Tensor;
    fn logsumexp_fw(&self, x: &Tensor, dim: u32) -> Tensor;
    fn max_fw(&self, x: &Tensor, dim: u32) -> Tensor;
    fn min_fw(&self, x: &Tensor, dim: u32) -> Tensor;
    fn broadcast_fw(&self, x: &Tensor, dim: u32, size: u32) -> Tensor;

    fn max_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, dim: u32, gx: &mut Tensor);
    fn min_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, dim: u32, gx: &mut Tensor);

    // matrix

    fn matmul_fw(&self, a: &Tensor, b: &Tensor) -> Tensor;
    fn transpose_fw(&self, x: &Tensor) -> Tensor;
    //fn permute_dims(&self, x: &Tensor, perm: &[u32]) -> Tensor;
    //fn flip(&self, x: &Tensor, dim: u32) -> Tensor;
    fn triangular_l_fw(&self, x: &Tensor, k: u32) -> Tensor;
    fn triangular_u_fw(&self, x: &Tensor, k: u32) -> Tensor;

    fn matmul_bw_a(&self, a: &Tensor, b: &Tensor, y: &Tensor, gy: &Tensor, ga: &mut Tensor);
    fn matmul_bw_b(&self, a: &Tensor, b: &Tensor, y: &Tensor, gy: &Tensor, gb: &mut Tensor);
    fn transpose_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, gx: &mut Tensor);
    fn triangular_l_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, k: u32, gx: &mut Tensor);
    fn triangular_u_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, k: u32, gx: &mut Tensor);

    // ramp

    fn prelu_fw(&self, x: &Tensor, a: f32) -> Tensor;
    fn elu_fw(&self, x: &Tensor, a: f32) -> Tensor;

    fn prelu_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, a: f32, gx: &mut Tensor);
    fn elu_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, a: f32, gx: &mut Tensor);

    // manipulation

    fn slice_fw(&self, x: &Tensor, dim: u32, lower: u32, upper: u32) -> Tensor;
    fn pick_fw(&self, x: &Tensor, ids: &[u32], dim: u32) -> Tensor;
    fn concat_fw(&self, xs: &[&Tensor], dim: u32) -> Tensor;

    fn slice_bw(&self, gy: &Tensor, dim: u32, lower: u32, gx: &mut Tensor);
    fn pick_bw(&self, gy: &Tensor, ids: &[u32], dim: u32, gx: &mut Tensor);

    // batch

    fn batch_sum_fw(&self, x: &Tensor) -> Tensor;
    fn batch_slice_fw(&self, x: &Tensor, lower: u32, upper: u32) -> Tensor;
    fn batch_pick_fw(&self, x: &Tensor, ids: &[u32]) -> Tensor;
    fn batch_concat_fw(&self, xs: &[&Tensor]) -> Tensor;

    fn batch_slice_bw(&self, gy: &Tensor, lower: u32, gx: &mut Tensor);
    fn batch_pick_bw(&self, gy: &Tensor, ids: &[u32], gx: &mut Tensor);
}

impl<'dev> BasicDeviceFunctions for Device<'dev> {
    // utility

    fn argmax(&self, x: &Tensor, dim: u32) -> Vec<u32> {
        assert!(x.device() == self);
        let mut ret = vec![0; (x.shape.size() / x.shape[dim]) as usize];
        self.call_fw_u32_impl("argmax_impl", &[x], &[dim], &[], &mut ret);
        ret
    }

    fn argmin(&self, x: &Tensor, dim: u32) -> Vec<u32> {
        assert!(x.device() == self);
        let mut ret = vec![0; (x.shape.size() / x.shape[dim]) as usize];
        self.call_fw_u32_impl("argmin_impl", &[x], &[dim], &[], &mut ret);
        ret
    }

    fn argsort(&self, x: &Tensor, dim: u32) -> Vec<u32> {
        assert!(x.device() == self);
        let mut ret = vec![0; x.shape.size() as usize];
        self.call_fw_u32_impl("argsort_impl", &[x], &[dim], &[], &mut ret);
        ret
    }

    // basic

    define_fw_ab!(powf_fw, "powf_fw_impl", elementwise);
    define_fw_const!(powf_const_l_fw, "powf_const_l_fw_impl");
    define_fw_const!(powf_const_r_fw, "powf_const_r_fw_impl");
    define_fw_ab!(powf_scalar_l_fw, "powf_scalar_l_fw_impl", scalar_op);
    define_fw_ab!(powf_scalar_r_fw, "powf_scalar_r_fw_impl", scalar_op);

    fn powi_fw(&self, x: &Tensor, k: i32) -> Tensor {
        assert!(x.device() == self);
        let mut y = self.new_tensor(x.shape);
        y.alloc();
        self.call_fw_impl("powi_fw_impl", &[x], &[k as u32], &[], &mut [&mut y]);
        y
    }

    define_fw_x!(sqrt_fw, "sqrt_fw_impl");
    define_fw_x!(abs_fw, "abs_fw_impl");

    define_bw_ab_a!(powf_bw_a, "powf_bw_a_impl", elementwise);
    define_bw_ab_b!(powf_bw_b, "powf_bw_b_impl", elementwise);
    define_bw_const!(powf_const_l_bw, "powf_const_l_bw_impl");
    define_bw_const!(powf_const_r_bw, "powf_const_r_bw_impl");

    fn powi_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, k: i32, gx: &mut Tensor) {
        assert!(x.device() == self);
        assert!(y.device() == self);
        assert!(gy.device() == self);
        assert!(gx.device() == self);
        assert!(x.shape == gx.shape);
        assert!(y.shape == gy.shape);
        assert!(x.shape == y.shape);
        self.call_bw_impl("powi_bw_impl", &[x], &[y], &[gy], &[k as u32], &[], gx);
    }

    define_bw_x!(sqrt_bw, "sqrt_bw_impl");
    define_bw_x!(abs_bw, "abs_bw_impl");

    // trigonometric

    define_fw_x!(sin_fw, "sin_fw_impl");
    define_fw_x!(cos_fw, "cos_fw_impl");
    define_fw_x!(tan_fw, "tan_fw_impl");

    define_bw_x!(sin_bw, "sin_bw_impl");
    define_bw_x!(cos_bw, "cos_bw_impl");
    define_bw_x!(tan_bw, "tan_bw_impl");

    // exp

    define_fw_x!(exp_fw, "exp_fw_impl");
    define_fw_x!(ln_fw, "ln_fw_impl");
    define_fw_x!(tanh_fw, "tanh_fw_impl");
    define_fw_x!(sigmoid_fw, "sigmoid_fw_impl");
    define_fw_x!(softplus_fw, "softplus_fw_impl");

    define_bw_x!(exp_bw, "exp_bw_impl");
    define_bw_x!(ln_bw, "ln_bw_impl");
    define_bw_x!(tanh_bw, "tanh_bw_impl");
    define_bw_x!(sigmoid_bw, "sigmoid_bw_impl");

    // reduction

    fn sum_fw(&self, x: &Tensor, dim: u32) -> Tensor {
        assert!(x.device() == self);
        let mut y = self.new_tensor(x.shape.resize_dim(dim, 1));
        y.alloc();
        self.call_fw_impl("sum_fw_impl", &[x], &[dim], &[], &mut [&mut y]);
        y
    }

    fn max_fw(&self, x: &Tensor, dim: u32) -> Tensor {
        assert!(x.device() == self);
        let mut y = self.new_tensor(x.shape.resize_dim(dim, 1));
        y.alloc();
        self.call_fw_impl("max_fw_impl", &[x], &[dim], &[], &mut [&mut y]);
        y
    }

    fn min_fw(&self, x: &Tensor, dim: u32) -> Tensor {
        assert!(x.device() == self);
        let mut y = self.new_tensor(x.shape.resize_dim(dim, 1));
        y.alloc();
        self.call_fw_impl("min_fw_impl", &[x], &[dim], &[], &mut [&mut y]);
        y
    }

    fn logsumexp_fw(&self, x: &Tensor, dim: u32) -> Tensor {
        assert!(x.device() == self);
        let mut y = self.new_tensor(x.shape.resize_dim(dim, 1));
        y.alloc();
        self.call_fw_impl("logsumexp_fw_impl", &[x], &[dim], &[], &mut [&mut y]);
        y
    }

    fn broadcast_fw(&self, x: &Tensor, dim: u32, size: u32) -> Tensor {
        assert!(x.device() == self);
        let mut y = self.new_tensor(shape_ops::broadcast(x.shape, dim, size));
        y.alloc();
        self.call_fw_impl("broadcast_fw_impl", &[x], &[dim, size], &[], &mut [&mut y]);
        y
    }

    fn max_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, dim: u32, gx: &mut Tensor) {
        assert!(x.device() == self);
        assert!(y.device() == self);
        assert!(gy.device() == self);
        assert!(gx.device() == self);
        assert!(x.shape == gx.shape);
        assert!(y.shape == gy.shape);
        assert!(y.shape == x.shape.resize_dim(dim, 1));
        self.call_bw_impl("max_bw_impl", &[x], &[y], &[gy], &[dim], &[], gx);
    }

    fn min_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, dim: u32, gx: &mut Tensor) {
        assert!(x.device() == self);
        assert!(y.device() == self);
        assert!(gy.device() == self);
        assert!(gx.device() == self);
        assert!(x.shape == gx.shape);
        assert!(y.shape == gy.shape);
        assert!(y.shape == x.shape.resize_dim(dim, 1));
        self.call_bw_impl("min_bw_impl", &[x], &[y], &[gy], &[dim], &[], gx);
    }

    // matrix

    fn matmul_fw(&self, a: &Tensor, b: &Tensor) -> Tensor {
        assert!(a.device() == self);
        assert!(b.device() == self);
        let mut y = self.new_tensor(shape_ops::matmul(a.shape, b.shape));
        y.alloc();
        self.call_fw_impl("matmul_fw_impl", &[a, b], &[], &[], &mut [&mut y]);
        y
    }

    fn transpose_fw(&self, x: &Tensor) -> Tensor {
        assert!(x.device() == self);
        let mut y = self.new_tensor(shape_ops::transpose(x.shape));
        y.alloc();
        self.call_fw_impl("transpose_fw_impl", &[x], &[], &[], &mut [&mut y]);
        y
    }

    fn triangular_l_fw(&self, x: &Tensor, k: u32) -> Tensor {
        assert!(x.device() == self);
        let xs = x.shape;
        assert!(xs.is_matrix() && xs[0] == xs[1]);
        let mut y = self.new_tensor(xs);
        y.alloc();
        self.call_fw_impl("triangular_l_fw_impl", &[x], &[k], &[], &mut [&mut y]);
        y
    }

    fn triangular_u_fw(&self, x: &Tensor, k: u32) -> Tensor {
        assert!(x.device() == self);
        assert!(x.shape.is_matrix() && x.shape[0] == x.shape[1]);
        let mut y = self.new_tensor(x.shape);
        y.alloc();
        self.call_fw_impl("triangular_u_fw_impl", &[x], &[k], &[], &mut [&mut y]);
        y
    }

    fn matmul_bw_a(&self, a: &Tensor, b: &Tensor, y: &Tensor, gy: &Tensor, ga: &mut Tensor) {
        assert!(a.device() == self);
        assert!(b.device() == self);
        assert!(y.device() == self);
        assert!(gy.device() == self);
        assert!(ga.device() == self);
        assert!(a.shape == ga.shape);
        assert!(y.shape == gy.shape);
        assert!(y.shape == shape_ops::matmul(a.shape, b.shape));
        self.call_bw_impl("matmul_bw_a_impl", &[a, b], &[y], &[gy], &[], &[], ga);
    }

    fn matmul_bw_b(&self, a: &Tensor, b: &Tensor, y: &Tensor, gy: &Tensor, gb: &mut Tensor) {
        assert!(a.device() == self);
        assert!(b.device() == self);
        assert!(y.device() == self);
        assert!(gy.device() == self);
        assert!(gb.device() == self);
        assert!(b.shape == gb.shape);
        assert!(y.shape == gy.shape);
        assert!(y.shape == shape_ops::matmul(a.shape, b.shape));
        self.call_bw_impl("matmul_bw_b_impl", &[a, b], &[y], &[gy], &[], &[], gb);
    }

    fn transpose_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, gx: &mut Tensor) {
        assert!(x.device() == self);
        assert!(y.device() == self);
        assert!(gy.device() == self);
        assert!(gx.device() == self);
        assert!(x.shape == gx.shape);
        assert!(y.shape == gy.shape);
        assert!(y.shape == shape_ops::transpose(x.shape));
        self.call_bw_impl("transpose_bw_impl", &[x], &[y], &[gy], &[], &[], gx);
    }

    fn triangular_l_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, k: u32, gx: &mut Tensor) {
        assert!(x.device() == self);
        assert!(y.device() == self);
        assert!(gy.device() == self);
        assert!(gx.device() == self);
        assert!(x.shape == gx.shape);
        assert!(y.shape == gy.shape);
        assert!(y.shape == x.shape);
        self.call_bw_impl("triangular_l_bw_impl", &[x], &[y], &[gy], &[k], &[], gx);
    }

    fn triangular_u_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, k: u32, gx: &mut Tensor) {
        assert!(x.device() == self);
        assert!(y.device() == self);
        assert!(gy.device() == self);
        assert!(gx.device() == self);
        assert!(x.shape == gx.shape);
        assert!(y.shape == gy.shape);
        assert!(y.shape == x.shape);
        self.call_bw_impl("triangular_u_bw_impl", &[x], &[y], &[gy], &[k], &[], gx);
    }

    // ramp

    define_fw_const!(prelu_fw, "prelu_fw_impl");
    define_fw_const!(elu_fw, "elu_fw_impl");

    define_bw_const!(prelu_bw, "prelu_bw_impl");
    define_bw_const!(elu_bw, "elu_bw_impl");

    // manipulation

    fn slice_fw(&self, x: &Tensor, dim: u32, lower: u32, upper: u32) -> Tensor {
        assert!(x.device() == self);
        let mut y = self.new_tensor(shape_ops::slice(x.shape, dim, lower, upper));
        y.alloc();
        self.call_fw_impl("slice_fw_impl", &[x], &[dim, lower], &[], &mut [&mut y]);
        y
    }

    fn pick_fw(&self, x: &Tensor, ids: &[u32], dim: u32) -> Tensor {
        assert!(x.device() == self);
        let mut u32data = vec![0; ids.len() + 1];
        u32data[0] = dim;
        u32data[1..].clone_from_slice(ids);
        let mut y = self.new_tensor(shape_ops::pick(x.shape, ids, dim));
        y.alloc();
        self.call_fw_impl("pick_fw_impl", &[x], &u32data, &[], &mut [&mut y]);
        y
    }

    fn concat_fw(&self, xs: &[&Tensor], dim: u32) -> Tensor {
        assert!(xs.len() != 0);
        let mut shapes = vec![];
        for x in xs {
            assert!(x.device() == self);
            shapes.push(x.shape);
        }
        let mut y = self.new_tensor(shape_ops::concat(&shapes, dim));
        y.alloc();
        self.call_fw_impl("concat_fw_impl", xs, &[dim], &[], &mut [&mut y]);
        y
    }

    fn slice_bw(&self, gy: &Tensor, dim: u32, lower: u32, gx: &mut Tensor) {
        assert!(gy.device() == self);
        assert!(gx.device() == self);
        let sy = gy.shape;
        assert!(shape_ops::slice(gx.shape, dim, lower, lower + sy[dim]) == sy);
        self.call_bw_impl("slice_bw_impl", &[], &[], &[gy], &[dim, lower], &[], gx);
    }

    fn pick_bw(&self, gy: &Tensor, ids: &[u32], dim: u32, gx: &mut Tensor) {
        assert!(gy.device() == self);
        assert!(gx.device() == self);
        assert!(shape_ops::pick(gx.shape, ids, dim) == gy.shape);
        let mut u32data = vec![0; ids.len() + 1];
        u32data[0] = dim;
        u32data[1..].clone_from_slice(ids);
        self.call_bw_impl("pick_bw_impl", &[], &[], &[gy], &u32data, &[], gx);
    }

    // batch

    fn batch_sum_fw(&self, x: &Tensor) -> Tensor {
        assert!(x.device() == self);
        let mut y = self.new_tensor(x.shape.resize_batch(1));
        y.alloc();
        self.call_fw_impl("batch_sum_fw_impl", &[x], &[], &[], &mut [&mut y]);
        y
    }

    fn batch_slice_fw(&self, x: &Tensor, lower: u32, upper: u32) -> Tensor {
        assert!(x.device() == self);
        let mut y = self.new_tensor(shape_ops::batch_slice(x.shape, lower, upper));
        y.alloc();
        self.call_fw_impl(
            "batch_slice_fw_impl",
            &[x],
            &[lower, upper],
            &[],
            &mut [&mut y],
        );
        y
    }

    fn batch_pick_fw(&self, x: &Tensor, ids: &[u32]) -> Tensor {
        assert!(x.device() == self);
        let mut y = self.new_tensor(shape_ops::batch_pick(x.shape, ids));
        y.alloc();
        self.call_fw_impl("batch_pick_fw_impl", &[x], ids, &[], &mut [&mut y]);
        y
    }

    fn batch_concat_fw(&self, xs: &[&Tensor]) -> Tensor {
        assert!(xs.len() != 0);
        let mut shapes = vec![];
        for x in xs {
            assert!(x.device() == self);
            shapes.push(x.shape);
        }
        let mut y = self.new_tensor(shape_ops::batch_concat(&shapes));
        y.alloc();
        self.call_fw_impl("batch_concat_fw_impl", xs, &[], &[], &mut [&mut y]);
        y
    }

    fn batch_slice_bw(&self, gy: &Tensor, lower: u32, gx: &mut Tensor) {
        assert!(gy.device() == self);
        assert!(gx.device() == self);
        let sy = gy.shape;
        assert!(shape_ops::batch_slice(gx.shape, lower, lower + sy.batch()) == sy);
        self.call_bw_impl("slice_bw_impl", &[], &[], &[gy], &[lower], &[], gx);
    }

    fn batch_pick_bw(&self, gy: &Tensor, ids: &[u32], gx: &mut Tensor) {
        assert!(gy.device() == self);
        assert!(gx.device() == self);
        assert!(shape_ops::batch_pick(gx.shape, ids) == gy.shape);
        self.call_bw_impl("pick_bw_impl", &[], &[], &[gy], ids, &[], gx);
    }
}

#[cfg(test)]
mod tests {
    use super::BasicDeviceFunctions;
    use crate::devices as D;

    #[test]
    fn check_argmax() {
        let dev = D::Naive::new();
        let x_data = vec![1., 2., 3., 4., 2., 3., 4., 1., 3., 4., 1., 2.];
        let y1_data = vec![3, 2, 1];
        let y2_data = vec![2, 2, 1, 0];
        let x = dev.new_tensor_by_slice(shape![4, 3], &x_data);
        let y1 = dev.argmax(&x, 0);
        let y2 = dev.argmax(&x, 1);
        assert_eq!(y1_data, y1.to_vec());
        assert_eq!(y2_data, y2.to_vec());
    }
}
