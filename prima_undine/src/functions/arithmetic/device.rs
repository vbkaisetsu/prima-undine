use crate::{shape_ops, Device, Tensor};

pub trait ArithmeticDeviceFunctions {
    fn add_assign(&self, x: &Tensor, y: &mut Tensor);
    fn sub_assign(&self, x: &Tensor, y: &mut Tensor);
    fn mul_assign_const(&self, k: f32, y: &mut Tensor);

    fn neg_fw(&self, x: &Tensor) -> Tensor;
    fn neg_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, gx: &mut Tensor);

    fn add_fw(&self, a: &Tensor, b: &Tensor) -> Tensor;
    fn add_bw_a(&self, a: &Tensor, b: &Tensor, y: &Tensor, gy: &Tensor, ga: &mut Tensor);
    fn add_bw_b(&self, a: &Tensor, b: &Tensor, y: &Tensor, gy: &Tensor, gb: &mut Tensor);
    fn add_const_fw(&self, x: &Tensor, k: f32) -> Tensor;
    fn add_const_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, k: f32, gx: &mut Tensor);
    fn add_scalar_fw(&self, a: &Tensor, b: &Tensor) -> Tensor;

    fn sub_fw(&self, a: &Tensor, b: &Tensor) -> Tensor;
    fn sub_bw_a(&self, a: &Tensor, b: &Tensor, y: &Tensor, gy: &Tensor, ga: &mut Tensor);
    fn sub_bw_b(&self, a: &Tensor, b: &Tensor, y: &Tensor, gy: &Tensor, gb: &mut Tensor);
    fn sub_const_l_fw(&self, x: &Tensor, k: f32) -> Tensor;
    fn sub_const_l_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, k: f32, gx: &mut Tensor);
    fn sub_const_r_fw(&self, x: &Tensor, k: f32) -> Tensor;
    fn sub_const_r_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, k: f32, gx: &mut Tensor);
    fn sub_scalar_l_fw(&self, a: &Tensor, b: &Tensor) -> Tensor;
    fn sub_scalar_r_fw(&self, a: &Tensor, b: &Tensor) -> Tensor;

    fn mul_fw(&self, a: &Tensor, b: &Tensor) -> Tensor;
    fn mul_bw_a(&self, a: &Tensor, b: &Tensor, y: &Tensor, gy: &Tensor, ga: &mut Tensor);
    fn mul_bw_b(&self, a: &Tensor, b: &Tensor, y: &Tensor, gy: &Tensor, gb: &mut Tensor);
    fn mul_const_fw(&self, x: &Tensor, k: f32) -> Tensor;
    fn mul_const_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, k: f32, gx: &mut Tensor);
    fn mul_scalar_fw(&self, a: &Tensor, b: &Tensor) -> Tensor;

    fn div_fw(&self, a: &Tensor, b: &Tensor) -> Tensor;
    fn div_bw_a(&self, a: &Tensor, b: &Tensor, y: &Tensor, gy: &Tensor, ga: &mut Tensor);
    fn div_bw_b(&self, a: &Tensor, b: &Tensor, y: &Tensor, gy: &Tensor, gb: &mut Tensor);
    fn div_const_l_fw(&self, x: &Tensor, k: f32) -> Tensor;
    fn div_const_l_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, k: f32, gx: &mut Tensor);
    fn div_const_r_fw(&self, x: &Tensor, k: f32) -> Tensor;
    fn div_const_r_bw(&self, x: &Tensor, y: &Tensor, gy: &Tensor, k: f32, gx: &mut Tensor);
    fn div_scalar_l_fw(&self, a: &Tensor, b: &Tensor) -> Tensor;
    fn div_scalar_r_fw(&self, a: &Tensor, b: &Tensor) -> Tensor;
}

impl<'dev> ArithmeticDeviceFunctions for Device<'dev> {
    fn add_assign(&self, x: &Tensor, y: &mut Tensor) {
        assert!(x.device() == self);
        assert!(y.device() == self);
        self.call_fw_impl("add_assign_impl", &[x], &[], &[], &mut [y]);
    }

    fn sub_assign(&self, x: &Tensor, y: &mut Tensor) {
        assert!(x.device() == self);
        assert!(y.device() == self);
        self.call_fw_impl("sub_assign_impl", &[x], &[], &[], &mut [y]);
    }

    fn mul_assign_const(&self, k: f32, y: &mut Tensor) {
        assert!(y.device() == self);
        self.call_fw_impl("mul_assign_const_impl", &[], &[], &[k], &mut [y]);
    }

    define_fw_x!(neg_fw, "neg_fw_impl");
    define_bw_x!(neg_bw, "neg_bw_impl");

    define_fw_ab!(add_fw, "add_fw_impl", elementwise);
    define_bw_ab_a!(add_bw_a, "add_bw_a_impl", elementwise);
    define_bw_ab_b!(add_bw_b, "add_bw_b_impl", elementwise);
    define_fw_const!(add_const_fw, "add_const_fw_impl");
    define_bw_const!(add_const_bw, "add_const_bw_impl");
    define_fw_ab!(add_scalar_fw, "add_scalar_fw_impl", scalar_op);

    define_fw_ab!(sub_fw, "sub_fw_impl", elementwise);
    define_bw_ab_a!(sub_bw_a, "sub_bw_a_impl", elementwise);
    define_bw_ab_b!(sub_bw_b, "sub_bw_b_impl", elementwise);
    define_fw_const!(sub_const_l_fw, "sub_const_l_fw_impl");
    define_bw_const!(sub_const_l_bw, "sub_const_l_bw_impl");
    define_fw_const!(sub_const_r_fw, "sub_const_r_fw_impl");
    define_bw_const!(sub_const_r_bw, "sub_const_r_bw_impl");
    define_fw_ab!(sub_scalar_l_fw, "sub_scalar_l_fw_impl", scalar_op);
    define_fw_ab!(sub_scalar_r_fw, "sub_scalar_r_fw_impl", scalar_op);

    define_fw_ab!(mul_fw, "mul_fw_impl", elementwise);
    define_bw_ab_a!(mul_bw_a, "mul_bw_a_impl", elementwise);
    define_bw_ab_b!(mul_bw_b, "mul_bw_b_impl", elementwise);
    define_fw_const!(mul_const_fw, "mul_const_fw_impl");
    define_bw_const!(mul_const_bw, "mul_const_bw_impl");
    define_fw_ab!(mul_scalar_fw, "mul_scalar_fw_impl", scalar_op);

    define_fw_ab!(div_fw, "div_fw_impl", elementwise);
    define_bw_ab_a!(div_bw_a, "div_bw_a_impl", elementwise);
    define_bw_ab_b!(div_bw_b, "div_bw_b_impl", elementwise);
    define_fw_const!(div_const_l_fw, "div_const_l_fw_impl");
    define_bw_const!(div_const_l_bw, "div_const_l_bw_impl");
    define_fw_const!(div_const_r_fw, "div_const_r_fw_impl");
    define_bw_const!(div_const_r_bw, "div_const_r_bw_impl");
    define_fw_ab!(div_scalar_l_fw, "div_scalar_l_fw_impl", scalar_op);
    define_fw_ab!(div_scalar_r_fw, "div_scalar_r_fw_impl", scalar_op);
}
