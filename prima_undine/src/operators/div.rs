use std::cell::RefCell;

use crate::functions::{ArithmeticDeviceFunctions, BasicFunctions};
use crate::{Device, Tensor};

define_operator_ab!(
    Div,
    div_fw,
    |device: &Device, x: &[&Tensor], y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]| {
        device.div_bw_a(x[0], x[1], y[0], gy[0], &mut *gx[0].borrow_mut());
        device.div_bw_b(x[0], x[1], y[0], gy[0], &mut *gx[1].borrow_mut());
    }
);

define_operator_x!(DivConstL, div_const_l_fw, div_const_l_bw, k, f32);
define_operator_x!(DivConstR, div_const_r_fw, div_const_r_bw, k, f32);

define_operator_ab!(
    DivScalarL,
    div_scalar_l_fw,
    |_device: &Device, x: &[&Tensor], y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]| {
        let ref a = gy[0] / x[0];
        *gx[0].borrow_mut() -= &(a * y[0]);
        *gx[1].borrow_mut() += &a.flatten().sum(0);
    }
);

define_operator_ab!(
    DivScalarR,
    div_scalar_r_fw,
    |_device: &Device, x: &[&Tensor], y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]| {
        let ref a = gy[0] / x[1];
        *gx[0].borrow_mut() += a;
        *gx[1].borrow_mut() -= (a * y[0]).flatten().sum(0);
    }
);
