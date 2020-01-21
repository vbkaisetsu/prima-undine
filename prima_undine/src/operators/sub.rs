use std::cell::RefCell;

use crate::functions::{ArithmeticDeviceFunctions, BasicFunctions};
use crate::{Device, Tensor};

define_operator_ab!(
    Sub,
    sub_fw,
    |device: &Device, x: &[&Tensor], y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]| {
        device.sub_bw_a(x[0], x[1], y[0], gy[0], &mut *gx[0].borrow_mut());
        device.sub_bw_b(x[0], x[1], y[0], gy[0], &mut *gx[1].borrow_mut());
    }
);

define_operator_x!(SubConstL, sub_const_l_fw, sub_const_l_bw, k, f32);
define_operator_x!(SubConstR, sub_const_r_fw, sub_const_r_bw, k, f32);

define_operator_ab!(
    SubScalarL,
    sub_scalar_l_fw,
    |_device: &Device, _x: &[&Tensor], _y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]| {
        *gx[0].borrow_mut() -= gy[0];
        *gx[1].borrow_mut() += gy[0].flatten().sum(0);
    }
);

define_operator_ab!(
    SubScalarR,
    sub_scalar_r_fw,
    |_device: &Device, _x: &[&Tensor], _y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]| {
        *gx[0].borrow_mut() += gy[0];
        *gx[1].borrow_mut() -= gy[0].flatten().sum(0);
    }
);
