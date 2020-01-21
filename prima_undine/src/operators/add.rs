use std::cell::RefCell;

use crate::functions::{ArithmeticDeviceFunctions, BasicFunctions};
use crate::{Device, Tensor};

define_operator_ab!(
    Add,
    add_fw,
    |device: &Device, x: &[&Tensor], y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]| {
        device.add_bw_a(x[0], x[1], y[0], gy[0], &mut *gx[0].borrow_mut());
        device.add_bw_b(x[0], x[1], y[0], gy[0], &mut *gx[1].borrow_mut());
    }
);

define_operator_x!(AddConst, add_const_fw, add_const_bw, k, f32);

define_operator_ab!(
    AddScalar,
    add_scalar_fw,
    |_device: &Device, _x: &[&Tensor], _y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]| {
        *gx[0].borrow_mut() += gy[0];
        *gx[1].borrow_mut() += gy[0].flatten().sum(0);
    }
);
