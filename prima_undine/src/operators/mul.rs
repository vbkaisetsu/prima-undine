use std::cell::RefCell;

use crate::functions::{ArithmeticDeviceFunctions, BasicFunctions};
use crate::{Device, Tensor};

define_operator_ab!(
    Mul,
    mul_fw,
    |device: &Device, x: &[&Tensor], y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]| {
        device.mul_bw_a(x[0], x[1], y[0], gy[0], &mut *gx[0].borrow_mut());
        device.mul_bw_b(x[0], x[1], y[0], gy[0], &mut *gx[1].borrow_mut());
    }
);

define_operator_x!(MulConst, mul_const_fw, mul_const_bw, k, f32);

define_operator_ab!(
    MulScalar,
    mul_scalar_fw,
    |_device: &Device, x: &[&Tensor], _y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]| {
        *gx[0].borrow_mut() += x[1] * gy[0];
        *gx[1].borrow_mut() += (x[0] * gy[0]).flatten().sum(0);
    }
);
