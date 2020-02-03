use std::cell::RefCell;

use crate::functions::{BasicDeviceFunctions, BasicFunctions};
use crate::{Device, Tensor};

define_operator_ab!(
    Pow,
    powf_fw,
    |device: &Device, x: &[&Tensor], y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]| {
        device.powf_bw_a(x[0], x[1], y[0], gy[0], &mut *gx[0].borrow_mut());
        device.powf_bw_b(x[0], x[1], y[0], gy[0], &mut *gx[1].borrow_mut());
    }
);

define_operator_x!(PowConstL, powf_const_l_fw, powf_const_l_bw, k, f32);
define_operator_x!(PowConstR, powf_const_r_fw, powf_const_r_bw, k, f32);

define_operator_ab!(
    PowScalarL,
    powf_scalar_l_fw,
    |_device: &Device, _x: &[&Tensor], _y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]| {
        *gx[0].borrow_mut() -= gy[0];
        *gx[1].borrow_mut() += gy[0].flatten().sum(0);
    }
);

// y = k - x
// dy/dx = -1
// dy/dk = 1
define_operator_ab!(
    PowScalarR,
    powf_scalar_r_fw,
    |device: &Device, x: &[&Tensor], y: &[&Tensor], gy: &[&Tensor], gx: &[&RefCell<Tensor>]| {
        device.powf_bw_a(x[0], x[1], y[0], gy[0], &mut *gx[0].borrow_mut());
        device.powf_bw_b(x[0], x[1], y[0], gy[0], &mut *gx[1].borrow_mut());
    }
);
