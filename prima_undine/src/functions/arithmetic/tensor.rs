use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Div;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Neg;
use std::ops::Sub;
use std::ops::SubAssign;

use crate::functions::{ArithmeticDeviceFunctions, ArithmeticFunctions};
use crate::Tensor;

impl<'dev> AddAssign<Tensor<'_>> for Tensor<'dev> {
    fn add_assign(&mut self, other: Tensor) {
        self.device().add_assign(&other, self);
    }
}

impl<'dev> SubAssign<Tensor<'_>> for Tensor<'dev> {
    fn sub_assign(&mut self, other: Tensor) {
        self.device().sub_assign(&other, self);
    }
}

impl<'dev> MulAssign<f32> for Tensor<'dev> {
    fn mul_assign(&mut self, k: f32) {
        self.device().mul_assign_const(k, self);
    }
}

impl<'dev> AddAssign<&Tensor<'_>> for Tensor<'dev> {
    fn add_assign(&mut self, other: &Tensor) {
        self.device().add_assign(other, self);
    }
}

impl<'dev> SubAssign<&Tensor<'_>> for Tensor<'dev> {
    fn sub_assign(&mut self, other: &Tensor) {
        self.device().sub_assign(other, self);
    }
}

impl<'dev> Neg for Tensor<'dev> {
    type Output = Tensor<'dev>;
    fn neg(self) -> Self::Output {
        self.device().neg_fw(&self)
    }
}

impl<'dev> Neg for &Tensor<'dev> {
    type Output = Tensor<'dev>;
    fn neg(self) -> Self::Output {
        self.device().neg_fw(self)
    }
}

macro_rules! define_ab_op {
    ( $name:ident, $fn:ident, $fw:ident, $scalar_l_fw:ident, $scalar_r_fw:ident ) => {
        impl<'dev> $name<Tensor<'dev>> for Tensor<'dev> {
            type Output = Tensor<'dev>;
            fn $fn(self, rhs: Tensor) -> Self::Output {
                if self.shape.is_scalar() {
                    self.device().$scalar_l_fw(&rhs, &self)
                } else if rhs.shape.is_scalar() {
                    self.device().$scalar_r_fw(&self, &rhs)
                } else {
                    self.device().$fw(&self, &rhs)
                }
            }
        }
        impl<'dev> $name<&Tensor<'dev>> for Tensor<'dev> {
            type Output = Tensor<'dev>;
            fn $fn(self, rhs: &Tensor) -> Self::Output {
                if self.shape.is_scalar() {
                    self.device().$scalar_l_fw(rhs, &self)
                } else if rhs.shape.is_scalar() {
                    self.device().$scalar_r_fw(&self, rhs)
                } else {
                    self.device().$fw(&self, rhs)
                }
            }
        }
        impl<'dev> $name<Tensor<'dev>> for &Tensor<'dev> {
            type Output = Tensor<'dev>;
            fn $fn(self, rhs: Tensor) -> Self::Output {
                if self.shape.is_scalar() {
                    self.device().$scalar_l_fw(&rhs, self)
                } else if rhs.shape.is_scalar() {
                    self.device().$scalar_r_fw(self, &rhs)
                } else {
                    self.device().$fw(self, &rhs)
                }
            }
        }
        impl<'dev> $name<&Tensor<'dev>> for &Tensor<'dev> {
            type Output = Tensor<'dev>;
            fn $fn(self, rhs: &Tensor) -> Self::Output {
                if self.shape.is_scalar() {
                    self.device().$scalar_l_fw(rhs, self)
                } else if rhs.shape.is_scalar() {
                    self.device().$scalar_r_fw(self, rhs)
                } else {
                    self.device().$fw(self, rhs)
                }
            }
        }
    };
}

define_ab_op!(Add, add, add_fw, add_scalar_fw, add_scalar_fw);
define_ab_op!(Sub, sub, sub_fw, sub_scalar_l_fw, sub_scalar_r_fw);
define_ab_op!(Mul, mul, mul_fw, mul_scalar_fw, mul_scalar_fw);
define_ab_op!(Div, div, div_fw, div_scalar_l_fw, div_scalar_r_fw);

macro_rules! define_const_op {
    ( $name:ident, $fn:ident, $const_l_fw:ident, $const_r_fw:ident ) => {
        impl<'dev> $name<f32> for Tensor<'dev> {
            type Output = Tensor<'dev>;
            fn $fn(self, k: f32) -> Self::Output {
                self.device().$const_r_fw(&self, k)
            }
        }
        impl<'dev> $name<&f32> for Tensor<'dev> {
            type Output = Tensor<'dev>;
            fn $fn(self, k: &f32) -> Self::Output {
                self.device().$const_r_fw(&self, *k)
            }
        }
        impl<'dev> $name<f32> for &Tensor<'dev> {
            type Output = Tensor<'dev>;
            fn $fn(self, k: f32) -> Self::Output {
                self.device().$const_r_fw(self, k)
            }
        }
        impl<'dev> $name<&f32> for &Tensor<'dev> {
            type Output = Tensor<'dev>;
            fn $fn(self, k: &f32) -> Self::Output {
                self.device().$const_r_fw(self, *k)
            }
        }
        impl<'dev> $name<Tensor<'dev>> for f32 {
            type Output = Tensor<'dev>;
            fn $fn(self, other: Tensor<'dev>) -> Self::Output {
                other.device().$const_l_fw(&other, self)
            }
        }
        impl<'dev> $name<&Tensor<'dev>> for f32 {
            type Output = Tensor<'dev>;
            fn $fn(self, other: &Tensor<'dev>) -> Self::Output {
                other.device().$const_l_fw(other, self)
            }
        }
        impl<'dev> $name<Tensor<'dev>> for &f32 {
            type Output = Tensor<'dev>;
            fn $fn(self, other: Tensor<'dev>) -> Self::Output {
                other.device().$const_l_fw(&other, *self)
            }
        }
        impl<'dev> $name<&Tensor<'dev>> for &f32 {
            type Output = Tensor<'dev>;
            fn $fn(self, other: &Tensor<'dev>) -> Self::Output {
                other.device().$const_l_fw(other, *self)
            }
        }
    };
}

define_const_op!(Add, add, add_const_fw, add_const_fw);
define_const_op!(Sub, sub, sub_const_l_fw, sub_const_r_fw);
define_const_op!(Mul, mul, mul_const_fw, mul_const_fw);
define_const_op!(Div, div, div_const_l_fw, div_const_r_fw);

impl<'dev> ArithmeticFunctions<Tensor<'dev>> for Tensor<'dev> {}
impl<'dev> ArithmeticFunctions<Tensor<'dev>> for &Tensor<'dev> {}
