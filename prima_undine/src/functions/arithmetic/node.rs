use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Sub;

use crate::functions::ArithmeticFunctions;
use crate::{operators as op, Node};

impl<'arg, 'dev> Neg for Node<'arg, 'dev> {
    type Output = Node<'arg, 'dev>;
    fn neg(self) -> Self::Output {
        Node::create(op::Neg::new(self.device()), &[&self])
            .pop()
            .unwrap()
    }
}

impl<'arg, 'dev> Neg for &Node<'arg, 'dev> {
    type Output = Node<'arg, 'dev>;
    fn neg(self) -> Self::Output {
        Node::create(op::Neg::new(self.device()), &[self])
            .pop()
            .unwrap()
    }
}

macro_rules! define_ab_op {
    ( $name:ident, $fn:ident, $fw:ident, $scalar_l_fw:ident, $scalar_r_fw:ident ) => {
        impl<'arg, 'dev> $name<Node<'arg, 'dev>> for Node<'arg, 'dev> {
            type Output = Node<'arg, 'dev>;
            fn $fn(self, rhs: Node<'arg, 'dev>) -> Self::Output {
                if self.inner_value().shape.is_scalar() {
                    Node::create(op::$scalar_l_fw::new(self.device()), &[&rhs, &self])
                        .pop()
                        .unwrap()
                } else if rhs.inner_value().shape.is_scalar() {
                    Node::create(op::$scalar_r_fw::new(self.device()), &[&self, &rhs])
                        .pop()
                        .unwrap()
                } else {
                    Node::create(op::$fw::new(self.device()), &[&self, &rhs])
                        .pop()
                        .unwrap()
                }
            }
        }
        impl<'arg, 'dev> $name<&Node<'arg, 'dev>> for Node<'arg, 'dev> {
            type Output = Node<'arg, 'dev>;
            fn $fn(self, rhs: &Node<'arg, 'dev>) -> Self::Output {
                if self.inner_value().shape.is_scalar() {
                    Node::create(op::$scalar_l_fw::new(self.device()), &[rhs, &self])
                        .pop()
                        .unwrap()
                } else if rhs.inner_value().shape.is_scalar() {
                    Node::create(op::$scalar_r_fw::new(self.device()), &[&self, rhs])
                        .pop()
                        .unwrap()
                } else {
                    Node::create(op::$fw::new(self.device()), &[&self, rhs])
                        .pop()
                        .unwrap()
                }
            }
        }
        impl<'arg, 'dev> $name<Node<'arg, 'dev>> for &Node<'arg, 'dev> {
            type Output = Node<'arg, 'dev>;
            fn $fn(self, rhs: Node<'arg, 'dev>) -> Self::Output {
                if self.inner_value().shape.is_scalar() {
                    Node::create(op::$scalar_l_fw::new(self.device()), &[&rhs, self])
                        .pop()
                        .unwrap()
                } else if rhs.inner_value().shape.is_scalar() {
                    Node::create(op::$scalar_r_fw::new(self.device()), &[self, &rhs])
                        .pop()
                        .unwrap()
                } else {
                    Node::create(op::$fw::new(self.device()), &[self, &rhs])
                        .pop()
                        .unwrap()
                }
            }
        }
        impl<'arg, 'dev> $name<&Node<'arg, 'dev>> for &Node<'arg, 'dev> {
            type Output = Node<'arg, 'dev>;
            fn $fn(self, rhs: &Node<'arg, 'dev>) -> Self::Output {
                if self.inner_value().shape.is_scalar() {
                    Node::create(op::$scalar_l_fw::new(self.device()), &[rhs, self])
                        .pop()
                        .unwrap()
                } else if rhs.inner_value().shape.is_scalar() {
                    Node::create(op::$scalar_r_fw::new(self.device()), &[self, rhs])
                        .pop()
                        .unwrap()
                } else {
                    Node::create(op::$fw::new(self.device()), &[self, rhs])
                        .pop()
                        .unwrap()
                }
            }
        }
    };
}

define_ab_op!(Add, add, Add, AddScalar, AddScalar);
define_ab_op!(Sub, sub, Sub, SubScalarL, SubScalarR);
define_ab_op!(Mul, mul, Mul, MulScalar, MulScalar);
define_ab_op!(Div, div, Div, DivScalarL, DivScalarR);

macro_rules! define_const_op {
    ( $name:ident, $fn:ident, $const_l:ident, $const_r:ident ) => {
        impl<'arg, 'dev> $name<f32> for Node<'arg, 'dev> {
            type Output = Node<'arg, 'dev>;
            fn $fn(self, rhs: f32) -> Self::Output {
                Node::create(op::$const_r::new(self.device(), rhs), &[&self])
                    .pop()
                    .unwrap()
            }
        }
        impl<'arg, 'dev> $name<&f32> for Node<'arg, 'dev> {
            type Output = Node<'arg, 'dev>;
            fn $fn(self, rhs: &f32) -> Self::Output {
                Node::create(op::$const_r::new(self.device(), *rhs), &[&self])
                    .pop()
                    .unwrap()
            }
        }
        impl<'arg, 'dev> $name<f32> for &Node<'arg, 'dev> {
            type Output = Node<'arg, 'dev>;
            fn $fn(self, rhs: f32) -> Self::Output {
                Node::create(op::$const_r::new(self.device(), rhs), &[self])
                    .pop()
                    .unwrap()
            }
        }
        impl<'arg, 'dev> $name<&f32> for &Node<'arg, 'dev> {
            type Output = Node<'arg, 'dev>;
            fn $fn(self, rhs: &f32) -> Self::Output {
                Node::create(op::$const_r::new(self.device(), *rhs), &[self])
                    .pop()
                    .unwrap()
            }
        }
        impl<'arg, 'dev> $name<Node<'arg, 'dev>> for f32 {
            type Output = Node<'arg, 'dev>;
            fn $fn(self, rhs: Node<'arg, 'dev>) -> Self::Output {
                Node::create(op::$const_l::new(rhs.device(), self), &[&rhs])
                    .pop()
                    .unwrap()
            }
        }
        impl<'arg, 'dev> $name<&Node<'arg, 'dev>> for f32 {
            type Output = Node<'arg, 'dev>;
            fn $fn(self, rhs: &Node<'arg, 'dev>) -> Self::Output {
                Node::create(op::$const_l::new(rhs.device(), self), &[rhs])
                    .pop()
                    .unwrap()
            }
        }
        impl<'arg, 'dev> $name<Node<'arg, 'dev>> for &f32 {
            type Output = Node<'arg, 'dev>;
            fn $fn(self, rhs: Node<'arg, 'dev>) -> Self::Output {
                Node::create(op::$const_l::new(rhs.device(), *self), &[&rhs])
                    .pop()
                    .unwrap()
            }
        }
        impl<'arg, 'dev> $name<&Node<'arg, 'dev>> for &f32 {
            type Output = Node<'arg, 'dev>;
            fn $fn(self, rhs: &Node<'arg, 'dev>) -> Self::Output {
                Node::create(op::$const_l::new(rhs.device(), *self), &[rhs])
                    .pop()
                    .unwrap()
            }
        }
    };
}

define_const_op!(Add, add, AddConst, AddConst);
define_const_op!(Sub, sub, SubConstL, SubConstR);
define_const_op!(Mul, mul, MulConst, MulConst);
define_const_op!(Div, div, DivConstL, DivConstR);

impl<'arg, 'dev> ArithmeticFunctions<Node<'arg, 'dev>> for Node<'arg, 'dev> {}
impl<'arg, 'dev> ArithmeticFunctions<Node<'arg, 'dev>> for &Node<'arg, 'dev> {}
