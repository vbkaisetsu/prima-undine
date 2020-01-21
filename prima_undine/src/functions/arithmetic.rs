pub mod device;
mod node;
mod tensor;

use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Sub;

pub trait ArithmeticFunctions<T>:
    Add<T, Output = T>
    + for<'a> Add<&'a T, Output = T>
    + Add<f32, Output = T>
    + for<'a> Add<&'a f32, Output = T>
    + Sub<T, Output = T>
    + for<'a> Sub<&'a T, Output = T>
    + Sub<f32, Output = T>
    + for<'a> Sub<&'a f32, Output = T>
    + Mul<T, Output = T>
    + for<'a> Mul<&'a T, Output = T>
    + Mul<f32, Output = T>
    + for<'a> Mul<&'a f32, Output = T>
    + Div<T, Output = T>
    + for<'a> Div<&'a T, Output = T>
    + Div<f32, Output = T>
    + for<'a> Div<&'a f32, Output = T>
    + Neg
{
}
