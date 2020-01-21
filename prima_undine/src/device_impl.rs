use std::ffi::c_void;
use std::sync::atomic::AtomicPtr;

use crate::Tensor;

pub trait DeviceImpl: Sync + Send {
    fn identifier(&self) -> String;
    fn new_handle(&self, size: u32) -> AtomicPtr<c_void>;
    fn drop_handle(&self, handle: &AtomicPtr<c_void>);
}

pub trait FunctionFwImpl: Sync + Send {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], f32data: &[f32], ys: &mut [&mut Tensor]);
}

pub trait FunctionFwU32Impl: Sync + Send {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], f32data: &[f32], ys: &mut [u32]);
}

pub trait FunctionFwF32Impl: Sync + Send {
    fn call(&self, xs: &[&Tensor], u32data: &[u32], f32data: &[f32], ys: &mut [f32]);
}

pub trait FunctionBwImpl: Sync + Send {
    fn call(
        &self,
        xs: &[&Tensor],
        ys: &[&Tensor],
        gys: &[&Tensor],
        u32data: &[u32],
        f32data: &[f32],
        gx: &mut Tensor,
    );
}
