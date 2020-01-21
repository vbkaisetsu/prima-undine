use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt;
use std::sync::atomic::AtomicPtr;
use std::sync::Arc;

use crate::device_impl::{
    DeviceImpl, FunctionBwImpl, FunctionFwF32Impl, FunctionFwImpl, FunctionFwU32Impl,
};
use crate::memory_pool::MemoryPool;
use crate::{Initializer, Parameter, Shape, Tensor};

pub struct Device<'dev>
where
    Self: Send + Sync,
{
    pub(crate) imp: Arc<Box<dyn DeviceImpl + 'dev>>,
    pub(crate) mem_pool: MemoryPool<'dev>,
    pub(crate) fw_impl: HashMap<String, Box<dyn FunctionFwImpl + 'dev>>,
    pub(crate) fw_u32_impl: HashMap<String, Box<dyn FunctionFwU32Impl + 'dev>>,
    pub(crate) fw_f32_impl: HashMap<String, Box<dyn FunctionFwF32Impl + 'dev>>,
    pub(crate) bw_impl: HashMap<String, Box<dyn FunctionBwImpl + 'dev>>,
}

impl<'dev> Device<'dev> {
    pub fn new<T: DeviceImpl + 'dev>(imp: T) -> Device<'dev> {
        let imp: Arc<Box<dyn DeviceImpl + 'dev>> = Arc::new(Box::new(imp));
        Device {
            imp: Arc::clone(&imp),
            mem_pool: MemoryPool::new(imp),
            fw_impl: HashMap::new(),
            bw_impl: HashMap::new(),
            fw_u32_impl: HashMap::new(),
            fw_f32_impl: HashMap::new(),
        }
    }

    pub fn register_fw_impl<T: FunctionFwImpl + 'dev>(&mut self, name: &str, func: T) {
        self.fw_impl.insert(name.to_string(), Box::new(func));
    }

    pub fn register_fw_u32_impl<T: FunctionFwU32Impl + 'dev>(&mut self, name: &str, func: T) {
        self.fw_u32_impl.insert(name.to_string(), Box::new(func));
    }

    pub fn register_fw_f32_impl<T: FunctionFwF32Impl + 'dev>(&mut self, name: &str, func: T) {
        self.fw_f32_impl.insert(name.to_string(), Box::new(func));
    }

    pub fn register_bw_impl<T: FunctionBwImpl + 'dev>(&mut self, name: &str, func: T) {
        self.bw_impl.insert(name.to_string(), Box::new(func));
    }

    pub fn call_fw_impl(
        &self,
        name: &str,
        xs: &[&Tensor],
        u32data: &[u32],
        f32data: &[f32],
        y: &mut [&mut Tensor],
    ) {
        if let Some(fw) = self.fw_impl.get(name) {
            fw.call(xs, u32data, f32data, y);
        } else {
            panic!("{} is not implemented", name);
        }
    }

    pub fn call_fw_u32_impl(
        &self,
        name: &str,
        xs: &[&Tensor],
        u32data: &[u32],
        f32data: &[f32],
        y: &mut [u32],
    ) {
        if let Some(fw) = self.fw_u32_impl.get(name) {
            fw.call(xs, u32data, f32data, y);
        } else {
            panic!("{} is not implemented", name);
        }
    }

    pub fn call_fw_f32_impl(
        &self,
        name: &str,
        xs: &[&Tensor],
        u32data: &[u32],
        f32data: &[f32],
        y: &mut [f32],
    ) {
        if let Some(fw) = self.fw_f32_impl.get(name) {
            fw.call(xs, u32data, f32data, y);
        } else {
            panic!("{} is not implemented", name);
        }
    }

    pub fn call_bw_impl(
        &self,
        name: &str,
        xs: &[&Tensor],
        ys: &[&Tensor],
        gys: &[&Tensor],
        u32data: &[u32],
        f32data: &[f32],
        gx: &mut Tensor,
    ) {
        if let Some(bw) = self.bw_impl.get(name) {
            bw.call(xs, ys, gys, u32data, f32data, gx);
        } else {
            panic!("{} is not implemented", name);
        }
    }

    pub fn identifier(&self) -> String {
        self.imp.identifier()
    }

    pub fn new_handle(&self, size: u32) -> AtomicPtr<c_void> {
        self.mem_pool.new_handle(size)
    }

    pub fn drop_handle(&self, handle: &AtomicPtr<c_void>) {
        self.mem_pool.drop_handle(handle);
    }

    pub fn new_tensor(&'dev self, shape: Shape) -> Tensor<'dev> {
        Tensor::new(self, shape)
    }

    pub fn new_tensor_by_constant(&'dev self, shape: Shape, k: f32) -> Tensor<'dev> {
        let mut tensor = self.new_tensor(shape);
        tensor.alloc();
        tensor.reset(k);
        tensor
    }

    pub fn new_tensor_by_slice(&'dev self, shape: Shape, values: &[f32]) -> Tensor<'dev> {
        let mut tensor = self.new_tensor(shape);
        tensor.alloc();
        self.reset_tensor_by_slice(&mut tensor, values);
        tensor
    }

    pub fn copy_tensor(&'dev self, source: &Tensor) -> Tensor<'dev> {
        let mut tensor = self.new_tensor(source.shape);
        tensor.alloc();
        self.reset_tensor_by_tensor(&mut tensor, source);
        tensor
    }

    pub fn new_parameter(&self, shape: Shape, initializer: &dyn Initializer) -> Parameter {
        let mut value = self.new_tensor(shape);
        let mut gradient = self.new_tensor(shape);
        initializer.apply(&mut value);
        gradient.alloc();
        gradient.reset(0.);
        Parameter::new(value, gradient)
    }

    pub fn reset_tensor(&self, x: &mut Tensor, k: f32) {
        self.call_fw_impl("reset_tensor_impl", &[], &[], &[k], &mut [x]);
    }

    pub fn reset_tensor_by_slice(&self, x: &mut Tensor, values: &[f32]) {
        assert!(x.shape.size() as usize == values.len());
        self.call_fw_impl("reset_tensor_by_slice_impl", &[], &[], values, &mut [x]);
    }

    pub fn reset_tensor_by_tensor(&self, x: &mut Tensor, source: &Tensor) {
        assert!(x.valid());
        assert!(source.valid());
        assert!(x.shape.size() == source.shape.size());
        self.call_fw_impl("reset_tensor_by_tensor_impl", &[source], &[], &[], &mut [x]);
    }

    pub fn tensor_to_vector(&self, x: &Tensor) -> Vec<f32> {
        let mut ret = vec![0.; x.shape.size() as usize];
        self.call_fw_f32_impl("tensor_to_vector_impl", &[x], &[], &[], ret.as_mut_slice());
        ret
    }

    pub fn tensor_to_float(&self, x: &Tensor) -> f32 {
        assert!(x.shape.size() == 1);
        let mut ret = [0.];
        self.call_fw_f32_impl("tensor_to_vector_impl", &[x], &[], &[], &mut ret);
        ret[0]
    }
}

impl<'dev> fmt::Debug for Device<'dev> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Device {{}}")
    }
}

impl<'dev> PartialEq for &Device<'dev> {
    fn eq(&self, other: &Self) -> bool {
        *self as *const Device == *other as *const Device
    }
}
