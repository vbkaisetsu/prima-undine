use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicPtr, Ordering};

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{Device, Parameter, Shape};

pub struct Tensor<'dev>
where
    Self: Send + Sync,
{
    device: Option<&'dev Device<'dev>>,
    pub(crate) shape: Shape,
    handle: AtomicPtr<c_void>,
    host_values: Option<Vec<f32>>,
    reference: Option<&'dev Tensor<'dev>>,
}

impl<'dev> Tensor<'dev> {
    pub fn new(device: &'dev Device<'dev>, shape: Shape) -> Tensor<'dev> {
        Tensor {
            device: Some(device),
            shape: shape,
            handle: AtomicPtr::new(ptr::null_mut()),
            host_values: None,
            reference: None,
        }
    }

    pub fn new_host_tensor(shape: Shape, values: Vec<f32>) -> Tensor<'dev> {
        Tensor {
            device: None,
            shape: shape,
            handle: AtomicPtr::new(ptr::null_mut()),
            host_values: Some(values),
            reference: None,
        }
    }

    pub fn move_to_device(&mut self, device: &'dev Device<'dev>) {
        assert!(self.host_values.is_some());
        self.device = Some(device);
        self.handle = self.device.unwrap().new_handle(self.shape.size());
        let values = self.host_values.take().unwrap();
        device.reset_tensor_by_slice(self, &values);
    }

    pub fn alloc(&mut self) {
        assert!(!self.valid());
        assert!(self.host_values.is_none());
        self.handle = self.device.unwrap().new_handle(self.shape.size());
    }

    pub fn valid(&self) -> bool {
        !self.handle.load(Ordering::Acquire).is_null()
    }

    pub fn refer(&mut self, other: &'dev Tensor<'dev>) {
        assert_eq!(self.device, other.device);
        assert!(!self.valid());
        assert_eq!(self.shape, other.shape);
        assert!(self.host_values.is_none());
        self.handle
            .store(other.handle.load(Ordering::Acquire), Ordering::Release);
        self.reference = Some(other);
    }

    pub fn replace(&mut self, other: Tensor) {
        assert_eq!(self.device, other.device);
        assert!(!self.valid());
        assert_eq!(self.shape, other.shape);
        assert!(self.host_values.is_none());
        self.handle
            .store(other.handle.load(Ordering::Acquire), Ordering::Release);
        other.handle.store(ptr::null_mut(), Ordering::Release);
    }

    pub fn device(&self) -> &'dev Device<'dev> {
        self.device.unwrap()
    }

    pub fn handle(&self) -> &AtomicPtr<c_void> {
        &self.handle
    }

    pub fn reset(&mut self, k: f32) {
        self.device().reset_tensor(self, k);
    }

    pub fn update_shape(&mut self, shape: Shape) {
        assert!(shape.size() == self.shape.size());
        self.shape = shape;
    }
}

impl<'dev> Drop for Tensor<'dev> {
    fn drop(&mut self) {
        if !self.handle.load(Ordering::Acquire).is_null() && self.reference.is_none() {
            self.device().drop_handle(&self.handle);
        }
    }
}

#[derive(Deserialize)]
struct TensorSerde {
    shape: Shape,
    values: Vec<f32>,
}

impl<'de, 'dev> Deserialize<'de> for Tensor<'dev> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let t: TensorSerde = Deserialize::deserialize(deserializer)?;
        Ok(Tensor::new_host_tensor(t.shape, t.values))
    }
}

impl<'dev> Serialize for Tensor<'dev> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut s = serializer.serialize_struct("Tensor", 2)?;
        s.serialize_field("shape", &self.shape)?;
        s.serialize_field("values", &self.device().tensor_to_vector(self))?;
        s.end()
    }
}

impl<'arg, 'dev> From<&'arg Tensor<'dev>> for Tensor<'arg>
where
    'dev: 'arg,
{
    fn from(item: &'arg Tensor<'dev>) -> Self {
        let mut t = item.device().new_tensor(item.shape);
        t.refer(item);
        t
    }
}

impl<'arg, 'dev> From<&'arg mut Parameter<'dev>> for Tensor<'dev>
where
    'dev: 'arg,
{
    fn from(item: &'arg mut Parameter<'dev>) -> Self {
        let dev = item.device();
        dev.copy_tensor(&item.value)
    }
}
