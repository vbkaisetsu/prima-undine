use std::slice;
use std::sync::{Arc, Mutex};

use crate::device_impl::FunctionFwImpl;
use crate::Randomizer;
use crate::Tensor;

pub struct RandomBernoulliImpl {
    randomizer: Arc<Mutex<Box<dyn Randomizer>>>,
}

impl RandomBernoulliImpl {
    pub fn new(randomizer: Arc<Mutex<Box<dyn Randomizer>>>) -> RandomBernoulliImpl {
        RandomBernoulliImpl {
            randomizer: randomizer,
        }
    }
}

impl FunctionFwImpl for RandomBernoulliImpl {
    fn call(&self, _xs: &[&Tensor], _u32data: &[u32], f32data: &[f32], ys: &mut [&mut Tensor]) {
        let p = f32data[0];
        let y = &mut ys[0];
        unsafe {
            let py = mut_ptr!(y);
            let mut yslice = slice::from_raw_parts_mut(py, y.shape.size() as usize);
            self.randomizer
                .lock()
                .unwrap()
                .fill_bernoulli(p, &mut yslice)
        }
    }
}

pub struct RandomNormalImpl {
    randomizer: Arc<Mutex<Box<dyn Randomizer>>>,
}

impl RandomNormalImpl {
    pub fn new(randomizer: Arc<Mutex<Box<dyn Randomizer>>>) -> RandomNormalImpl {
        RandomNormalImpl {
            randomizer: randomizer,
        }
    }
}

impl FunctionFwImpl for RandomNormalImpl {
    fn call(&self, _xs: &[&Tensor], _u32data: &[u32], f32data: &[f32], ys: &mut [&mut Tensor]) {
        let mean = f32data[0];
        let sd = f32data[1];
        let y = &mut ys[0];
        unsafe {
            let py = mut_ptr!(y);
            let mut yslice = slice::from_raw_parts_mut(py, y.shape.size() as usize);
            self.randomizer
                .lock()
                .unwrap()
                .fill_normal(mean, sd, &mut yslice)
        }
    }
}

pub struct RandomUniformImpl {
    randomizer: Arc<Mutex<Box<dyn Randomizer>>>,
}

impl RandomUniformImpl {
    pub fn new(randomizer: Arc<Mutex<Box<dyn Randomizer>>>) -> RandomUniformImpl {
        RandomUniformImpl {
            randomizer: randomizer,
        }
    }
}

impl FunctionFwImpl for RandomUniformImpl {
    fn call(&self, _xs: &[&Tensor], _u32data: &[u32], f32data: &[f32], ys: &mut [&mut Tensor]) {
        let lower = f32data[0];
        let upper = f32data[1];
        let y = &mut ys[0];
        unsafe {
            let py = mut_ptr!(y);
            let mut yslice = slice::from_raw_parts_mut(py, y.shape.size() as usize);
            self.randomizer
                .lock()
                .unwrap()
                .fill_uniform(lower, upper, &mut yslice)
        }
    }
}
