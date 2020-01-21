use std::cmp;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{Arc, Mutex};

use crate::device_impl::DeviceImpl;

const MAX_BITS: usize = 32;

fn calculate_shifts(x: u32) -> u32 {
    assert!(x != 0);

    // Flips all bits at the right of leftmost-1 to 1.
    let mut b = x | (x >> 16);
    b |= b >> 8;
    b |= b >> 4;
    b |= b >> 2;
    b |= b >> 1;

    // Counts the number of 1.
    let b = (b & 0x55555555u32) + ((b >> 1) & 0x55555555u32);
    let b = (b & 0x33333333u32) + ((b >> 2) & 0x33333333u32);
    let b = (b & 0x0f0f0f0fu32) + ((b >> 4) & 0x0f0f0f0fu32);
    let b = (b & 0x00ff00ffu32) + ((b >> 8) & 0x00ff00ffu32);
    let b = (b & 0x0000ffffu32) + ((b >> 16) & 0x0000ffffu32);

    // Adjusts the result.
    return b - (1u32 << (b - 1) == x) as u32;
}

pub struct MemoryPool<'dev>
where
    Self: Send + Sync,
{
    imp: Arc<Box<dyn DeviceImpl + 'dev>>,
    pool: Mutex<Vec<Vec<AtomicPtr<c_void>>>>,
    mem_shift: Mutex<HashMap<usize, usize>>,
}

impl<'dev> MemoryPool<'dev> {
    pub fn new(imp: Arc<Box<dyn DeviceImpl + 'dev>>) -> Self {
        Self {
            imp: imp,
            pool: Mutex::new((0..MAX_BITS).map(|_| vec![]).collect()),
            mem_shift: Mutex::new(HashMap::new()),
        }
    }

    pub fn new_handle(&self, size: u32) -> AtomicPtr<c_void> {
        assert!(size != 0);
        const MINIMUM_SIZE: u32 = 64;
        let size = cmp::max(size, MINIMUM_SIZE);
        let shift = calculate_shifts(size) as usize;
        assert!(shift <= MAX_BITS - 1);
        let mem_size = 1 << shift;
        let handle = if let Some(handle) = self.pool.lock().unwrap()[shift].pop() {
            handle
        } else {
            self.imp.new_handle(mem_size)
        };
        self.mem_shift
            .lock()
            .unwrap()
            .insert(handle.load(Ordering::Acquire) as usize, shift);
        handle
    }

    pub fn drop_handle(&self, handle: &AtomicPtr<c_void>) {
        let shift = self
            .mem_shift
            .lock()
            .unwrap()
            .remove(&(handle.load(Ordering::Acquire) as usize))
            .unwrap();
        self.pool.lock().unwrap()[shift].push(AtomicPtr::new(handle.load(Ordering::Acquire)));
    }
}

impl<'dev> Drop for MemoryPool<'dev> {
    fn drop(&mut self) {
        for v in self.pool.lock().unwrap().iter() {
            for handle in v {
                self.imp.drop_handle(handle);
            }
        }
    }
}
