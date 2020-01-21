use std::cmp;
use std::fmt;
use std::ops::Index;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

const MAX_DEPTH: u32 = 8;

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    dims: [u32; MAX_DEPTH as usize],
    batch: u32,
    depth: u32,
    volume: u32,
}

impl Shape {
    pub fn new(dims: &[u32], batch: u32) -> Shape {
        assert!(dims.len() <= MAX_DEPTH as usize);
        assert!(batch >= 1);
        let mut dims_filled = [1; MAX_DEPTH as usize];
        let mut volume = 1;
        let mut depth = 0;
        for i in 0..dims.len() {
            assert!(dims[i] != 0);
            dims_filled[i] = dims[i];
            if dims[i] != 1 {
                depth = i as u32 + 1;
            }
            volume *= dims[i];
        }
        Shape {
            dims: dims_filled,
            batch: batch,
            depth: depth,
            volume: volume,
        }
    }

    pub fn depth(&self) -> u32 {
        self.depth
    }

    pub fn dims(&self) -> &[u32] {
        &self.dims[..self.depth as usize]
    }

    pub fn batch(&self) -> u32 {
        self.batch
    }

    pub fn volume(&self) -> u32 {
        self.volume
    }

    pub fn lower_volume(&self, dim: u32) -> u32 {
        let mut ret = 1;
        for i in 0..cmp::min(dim, self.depth) as usize {
            ret *= self.dims[i];
        }
        ret
    }

    pub fn size(&self) -> u32 {
        self.batch * self.volume()
    }

    pub fn has_batch(&self) -> bool {
        self.batch != 1
    }

    pub fn has_compatible_batch(&self, rhs: Shape) -> bool {
        self.batch == rhs.batch || self.batch == 1 || rhs.batch == 1
    }

    pub fn is_scalar(&self) -> bool {
        self.depth == 0
    }

    pub fn is_column_vector(&self) -> bool {
        self.depth <= 1
    }

    pub fn is_matrix(&self) -> bool {
        self.depth <= 2
    }

    pub fn has_same_dims(&self, rhs: Shape) -> bool {
        let mut ok = true;
        for i in 0..self.depth as usize {
            ok = ok && self.dims[i] == rhs.dims[i];
        }
        ok && self.depth == rhs.depth
    }

    pub fn has_same_loo_dims(&self, rhs: Shape, dim: u32) -> bool {
        let mut nl = if self.depth == dim + 1 {
            dim
        } else {
            self.depth
        } as usize;
        while nl > 0 && self.dims[nl - 1] == 1 {
            nl -= 1;
        }
        let mut nr = if rhs.depth == dim + 1 { dim } else { rhs.depth } as usize;
        while nr > 0 && rhs.dims[nr - 1] == 1 {
            nr -= 1;
        }
        let mut p = nl == nr;
        for i in 0..nl {
            p = p && (self.dims[i] == rhs.dims[i] || i == dim as usize);
        }
        p
    }

    pub fn resize_dim(&self, dim: u32, m: u32) -> Shape {
        let mut ret = *self;
        ret.update_dim(dim, m);
        ret
    }

    pub fn resize_batch(&self, batch: u32) -> Shape {
        let mut ret = *self;
        ret.update_batch(batch);
        ret
    }

    pub fn update_dim(&mut self, dim: u32, m: u32) {
        assert!(dim < MAX_DEPTH);
        assert!(m != 0);
        if dim >= self.depth {
            let new_depth = dim + 1;
            for i in self.depth..new_depth {
                self.dims[i as usize] = 1;
            }
            self.depth = new_depth;
        }
        self.volume = self.volume / self.dims[dim as usize] * m;
        self.dims[dim as usize] = m;
        while self.depth > 0 && self.dims[self.depth as usize - 1] == 1 {
            self.depth -= 1;
        }
    }

    pub fn update_batch(&mut self, batch: u32) {
        assert!(batch >= 1);
        self.batch = batch;
    }
}

impl Index<u32> for Shape {
    type Output = u32;
    fn index(&self, index: u32) -> &Self::Output {
        if index < self.depth() {
            &self.dims[index as usize]
        } else {
            &1
        }
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shape {{ {:?}, {} }}", &self.dims(), self.batch)
    }
}

#[derive(Serialize, Deserialize)]
struct ShapeSerde {
    dims: Vec<u32>,
    batch: u32,
}

impl<'de> Deserialize<'de> for Shape {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s: ShapeSerde = Deserialize::deserialize(deserializer)?;
        Ok(Shape::new(&s.dims, s.batch))
    }
}

impl Serialize for Shape {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut s = serializer.serialize_struct("Shape", 2)?;
        s.serialize_field("dims", self.dims())?;
        s.serialize_field("batch", &self.batch())?;
        s.end()
    }
}
