use std::cmp;

use crate::Shape;

pub fn scalar_op(x: Shape, k: Shape) -> Shape {
    assert!(k.is_scalar() && x.has_compatible_batch(k));
    x.resize_batch(cmp::max(x.batch(), k.batch()))
}

pub fn elementwise(a: Shape, b: Shape) -> Shape {
    assert!(a.has_same_dims(b) && a.has_compatible_batch(b));
    a.resize_batch(cmp::max(a.batch(), b.batch()))
}

pub fn broadcast(x: Shape, dim: u32, size: u32) -> Shape {
    assert!(x[dim] == 1 && size != 0);
    x.resize_dim(dim, size)
}

pub fn matmul(l: Shape, r: Shape) -> Shape {
    assert!(l.is_matrix() && r.is_matrix() && l[1] == r[0] && l.has_compatible_batch(r));
    Shape::new(&[l[0], r[1]], cmp::max(l.batch(), r.batch()))
}

pub fn transpose(x: Shape) -> Shape {
    assert!(x.is_matrix());
    Shape::new(&[x[1], x[0]], x.batch())
}

pub fn slice(x: Shape, dim: u32, lower: u32, upper: u32) -> Shape {
    assert!(lower < upper);
    assert!(upper <= x[dim]);
    x.resize_dim(dim, upper - lower)
}

pub fn pick(x: Shape, ids: &[u32], dim: u32) -> Shape {
    let n = x[dim];
    let bi = ids.len() as u32;
    assert!(bi != 0 && (x.batch() == bi || !x.has_batch() || bi == 1));
    for i in 0..bi as usize {
        assert!(ids[i] < n);
    }
    x.resize_dim(dim, 1).resize_batch(cmp::max(x.batch(), bi))
}

pub fn concat(xs: &[Shape], dim: u32) -> Shape {
    assert!(xs.len() >= 1);
    let mut s0 = xs[0];
    let mut sum = s0[dim];
    for i in 1..xs.len() {
        let s = xs[i];
        assert!(s0.has_same_loo_dims(s, dim) && s0.has_compatible_batch(s));
        if !s0.has_batch() {
            s0.update_batch(s.batch());
        }
        sum += s[dim];
    }
    s0.resize_dim(dim, sum)
}

pub fn batch_pick(x: Shape, ids: &[u32]) -> Shape {
    let n = x.batch();
    let bi = ids.len() as u32;
    assert!(bi != 0);
    for &id in ids {
        assert!(id < n);
    }
    x.resize_batch(bi)
}

pub fn batch_slice(x: Shape, lower: u32, upper: u32) -> Shape {
    assert!(lower < upper && upper <= x.batch());
    x.resize_batch(upper - lower)
}

pub fn batch_concat(xs: &[Shape]) -> Shape {
    assert!(xs.len() != 0);
    let s0 = xs[0];
    let mut sum = s0.batch();
    for i in 1..xs.len() {
        let s = xs[i];
        assert!(s0.has_same_dims(s));
        sum += s.batch();
    }
    s0.resize_batch(sum)
}
