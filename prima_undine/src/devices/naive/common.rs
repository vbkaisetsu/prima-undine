macro_rules! const_ptr {
    ( $tensor:expr) => {
        ($tensor.handle().load(std::sync::atomic::Ordering::Acquire) as *const Vec<f32>)
            .as_ref()
            .unwrap()
            .as_ptr()
    };
}

macro_rules! mut_ptr {
    ( $tensor:expr ) => {
        ($tensor.handle().load(std::sync::atomic::Ordering::Acquire) as *mut Vec<f32>)
            .as_mut()
            .unwrap()
            .as_mut_ptr()
    };
}

macro_rules! define_naive_fw_x_impl {
    ( $name:ident , $op:expr ) => {
        define_empty_impl!($name);
        impl crate::device_impl::FunctionFwImpl for $name {
            fn call(
                &self,
                xs: &[&crate::Tensor],
                _u32data: &[u32],
                _f32data: &[f32],
                ys: &mut [&mut crate::Tensor],
            ) {
                let x = xs[0];
                let y = &mut ys[0];
                let size = y.shape.size() as usize;
                unsafe {
                    let px = const_ptr!(x);
                    let py = mut_ptr!(y);
                    for i in 0..size {
                        *py.add(i) = $op(*px.add(i));
                    }
                }
            }
        }
    };
}

macro_rules! define_naive_bw_x_impl {
    ( $name:ident , $op:expr ) => {
        define_empty_impl!($name);
        impl crate::device_impl::FunctionBwImpl for $name {
            fn call(
                &self,
                xs: &[&crate::Tensor],
                ys: &[&crate::Tensor],
                gys: &[&crate::Tensor],
                _u32data: &[u32],
                _f32data: &[f32],
                gx: &mut crate::Tensor,
            ) {
                let x = xs[0];
                let y = ys[0];
                let gy = gys[0];
                let size = gy.shape.size() as usize;
                unsafe {
                    let px = const_ptr!(x);
                    let py = const_ptr!(y);
                    let pgy = const_ptr!(gy);
                    let pgx = mut_ptr!(gx);
                    for i in 0..size {
                        *pgx.add(i) += $op(*px.add(i), *py.add(i), *pgy.add(i));
                    }
                }
            }
        }
    };
}

macro_rules! define_naive_fw_ab_impl {
    ( $name:ident , $op:expr ) => {
        define_empty_impl!($name);
        impl crate::device_impl::FunctionFwImpl for $name {
            fn call(
                &self,
                xs: &[&Tensor],
                _u32data: &[u32],
                _f32data: &[f32],
                ys: &mut [&mut Tensor],
            ) {
                let a = xs[0];
                let b = xs[1];
                let y = &mut ys[0];
                let volume = y.shape.volume() as usize;
                let a_shift = if a.shape.batch() == 1 { 0 } else { volume };
                let b_shift = if b.shape.batch() == 1 { 0 } else { volume };
                unsafe {
                    let mut pa = const_ptr!(a);
                    let mut pb = const_ptr!(b);
                    let mut py = mut_ptr!(y);
                    for _ in 0..y.shape.batch() {
                        for i in 0..volume {
                            *py.add(i) = $op(*pa.add(i), *pb.add(i));
                        }
                        pa = pa.add(a_shift);
                        pb = pb.add(b_shift);
                        py = py.add(volume);
                    }
                }
            }
        }
    };
}

macro_rules! define_naive_fw_const_impl {
    ( $name:ident , $op:expr ) => {
        define_empty_impl!($name);
        impl crate::device_impl::FunctionFwImpl for $name {
            fn call(
                &self,
                xs: &[&crate::Tensor],
                _u32data: &[u32],
                f32data: &[f32],
                ys: &mut [&mut crate::Tensor],
            ) {
                let x = xs[0];
                let k = f32data[0];
                let y = &mut ys[0];
                let size = y.shape.size() as usize;
                unsafe {
                    let px = const_ptr!(x);
                    let py = mut_ptr!(y);
                    for i in 0..size {
                        *py.add(i) = $op(*px.add(i), k);
                    }
                }
            }
        }
    };
}

macro_rules! define_naive_bw_const_impl {
    ( $name:ident , $op:expr ) => {
        define_empty_impl!($name);
        impl crate::device_impl::FunctionBwImpl for $name {
            fn call(
                &self,
                xs: &[&crate::Tensor],
                ys: &[&crate::Tensor],
                gys: &[&crate::Tensor],
                _u32data: &[u32],
                f32data: &[f32],
                gx: &mut crate::Tensor,
            ) {
                let x = xs[0];
                let y = ys[0];
                let gy = gys[0];
                let k = f32data[0];
                let size = gy.shape.size() as usize;
                unsafe {
                    let px = const_ptr!(x);
                    let py = const_ptr!(y);
                    let pgy = const_ptr!(gy);
                    let pgx = mut_ptr!(gx);
                    for i in 0..size {
                        *pgx.add(i) += $op(*px.add(i), *py.add(i), *pgy.add(i), k);
                    }
                }
            }
        }
    };
}

macro_rules! define_naive_fw_scalar_impl {
    ( $name:ident , $op:expr ) => {
        define_empty_impl!($name);
        impl crate::device_impl::FunctionFwImpl for $name {
            fn call(
                &self,
                xs: &[&Tensor],
                _u32data: &[u32],
                _f32data: &[f32],
                ys: &mut [&mut Tensor],
            ) {
                let x = xs[0];
                let k = xs[1];
                let y = &mut ys[0];
                let volume = y.shape.volume() as usize;
                let x_shift = if x.shape.batch() == 1 { 0 } else { volume };
                let k_shift = if k.shape.batch() == 1 { 0 } else { 1 };
                unsafe {
                    let mut px = const_ptr!(x);
                    let mut pk = const_ptr!(k);
                    let mut py = mut_ptr!(y);
                    for _ in 0..y.shape.batch() {
                        for i in 0..volume {
                            *py.add(i) = $op(*px.add(i), *pk);
                        }
                        px = px.add(x_shift);
                        pk = pk.add(k_shift);
                        py = py.add(volume);
                    }
                }
            }
        }
    };
}
