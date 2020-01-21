macro_rules! define_fw_ab {
    ( $f:ident, $f_impl:expr, $sop:ident ) => {
        fn $f(&self, a: &Tensor, b: &Tensor) -> Tensor {
            assert!(a.device() == self);
            assert!(b.device() == self);
            let mut y = self.new_tensor(shape_ops::$sop(a.shape, b.shape));
            y.alloc();
            self.call_fw_impl($f_impl, &[a, b], &[], &[], &mut [&mut y]);
            y
        }
    };
}

macro_rules! define_bw_ab_a {
    ( $f:ident, $f_impl:expr, $sop:ident ) => {
        fn $f(&self, a: &Tensor, b: &Tensor, y: &Tensor, gy: &Tensor, ga: &mut Tensor) {
            assert!(a.device() == self);
            assert!(b.device() == self);
            assert!(y.device() == self);
            assert!(gy.device() == self);
            assert!(ga.device() == self);
            assert!(a.shape == ga.shape);
            assert!(y.shape == gy.shape);
            assert!(y.shape == shape_ops::$sop(a.shape, b.shape));
            self.call_bw_impl($f_impl, &[a, b], &[y], &[gy], &[], &[], ga);
        }
    };
}

macro_rules! define_bw_ab_b {
    ( $f:ident , $f_impl:expr , $sop:ident ) => {
        fn $f(&self, a: &Tensor, b: &Tensor, y: &Tensor, gy: &Tensor, gb: &mut Tensor) {
            assert!(a.device() == self);
            assert!(b.device() == self);
            assert!(y.device() == self);
            assert!(gy.device() == self);
            assert!(gb.device() == self);
            assert!(b.shape == gb.shape);
            assert!(y.shape == gy.shape);
            assert!(y.shape == shape_ops::$sop(a.shape, b.shape));
            self.call_bw_impl($f_impl, &[a, b], &[y], &[gy], &[], &[], gb);
        }
    };
}

macro_rules! define_fw_const {
    ( $f:ident , $f_impl:expr ) => {
        fn $f(&self, x: &Tensor, k: f32) -> Tensor {
            assert!(x.device() == self);
            let mut y = self.new_tensor(x.shape);
            y.alloc();
            self.call_fw_impl($f_impl, &[x], &[], &[k], &mut [&mut y]);
            y
        }
    };
}

macro_rules! define_bw_const {
    ( $f:ident , $f_impl:expr ) => {
        fn $f(&self, x: &Tensor, y: &Tensor, gy: &Tensor, k: f32, gx: &mut Tensor) {
            assert!(x.device() == self);
            assert!(y.device() == self);
            assert!(gy.device() == self);
            assert!(gx.device() == self);
            assert!(x.shape == gx.shape);
            assert!(y.shape == gy.shape);
            assert!(x.shape == y.shape);
            self.call_bw_impl($f_impl, &[x], &[y], &[gy], &[], &[k], gx);
        }
    };
}

macro_rules! define_fw_x {
    ( $f:ident , $f_impl:expr ) => {
        fn $f(&self, x: &Tensor) -> Tensor {
            assert!(x.device() == self);
            let mut y = self.new_tensor(x.shape);
            y.alloc();
            self.call_fw_impl($f_impl, &[x], &[], &[], &mut [&mut y]);
            y
        }
    };
}

macro_rules! define_bw_x {
    ( $f:ident , $f_impl:expr ) => {
        fn $f(&self, x: &Tensor, y: &Tensor, gy: &Tensor, gx: &mut Tensor) {
            assert!(x.device() == self);
            assert!(y.device() == self);
            assert!(gy.device() == self);
            assert!(gx.device() == self);
            assert!(x.shape == gx.shape);
            assert!(y.shape == gy.shape);
            assert!(x.shape == y.shape);
            self.call_bw_impl($f_impl, &[x], &[y], &[gy], &[], &[], gx);
        }
    };
}

mod arithmetic;
mod basic;
mod random;

pub use self::arithmetic::device::ArithmeticDeviceFunctions;
pub use self::arithmetic::ArithmeticFunctions;
pub use self::basic::device::BasicDeviceFunctions;
pub use self::basic::BasicFunctions;
pub use self::random::device::RandomDeviceFunctions;
