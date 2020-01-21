macro_rules! extract_parameters {
    () => { "".to_string() };
    ( $fname:ident, $fval:expr $(, $name:ident, $val:expr)* ) => {
        "(".to_string() + stringify!($fname) + "=" +  &$fval.to_string() $( + "," + stringilfy!($name) + "=" + &$val.to_string() )* + ")"
    };
}

macro_rules! define_operator_struct {
    ($name:ident, $($param:ident, $type:ty),*) => {
        pub struct $name<'dev> {
            device: &'dev crate::Device<'dev>,
            $($param: $type,)*
        }
        impl<'dev> $name<'dev> {
            pub fn new(device: &'dev crate::Device $(, $param: $type)* ) -> $name<'dev> {
                $name {
                    device: device,
                    $($param: $param,)*
                }
            }
        }
    };
    ($name:ident) => { define_operator_struct!($name,); };
}

macro_rules! define_operator_ab {
    ($name:ident, $fw:ident, $bwfunc:expr) => {
        define_operator_struct!($name);
        impl<'arg, 'dev> crate::Operator<'arg, 'dev> for $name<'dev> {
            fn name(&self) -> String {
                stringify!($name).to_string()
            }
            fn device(&self) -> &'dev crate::Device<'dev> {
                self.device
            }
            fn forward_shape(&self, x: &[crate::Shape]) -> Vec<crate::Shape> {
                vec![crate::shape_ops::elementwise(x[0], x[1])]
            }
            fn forward(&self, x: &[&crate::Tensor], y: &mut [&mut crate::Tensor<'arg>]) {
                y[0].replace(self.device.$fw(x[0], x[1]));
            }
            fn backward(
                &self,
                x: &[&crate::Tensor],
                y: &[&crate::Tensor],
                gy: &[&crate::Tensor],
                gx: &[&std::cell::RefCell<crate::Tensor>],
            ) {
                $bwfunc(self.device, x, y, gy, gx);
            }
        }
    };
    ($name:ident) => {
        define_operator_ab!($name,);
    };
}

macro_rules! define_operator_x {
    ($name:ident, $fw:ident, $bw:ident, $($param:ident, $type:ty),*) => {
        define_operator_struct!($name $(, $param, $type)* );
        impl<'arg, 'dev> crate::Operator<'arg, 'dev> for $name<'dev> {
            fn name(&self) -> String {
                stringify!($name).to_string() + &extract_parameters!( $( $param, self.$param ),* )
            }
            fn device(&self) -> &'dev crate::Device<'dev> {
                self.device
            }
            fn forward_shape(&self, x: &[crate::Shape]) -> Vec<crate::Shape> {
                vec![x[0]]
            }
            fn forward(&self, x: &[&crate::Tensor], y: &mut [&mut crate::Tensor<'arg>]) {
                y[0].replace(self.device.$fw(x[0], $(self.$param,)*));
            }
            fn backward(&self, x: &[&crate::Tensor], y: &[&crate::Tensor], gy: &[&crate::Tensor], gx: &[&std::cell::RefCell<crate::Tensor>]) {
                self.device.$bw(x[0], y[0], gy[0], $(self.$param,)* &mut *gx[0].borrow_mut());
            }
        }
    };
    ($name:ident, $fw:ident, $bw:ident) => { define_operator_x!($name, $fw, $bw,); };
}
