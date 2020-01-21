macro_rules! define_empty_impl {
    ( $name:ident ) => {
        pub struct $name {}
        impl $name {
            pub fn new() -> $name {
                $name {}
            }
        }
    };
}

pub mod naive;

pub use naive::Naive;
