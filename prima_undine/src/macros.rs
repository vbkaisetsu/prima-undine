#[macro_export]
macro_rules! shape {
    ( $( $dims:expr ),* ; $batch:expr ) => {
        $crate::Shape::new(&[$($dims),*], $batch)
    };
    ( $( $dims:expr ),* ) => {
        shape![$($dims),*; 1]
    };
    ( $( $dims:expr, )* ) => {
        shape![$($dims),*]
    };
}
