use std::any;

fn get_type_name<T>(_: &T) -> String {
    format!("{}", any::type_name::<T>())
}