use std::collections::HashMap;
use std::sync::Mutex;
use once_cell::sync::Lazy;

pub static SOCKETS: Lazy<Mutex<HashMap<String, Option<String>>>> = Lazy::new(|| Mutex::new(HashMap::new()));

