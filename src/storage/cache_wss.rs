use std::collections::HashMap;
use tokio::sync::Mutex;
use once_cell::sync::Lazy;

pub static SOCKETS: Lazy<Mutex<HashMap<String, Option<String>>>> = Lazy::new(|| Mutex::new(HashMap::new()));
static QUEUE: Lazy<Mutex<Vec<String>>> = Lazy::new(|| Mutex::new(vec![]));


pub async fn inc_que(socket_id: String) {
    let mut que = QUEUE.lock().await;
    que.push(socket_id);
}


pub async fn dec_que(socket_id: String) {
    let pos = que_pos(socket_id).await;
    let mut que = QUEUE.lock().await;
    que.remove(pos);
}

pub async fn que_len() -> usize {
    let que = QUEUE.lock().await;
    que.len()
}

pub async fn que_pos(socket_id: String) -> usize {
    let que = QUEUE.lock().await;
    que.iter().position(|x| *x == socket_id).unwrap()
}
