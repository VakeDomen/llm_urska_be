use std::collections::HashMap;
use tokio::sync::Mutex;
use once_cell::sync::Lazy;

/// Global static map of WebSocket sockets, storing optional strings for each socket identified by a UUID string key.
pub static SOCKETS: Lazy<Mutex<HashMap<String, Option<String>>>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// Global static queue for managing socket IDs in the order they connect or are queued for some operation.
static QUEUE: Lazy<Mutex<Vec<String>>> = Lazy::new(|| Mutex::new(vec![]));

/// Increments the queue by adding a new socket ID to the end of the queue.
///
/// # Parameters
/// - `socket_id`: The unique identifier of a WebSocket connection to add to the queue.
pub async fn inc_que(socket_id: String) {
    let mut que = QUEUE.lock().await;
    que.push(socket_id);
}

/// Decrements the queue by removing a socket ID from the queue based on its current position.
///
/// # Parameters
/// - `socket_id`: The unique identifier of a WebSocket connection to remove from the queue.
pub async fn dec_que(socket_id: String) {
    let pos = que_pos(&socket_id).await;
    let mut que = QUEUE.lock().await;
    que.remove(pos);
}

/// Returns the current length of the queue.
///
/// # Returns
/// Returns the number of elements in the queue.
pub async fn que_len() -> usize {
    let que = QUEUE.lock().await;
    que.len()
}

/// Determines the position of a socket ID in the queue.
///
/// # Parameters
/// - `socket_id`: The unique identifier of a WebSocket connection whose position in the queue is needed.
///
/// # Returns
/// Returns the zero-based index of the socket ID in the queue.
///
/// # Panics
/// - Panics if the socket ID does not exist in the queue.
pub async fn que_pos(socket_id: &String) -> usize {
    let que = QUEUE.lock().await;
    que.iter().position(|x| *x == *socket_id).unwrap()
}
