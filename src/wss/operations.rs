use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio_tungstenite::{tungstenite::Message, WebSocketStream};
use log::error;
use anyhow::Result;
use crate::storage::cache_wss::SOCKETS;

use super::message::WSSMessage;

/// Retrieves the name associated with a given socket ID from the global socket data.
///
/// # Parameters
/// - `socket_id`: A reference to the string that identifies the socket.
///
/// # Returns
/// Returns an optional string containing the name associated with the socket ID. Returns `None` if no name is associated with the given ID.
pub async fn get_socket_name(socket_id: &String) -> Option<String> {
    let socket_data = SOCKETS.lock().await;
    let s = socket_data.get(socket_id);
    if let Some(s) = s {
        return s.clone();
    }
    None
}

/// Attempts to authenticate a socket by assigning it a new name and verifying its ID in the socket registry.
///
/// # Parameters
/// - `new_name`: The new name to assign to the socket.
/// - `socket_id`: The unique identifier for the socket.
///
/// # Returns
/// Returns `WSSMessage::Success` if the socket is authenticated successfully, or `WSSMessage::Error` with a message "Unauthorized" if the authentication fails.
pub async fn authenticate_socket(new_name: String, socket_id: String) -> WSSMessage {
    let mut socket_data = SOCKETS.lock().await;
    for (id, name) in socket_data.iter_mut() {
        if *id == socket_id {
            *name = Some(new_name.clone()); // Dereference to modify
            return WSSMessage::Success
        }
    }
    WSSMessage::Error("Unauthorized".into())
}

/// Checks if a socket ID is authenticated by looking up if it has an associated name in the global socket data.
///
/// # Parameters
/// - `socket_id`: A reference to the string that identifies the socket.
///
/// # Returns
/// Returns `true` if the socket has an associated name, indicating it is authenticated; otherwise, returns `false`.
pub async fn is_authenticated(socket_id: &String) -> bool {
    let socket_data = SOCKETS.lock().await;
    let s = socket_data.get(socket_id);
    if let Some(s) = s {
        return s.is_some();
    }
    false
}

/// Retrieves the next message from a WebSocket stream, if available.
///
/// # Parameters
/// - `websocket`: A mutable reference to the WebSocket stream.
///
/// # Returns
/// Returns `Ok(Some(Message))` if a message is available, `Ok(None)` if no messages are available, or an error if one occurs during message retrieval.
pub async fn get_message(websocket: &mut WebSocketStream<TcpStream>) -> Result<Option<Message>> {
    let msg = websocket.next().await;
    if msg.is_none() {
        return Ok(None);
    }
    let msg = msg.unwrap();
    Ok(Some(msg?))
}

/// Sends a WebSocket message through the provided WebSocket stream.
///
/// # Parameters
/// - `websocket`: A mutable reference to the WebSocket stream.
/// - `msg`: The `WSSMessage` to be sent.
///
/// # Returns
/// Returns `Ok(())` if the message is sent successfully, or an error if the sending fails.
pub async fn send_message(websocket: &mut WebSocketStream<TcpStream>, msg: WSSMessage) -> Result<()> {
    if let Err(e) = websocket.send(msg.into()).await {
        error!("[WSS operations] Something went wrong sending raw_msg to WS clinet: {:#?}", e);
        return Err(e.into())
    };
    Ok(())
}

/// Removes a socket ID from the global socket data, effectively untracking the socket.
///
/// # Parameters
/// - `socket_id`: A reference to the string that identifies the socket to be removed.
pub async fn remove_socket(socket_id: &String) {
    let mut sockets = SOCKETS.lock().await;
    sockets.remove(socket_id);
}