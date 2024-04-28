use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio_tungstenite::{tungstenite::{Error, Message}, WebSocketStream};
use log::error;
use anyhow::Result;
use crate::storage::cache_wss::SOCKETS;

use super::message::WSSMessage;

pub fn get_socket_name(socket_id: &String) -> Option<String> {
    let socket_data = SOCKETS.lock().unwrap();
    let s = socket_data.get(socket_id);
    if let Some(s) = s {
        return s.clone();
    }
    None
}

pub fn authenticate_socket(new_name: String, socket_id: String) -> WSSMessage {
    let mut socket_data = SOCKETS.lock().unwrap();
    for (id, name) in socket_data.iter_mut() {
        if *id == socket_id {
            *name = Some(new_name.clone()); // Dereference to modify
            return WSSMessage::Success
        }
    }
    WSSMessage::Error("Unauthorized".into())
}

pub fn is_authenticated(socket_id: &String) -> bool {
    let socket_data = SOCKETS.lock().unwrap();
    let s = socket_data.get(socket_id);
    if let Some(s) = s {
        return s.is_some();
    }
    false
}

pub async fn get_message(websocket: &mut WebSocketStream<TcpStream>) -> Result<Option<Message>> {
    let msg = websocket.next().await;
    if let None = msg {
        return Ok(None);
    }
    let msg = msg.unwrap();
    Ok(Some(msg?))
}

pub async fn send_message(websocket: &mut WebSocketStream<TcpStream>, msg: WSSMessage) -> Result<()> {
    if let Err(e) = websocket.send(msg.into()).await {
        error!("[WSS operations] Something went wrong sending raw_msg to WS clinet: {:#?}", e);
        return Err(e.into())
    };
    Ok(())
}

pub fn remove_socket(socket_id: &String) {
    let mut sockets = SOCKETS.lock().unwrap();
    sockets.remove(socket_id);
}