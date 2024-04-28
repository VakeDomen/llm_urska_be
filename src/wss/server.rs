use std::net::SocketAddr;
use log::{info, error};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::{tungstenite::{Result, Message, Error}, accept_async};
use uuid::Uuid;

use crate::{config::LISTEN_ADDRESS, storage::cache_wss::SOCKETS};

use super::{handler::handle, message::WSSMessage, operations::{get_message, remove_socket}};

pub async fn start_server() {
    println!("HELLO");
    info!("HELLO");
    let listener = TcpListener::bind(LISTEN_ADDRESS).await.expect("[WSS] Can't listen");
    info!("[WSS] Web socket server started: {}", LISTEN_ADDRESS);
    while let Ok((stream, _)) = listener.accept().await {
        info!("[WSS] New Connection");
        let peer = stream.peer_addr().expect("[WSS] connected streams should have a peer address");
        tokio::spawn(accept_connection(peer, stream));
    }
    info!("[WSS] Server closed!");
}

async fn accept_connection(peer: SocketAddr, stream: TcpStream) {
    if let Err(e) = handle_connection(peer, stream).await {
        match e {
            Error::ConnectionClosed | Error::Protocol(_) | Error::Utf8 => (),
            err => error!("[WSS] Error processing connection: {}", err),
        }
    }
}

pub async fn handle_connection(_: SocketAddr, stream: TcpStream) -> Result<()>  {
    let mut websocket = accept_async(stream).await.expect("Failed to accept new socket connection");
    let socket_id = Uuid::new_v4().to_string();
    // Add the connection to the list
    {
        let mut conns = SOCKETS.lock().unwrap();
        conns.insert(socket_id.clone(), None);
    }
    
    loop {
        let raw_msg: Option<Message> = match get_message(&mut websocket).await {
            Ok(m) => m,
            Err(_) => {
                remove_socket(&socket_id);
                break;
            }
        };

        if raw_msg.is_none() {
            continue;
        }

        let raw_msg = raw_msg.unwrap();

        if raw_msg.is_close() {
            info!("[WSS] Closing socket!");
            remove_socket(&socket_id);
            break;
        }

        if raw_msg.is_binary() || raw_msg.is_text() {
            let msg = WSSMessage::from(raw_msg);
            if let Err(e) = handle(msg, socket_id.clone(), &mut websocket).await {
                error!("[WSS Handler] Error handling socker message: {:#?}", e);
                remove_socket(&socket_id);
                break;
            };
        } 
    }
    Ok(())
}