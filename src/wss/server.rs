use std::net::SocketAddr;
use log::{info, error};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::{accept_async, tungstenite::{Error, Message, Result}};
use uuid::Uuid;

use crate::{config::LISTEN_ADDRESS, storage::cache_wss::SOCKETS};

use super::{handler::handle, message::WSSMessage, operations::{get_message, remove_socket}};

/// Initializes and manages a WebSocket server by listening for incoming connections on a predefined address.
///
/// # Errors
/// This function will panic if it fails to bind the TCP listener to the specified address or if it cannot retrieve the peer address from a connected stream.
///
/// # Panics
/// - Panics if the TCP listener cannot bind to `LISTEN_ADDRESS`.
/// - Panics if it fails to obtain a peer address from a connected stream.
pub async fn start_server() {
    let listener = TcpListener::bind(LISTEN_ADDRESS).await.expect("[WSS] Can't listen");
    info!("[WSS] Web socket server started: {}", LISTEN_ADDRESS);
    while let Ok((stream, addr)) = listener.accept().await {
        info!("[WSS] New Connection: {:?}", addr);
        let peer = stream.peer_addr().expect("[WSS] connected streams should have a peer address");
        tokio::spawn(accept_connection(peer, stream));
    }
    info!("[WSS] Server closed!");
}

/// Asynchronously handles a new WebSocket connection from a client.
///
/// This function is spawned for each new connection accepted by the WebSocket server.
/// It attempts to process the connection by calling `handle_connection`.
/// If an error occurs during the handling, it categorizes and logs the error appropriately.
///
/// # Parameters
/// - `peer`: The socket address of the peer (client).
/// - `stream`: The TCP stream associated with the connection.
///
/// # Errors
/// Processes the result from `handle_connection` and matches against known error types to decide on further action.
/// Ignores benign errors like `ConnectionClosed`, `Protocol` errors, or `Utf8` encoding issues, logging other unexpected errors.
async fn accept_connection(peer: SocketAddr, stream: TcpStream) {
    if let Err(e) = handle_connection(peer, stream).await {
        match e {
            Error::ConnectionClosed | Error::Protocol(_) | Error::Utf8 => (),
            err => error!("[WSS] Error processing connection: {}", err),
        }
    }
}

/// Handles a WebSocket connection after it has been established, processing messages in a continuous loop until the connection is closed.
///
/// This function upgrades a TCP stream to a WebSocket connection, generates a unique socket ID for it, and manages incoming WebSocket messages.
/// It continues to receive and process messages until a close command is received or an error occurs. Errors lead to the termination of the loop and the removal of the socket from the active connections list.
///
/// # Parameters
/// - `_`: The socket address of the peer, not currently used.
/// - `stream`: The TCP stream to be upgraded to a WebSocket connection.
///
/// # Returns
/// Returns `Ok(())` on successful completion of handling the connection or after a closure message. If an error occurs during WebSocket upgrade or message handling, the function panics.
///
/// # Panics
/// - Panics if the WebSocket handshake fails.
/// - Panics if message handling fails due to an unrecoverable error.
pub async fn handle_connection(_: SocketAddr, stream: TcpStream) -> Result<()>  {
    let mut websocket = accept_async(stream).await.expect("Failed to accept new socket connection");
    let socket_id = Uuid::new_v4().to_string();
    // Add the connection to the list
    {
        let mut conns = SOCKETS.lock().await;
        conns.insert(socket_id.clone(), None);
    }
    
    loop {
        let raw_msg: Option<Message> = match get_message(&mut websocket).await {
            Ok(m) => m,
            Err(_) => {
                remove_socket(&socket_id).await;
                break;
            }
        };

        if raw_msg.is_none() {
            continue;
        }

        let raw_msg = raw_msg.unwrap();

        if raw_msg.is_close() {
            info!("[WSS] Closing socket!");
            remove_socket(&socket_id).await;
            break;
        }

        if raw_msg.is_binary() || raw_msg.is_text() {
            let msg = WSSMessage::from(raw_msg);
            if let Err(e) = handle(msg, socket_id.clone(), &mut websocket).await {
                error!("[WSS Handler] Error handling socket message: {:#?}", e);
                remove_socket(&socket_id).await;
                break;
            };
        } 
    }
    Ok(())
}