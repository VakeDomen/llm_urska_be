use crate::llm::prompt::{prompt_model, Prompt};

use super::{message::WSSMessage, operations::send_message};
use anyhow::Result;
use tokio::net::TcpStream;
use tokio_tungstenite::WebSocketStream;

pub async fn handle(
    msg: WSSMessage, 
    _socket_id: String, 
    websocket: &mut WebSocketStream<TcpStream>
) -> Result<()> {
    match msg {
        WSSMessage::Question(question) => handle_question(question, websocket).await,
        _ => Ok(())
    }
}

async fn handle_question(
    question: String, 
    mut websocket: &mut WebSocketStream<TcpStream>
) -> Result<()> {
    send_message(websocket, WSSMessage::Success).await?;
    match prompt_model(Prompt::One(question), Some(websocket)).await {
        Ok(response) => send_message(&mut websocket, WSSMessage::Response(response)).await,
        Err(e) => send_message(&mut websocket, WSSMessage::Error(e.to_string())).await,
    }

}