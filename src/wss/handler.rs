use crate::{llm::prompt::{prompt_model, Prompt}, storage::cache_wss::{dec_que, inc_que, que_len, que_pos}};

use super::{message::WSSMessage, operations::send_message};
use anyhow::Result;
use tokio::net::TcpStream;
use tokio_tungstenite::WebSocketStream;

pub async fn handle(
    msg: WSSMessage, 
    socket_id: String, 
    websocket: &mut WebSocketStream<TcpStream>
) -> Result<()> {
    match msg {
        WSSMessage::Prompt(question) => handle_prompt(question, socket_id, websocket).await,
        WSSMessage::QueueLen => handle_que_len(websocket).await,
        WSSMessage::QueuePos => handle_que_pos(socket_id, websocket).await,
        _ => Ok(())
    }
}

async fn handle_que_len(
    mut websocket: &mut WebSocketStream<TcpStream>
) -> Result<()> {
    let len = que_len().await;
    send_message(&mut websocket, WSSMessage::QueLenResponse(len)).await
}


async fn handle_que_pos(
    socket_id: String,
    mut websocket: &mut WebSocketStream<TcpStream>
) -> Result<()> {
    let pos = que_pos(socket_id).await;
    send_message(&mut websocket, WSSMessage::QuePosResponse(pos)).await
}

async fn handle_prompt(
    question: String,
    socket_id: String, 
    mut websocket: &mut WebSocketStream<TcpStream>
) -> Result<()> {
    inc_que(socket_id.clone()).await;
    send_message(websocket, WSSMessage::Success).await?;
    let _ = match prompt_model(Prompt::One(question), Some(websocket)).await {
        Ok(response) => send_message(&mut websocket, WSSMessage::PromptResponse(response)).await,
        Err(e) => send_message(&mut websocket, WSSMessage::Error(e.to_string())).await,
    };
    dec_que(socket_id).await;
    Ok(())
}