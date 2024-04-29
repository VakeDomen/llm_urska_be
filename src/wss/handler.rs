use std::{thread, time::Duration};

use crate::{llm::{embedding::generate_prompt_embedding, prompt::{prompt_model, Prompt}}, storage::{cache_wss::{dec_que, inc_que, que_len, que_pos}, qdrant::vector_search}};

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
    let pos = que_pos(&socket_id).await;
    send_message(&mut websocket, WSSMessage::QuePosResponse(pos)).await
}

async fn handle_prompt(
    question: String,
    socket_id: String, 
    mut websocket: &mut WebSocketStream<TcpStream>
) -> Result<()> {
    inc_que(socket_id.clone()).await;
    send_message(websocket, WSSMessage::Success).await?;
    
    let mut pos = que_pos(&socket_id).await;
    while pos > 2 {
        send_message(websocket, WSSMessage::QuePosResponse(pos)).await?;
        thread::sleep(Duration::from_secs(1));
        pos = que_pos(&socket_id).await;
    }

    let emb = generate_prompt_embedding(&question, Some(websocket)).await?;
    let result = vector_search(emb).await?;
    println!("RES: {:#?}", result);

    let _ = match prompt_model(Prompt::One(question), Some(websocket)).await {
        Ok(response) => send_message(&mut websocket, WSSMessage::PromptResponse(response)).await,
        Err(e) => send_message(&mut websocket, WSSMessage::Error(e.to_string())).await,
    };
    dec_que(socket_id).await;
    Ok(())
}