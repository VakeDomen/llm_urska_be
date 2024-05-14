use std::{thread, time::Duration};

use crate::{
    config::USE_HYDE, 
    controllers::{collector::sample_passages, reranker::rerank}, 
    llm::{embedding::generate_prompt_embedding, prompt::{prompt_model, Prompt}}, 
    storage::{
        cache_wss::{dec_que, inc_que, que_len, que_pos}, 
        models::prompt::NewPrompt, 
        mysql::{insert_prompt, rate_passage, rate_response},
    }
};

use super::{message::{Rating, WSSMessage}, operations::send_message};
use anyhow::Result;
use tokio::net::TcpStream;
use tokio_tungstenite::WebSocketStream;

/// Handles incoming WebSocket messages based on their type.
///
/// This function routes different `WSSMessage` types to specific handling functions. It is the primary entry point for processing all messages from WebSocket clients.
///
/// # Parameters
/// - `msg`: The message received from the WebSocket client.
/// - `socket_id`: Unique identifier for the WebSocket connection.
/// - `websocket`: Mutable reference to the WebSocket stream.
///
/// # Returns
/// Returns `Ok(())` if the message is handled successfully. Otherwise, it forwards any errors encountered during handling.
pub async fn handle(
    msg: WSSMessage, 
    socket_id: String, 
    websocket: &mut WebSocketStream<TcpStream>
) -> Result<()> {
    match msg {
        WSSMessage::Prompt(question, collections) => handle_prompt(question, collections, socket_id, websocket).await,
        WSSMessage::QueueLen => handle_que_len(websocket).await,
        WSSMessage::QueuePos => handle_que_pos(socket_id, websocket).await,
        WSSMessage::RatePassage(passage_id, rating) => handle_rate_passage(passage_id, rating, websocket).await,
        WSSMessage::RateResponse(response_id, rating) => handle_rate_response(response_id, rating, websocket).await,
        _ => Ok(())
    }
}

async fn handle_rate_response(
    response_id: String, 
    rating: Rating, 
    websocket: &mut WebSocketStream<TcpStream>
) -> std::prelude::v1::Result<(), anyhow::Error> {
    match rate_response(response_id, rating) {
        Ok(_) => send_message(websocket, WSSMessage::Success).await,
        Err(e) => send_message(websocket, WSSMessage::Error(e.to_string())).await,
    }
}

async fn handle_rate_passage(
    passage_id: String, 
    rating: Rating, 
    websocket: &mut WebSocketStream<TcpStream>
) -> Result<()> {
    match rate_passage(passage_id, rating) {
        Ok(_) => send_message(websocket, WSSMessage::Success).await,
        Err(e) => send_message(websocket, WSSMessage::Error(e.to_string())).await,
    }
}

/// Sends the current queue length to the client.
///
/// # Parameters
/// - `websocket`: Mutable reference to the WebSocket stream.
///
/// # Returns
/// Forwards the result of sending the queue length message to the client.
async fn handle_que_len(
    websocket: &mut WebSocketStream<TcpStream>
) -> Result<()> {
    let len = que_len().await;
    send_message(websocket, WSSMessage::QueLenResponse(len)).await
}

/// Sends the position of the socket in the queue to the client.
///
/// # Parameters
/// - `socket_id`: Unique identifier for the WebSocket connection.
/// - `websocket`: Mutable reference to the WebSocket stream.
///
/// # Returns
/// Forwards the result of sending the queue position message to the client.
async fn handle_que_pos(
    socket_id: String,
    websocket: &mut WebSocketStream<TcpStream>
) -> Result<()> {
    let pos = que_pos(&socket_id).await;
    send_message(websocket, WSSMessage::QuePosResponse(pos)).await
}

/// Handles a prompt message from the client by adding it to the queue, generating a prompt response, and managing its position in the queue.
///
/// The function increases the queue, keeps the client updated on their position, generates and sends content based on the prompt, and decreases the queue once done.
///
/// # Parameters
/// - `question`: The question or prompt from the client.
/// - `socket_id`: Unique identifier for the WebSocket connection.
/// - `websocket`: Mutable reference to the WebSocket stream.
///
/// # Returns
/// Forwards any errors encountered during processing. Returns `Ok(())` if processing is completed successfully.
async fn handle_prompt(
    question: String,
    collections: Vec<String>,
    socket_id: String, 
    websocket: &mut WebSocketStream<TcpStream>
) -> Result<()> {
    // setup
    let state = NewPrompt::from(question.clone());
    inc_que(socket_id.clone()).await;
    send_message(websocket, WSSMessage::Success).await?;
    
    // que
    let state = state.enter_que();
    let mut pos = que_pos(&socket_id).await;
    let mut len = que_len().await;
    send_message(websocket, WSSMessage::QuePosResponse(pos)).await?;
    send_message(websocket, WSSMessage::QueLenResponse(len)).await?;
    while pos > 2 {
        thread::sleep(Duration::from_secs(1));
        pos = que_pos(&socket_id).await;
        len = que_len().await;
        send_message(websocket, WSSMessage::QuePosResponse(pos)).await?;
        send_message(websocket, WSSMessage::QueLenResponse(len)).await?;
    }

    // hyde
    let (hyde_prompt, state) = if USE_HYDE {
        let state = state.hyde_generation();
        match prompt_model(Prompt::Hyde(question.clone()), Some(websocket)).await {
            Ok(generated_response) => (generated_response.clone(), state.start_embedding(generated_response)),
            Err(e) => {
                send_message(websocket, WSSMessage::Error(e.to_string())).await?;
                (question.clone(), state.start_embedding(e.to_string()))
            },
        }
    } else {
        (question.clone(), state.start_embedding())
    };
    
    // embedding
    let emb = generate_prompt_embedding(&question, Some(websocket)).await?;
    let emb_hyde = generate_prompt_embedding(&hyde_prompt, Some(websocket)).await?;
    let passages = sample_passages(emb, emb_hyde, collections).await?;
    let passages = rerank(question.clone(), passages, websocket).await?;
    let number_of_results = passages.len();
    
    let prompt = match number_of_results {
        0 => Prompt::PlainQuestion(question),
        _ => Prompt::RagPrompt(question, passages.clone()),
    };

    // final response
    let state = state.start_responding(passages);
    let response = match prompt_model(prompt, Some(websocket)).await {
        Ok(generated_response) => generated_response,
        Err(e) => {
            send_message(websocket, WSSMessage::Error(e.to_string())).await?;
            e.to_string()
        },
    };
    let state = state.finalize(response);
    dec_que(socket_id).await;
    for passage in &state.passages {
        let _ = send_message(websocket, WSSMessage::PromptPassage(passage.id.clone(), passage.text.clone())).await;
    }
    let _ = send_message(websocket, WSSMessage::PromptResponse(state.id.clone(), state.response.clone())).await?;
    insert_prompt(state)?;
    Ok(())
}

