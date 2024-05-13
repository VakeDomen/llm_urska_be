use tokio::net::TcpStream;
use tokio_tungstenite::WebSocketStream;
use anyhow::Result;
use crate::{llm::prompt::{prompt_model, Prompt}, wss::{message::WSSMessage, operations::send_message}};

use super::collector::Passage;

pub async fn rerank(
    question: String, 
    passages: Vec<Passage>,
    websocket: &mut WebSocketStream<TcpStream>,
) -> Result<Vec<Passage>> {
    let mut filtered_passages = vec![];
    send_message(websocket, WSSMessage::PromptStatus("Sifting through docs...".to_string())).await?;
    for passage in passages {
        match prompt_model(Prompt::Rerank(question.clone(), passage.clone()), None).await {
            Ok(generated_response) => {
                // send_message(websocket, WSSMessage::PromptResponse(generated_response.clone())).await?;
                if generated_response.to_ascii_lowercase().contains("true") {
                    filtered_passages.push(passage)
                }
            },
            Err(e) => {
                send_message(websocket, WSSMessage::Error(e.to_string())).await?;
                filtered_passages.push(passage)
            },
        }
    }

    Ok(filtered_passages)
}