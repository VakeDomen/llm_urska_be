use std::io::Write;

use tokio::net::TcpStream;
use tokio_tungstenite::WebSocketStream;
use anyhow::Result;
use log::debug;

use crate::{config::VERBOSE_PROMPT, wss::{message::WSSMessage, operations::send_message}};

pub enum FlushType {
    Token,
    Status,
}

/// Prints a token and ensures the output buffer is flushed, used primarily for verbose logging.
///
/// # Arguments
/// * `token` - The token to print.
///
/// # Returns
/// A `Result` indicating success or any error during flushing.
pub async fn flush_message(
    content: &str, 
    websocket: &mut Option<&mut WebSocketStream<TcpStream>>,
    flush_type: FlushType,
) -> Result<()> {
    if let Some(ws) = websocket {
        match flush_type {
            FlushType::Token => send_message(ws, WSSMessage::PromptResponseToken(content.into())).await?,
            FlushType::Status => send_message(ws, WSSMessage::PromptStatus(content.into())).await?,
        }
        
    }

    if VERBOSE_PROMPT {
        debug!("{content}");
        std::io::stdout().flush()?;
    }
    Ok(())
}

