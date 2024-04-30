use log::error;
use serde::Serialize;
use serde_any::Format;
use tokio_tungstenite::tungstenite::Message;


/// Represents different types of messages that can be sent between the client and server over WebSocket connections.
///
/// This enum differentiates between various client commands and server responses, facilitating a structured communication protocol over WebSockets.
#[derive(Debug, Serialize)]
pub enum WSSMessage {
    /// Command from client requesting processing of a prompt.
    Prompt(String),
    /// Command from client to retrieve the current queue length.
    QueueLen,
    /// Command from client to retrieve the current queue position.
    QueuePos,
    /// Represents an unknown or unparsable command.
    Unknown,

    /// Response to client indicating the status of a prompt processing.
    PromptStatus(String),
    /// Response to client with the result of a prompt processing.
    PromptResponse(String),
    /// Response to client with the passage related to the prompt.
    PromptPassage(String),
    /// Response to client with tokenized elements of the prompt response.
    PromptResponseToken(String),
    /// Response to client with the current queue length.
    QueLenResponse(usize),
    /// Response to client with the position in the queue.
    QuePosResponse(usize),
    /// Generic success message.
    Success,
    /// Error message with details.
    Error(String)
}


/// Converts a `Message` into a `WSSMessage` by parsing the incoming text.
///
/// This implementation attempts to decode a WebSocket message text into a specific `WSSMessage`.
/// If the message format is incorrect or cannot be processed, it returns `WSSMessage::Unknown`.
impl From<Message> for WSSMessage {
    fn from(value: Message) -> Self {
        let message_string: String = match value.into_text() {
            Ok(t) => t,
            Err(e) => {
                error!("[WSS message parser] Error parsing WSS message: {:#?}", e);
                return Self::Unknown;
            },
        };

        // parse game command
        if message_string.starts_with("Prompt ") {
            let tokens: Vec<&str> = message_string.splitn(2, ' ').collect();
            if tokens.len() < 2 {
                error!("[WSS message parser] Invalid QUESTION command format.");
                return Self::Unknown;
            }

            let question_string = tokens[1].to_string();
            return Self::Prompt(question_string);
        } 

        if message_string.eq("QueueLen") {
            return Self::QueueLen;
        }

        if message_string.eq("QueuePos") {
            return Self::QueuePos;
        }

        Self::Unknown
    }
}

/// Converts a `WSSMessage` back into a WebSocket `Message` by serializing it to a JSON string.
///
/// This implementation encodes a `WSSMessage` into a JSON-formatted string. If serialization fails, it returns a message indicating the failure.
impl From<WSSMessage> for Message {
    fn from(val: WSSMessage) -> Self {
        match serde_any::to_string(&val, Format::Json)  {
            Ok(s) => Message::Text(s),
            Err(e) => Message::Text(format!("Couldn't serialize message: {:#?}", e)),
        }
    }
}