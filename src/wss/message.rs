use log::error;
use serde::Serialize;
use serde_any::Format;
use tokio_tungstenite::tungstenite::Message;

#[derive(Debug, Serialize)]
pub enum Rating {
    Positive, 
    Neutral,
    Negative,
}

/// Represents different types of messages that can be sent between the client and server over WebSocket connections.
///
/// This enum differentiates between various client commands and server responses, facilitating a structured communication protocol over WebSockets.
#[derive(Debug, Serialize)]
pub enum WSSMessage {
    /// Command from client requesting processing of a prompt.
    Prompt(String, Vec<String>), // question, vec<collections>

    RateResponse(String, Rating),

    RatePassage(String, Rating),

    /// Command from client to retrieve the current queue length.
    QueueLen,
    /// Command from client to retrieve the current queue position.
    QueuePos,
    /// Represents an unknown or unparsable command.
    Unknown,

    /// Response to client indicating the status of a prompt processing.
    PromptStatus(String),
    /// Response to client with the result of a prompt processing.
    PromptResponse(String, String), // id, content
    /// Response to client with the passage related to the prompt.
    PromptPassage(String, String), // id, content
    /// Response to client with tokenized elements of the prompt response.
    PromptResponseToken(String),
    /// Response to client with the current queue length.
    QueLenResponse(usize),
    /// Response to client with the position in the queue.
    QuePosResponse(i32),
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
            return parse_prompt_msg(message_string);
        } 

        if message_string.starts_with("RateResponse ") {
            return parse_rate_resonse(message_string);
        }

        if message_string.starts_with("RatePassage ") {
            return parse_rate_passage(message_string);
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

fn parse_rate_passage(message_string: String) -> WSSMessage {
    let parts: Vec<&str> = message_string
        .split(' ')
        .collect();

    if parts.len() != 3 {
        error!("[WSS message parser] Invalid rate passage command format.");
        return WSSMessage::Unknown;
    }

    let passage_id = parts[1].to_string();
    let rating = parse_rating(parts[2]);
    WSSMessage::RatePassage(passage_id, rating)
}

fn parse_rate_resonse(message_string: String) -> WSSMessage {
    let parts: Vec<&str> = message_string
        .split(' ')
        .collect();

    if parts.len() != 3 {
        error!("[WSS message parser] Invalid rate response command format.");
        return WSSMessage::Unknown;
    }

    let response_id = parts[1].to_string();
    let rating = parse_rating(parts[2]);
    WSSMessage::RateResponse(response_id, rating)
}


fn parse_rating(rating_string: &str) -> Rating {
    match rating_string.to_ascii_lowercase().as_str() {
        "positive" => Rating::Positive,
        "negative" => Rating::Negative,
        _ => Rating::Neutral,
    }
}

fn parse_prompt_msg(message_string: String) -> WSSMessage {
    let parts: Vec<&str> = message_string
        .splitn(2, ' ')
        .collect();
    
    if parts.len() < 2 {
        error!("[WSS message parser] Invalid QUESTION command format.");
        return WSSMessage::Unknown;
    }

    let prompt_string = parts[1].to_string();
    
    let prompt_parts: Vec<&str> = prompt_string
        .splitn(2, ' ')
        .collect();

    if prompt_parts.len() < 2 {
        return WSSMessage::Prompt(prompt_string, vec![]);
    }
    
    let question_string = prompt_parts[1].to_string();    
    // assume first part is collections to prompt sparated by comma and no space -> col1,col3,col4
    let collections: Vec<String> = prompt_parts[0]
        .to_string()
        .split(',')
        .map(|s| s.to_string())
        .collect();

    
    WSSMessage::Prompt(question_string, collections)
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