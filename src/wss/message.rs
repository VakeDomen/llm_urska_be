use log::error;
use serde::Serialize;
use serde_any::Format;
use tokio_tungstenite::tungstenite::Message;


#[derive(Debug, Serialize)]
pub enum WSSMessage {
    // from client
    Prompt(String),
    QueueLen,
    QueuePos,
    Unknown,

    // to client
    PromptStatus(String),
    PromptResponse(String),
    PromptPassage(String),
    PromptResponseToken(String),
    QueLenResponse(usize),
    QuePosResponse(usize),
    Success,
    Error(String)
}

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

impl From<WSSMessage> for Message {
    fn from(val: WSSMessage) -> Self {
        match serde_any::to_string(&val, Format::Json)  {
            Ok(s) => Message::Text(s),
            Err(e) => Message::Text(format!("Couldn't serialize message: {:#?}", e)),
        }
    }
}