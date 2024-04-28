use log::error;
use serde::Serialize;
use serde_any::Format;
use tokio_tungstenite::tungstenite::Message;


#[derive(Debug, Serialize)]
pub enum WSSMessage {
    // from client
    Question(String),
    Unknown,

    // to client
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
        if message_string.starts_with("QUESTION ") {
            let tokens: Vec<&str> = message_string.splitn(2, ' ').collect();
            if tokens.len() < 3 {
                println!("[WSS message parser] Invalid QUESTION command format.");
                return Self::Unknown;
            }

            let question_string = tokens[1].to_string();
            return Self::Question(question_string);
        } 

        Self::Unknown
    }
}

impl Into<Message> for WSSMessage {
    fn into(self) -> Message {
        match serde_any::to_string(&self, Format::Json)  {
            Ok(s) => Message::Text(s),
            Err(e) => Message::Text(format!("Couldn't serialize message: {:#?}", e)),
        }
    }
}