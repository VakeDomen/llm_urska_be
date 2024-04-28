use super::message::WSSMessage;


pub fn handle(msg: WSSMessage, socket_id: String) -> WSSMessage {
    match msg {
        _ => return WSSMessage::Unknown,
    }
}