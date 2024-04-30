use logging::logger::init_logging;
use wss::server::start_server;
use anyhow::Result;

mod llm;
mod config;
mod wss;
mod storage;
mod logging;

#[tokio::main]
async fn main() -> Result<()> {
    init_logging()?;
    start_server().await;
    Ok(())
}


