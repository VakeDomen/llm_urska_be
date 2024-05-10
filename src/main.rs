use logging::logger::init_logging;
use wss::server::start_server;
use anyhow::Result;
use dotenv::dotenv;

mod llm;
mod config;
mod wss;
mod storage;
mod logging;
mod controllers;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    init_logging()?;
    start_server().await;
    Ok(())
}


