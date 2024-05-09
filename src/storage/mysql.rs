extern crate mysql;
extern crate once_cell;

use mysql::*;
use mysql::prelude::*;
use once_cell::sync::Lazy;
use std::sync::Mutex;
use anyhow::Result;

use crate::config::MYSQL_URL;

use super::models::prompt::InsertablePrompt;

// Define your database URL here. Ensure it is correctly formatted and secure.

// Create a static MySQL connection pool wrapped in a Mutex and Lazy for thread safety and one-time initialization.
static MYSQL_POOL: Lazy<Mutex<Pool>> = Lazy::new(|| {
    let pool = match Pool::new(MYSQL_URL) {
        Ok(p) => p,
        Err(e) => panic!("Can't establish MySQL DB connection: {:#?}", e),
    };
    Mutex::new(pool)
});

// Function to get a database connection from the pool
fn get_db_conn() -> Result<PooledConn> {
    // Access the pool inside the mutex, lock can fail if the mutex is poisoned
    let pool = MYSQL_POOL.lock().unwrap();
    Ok(pool.get_conn()?)
}

pub fn insert_prompt(prompt: InsertablePrompt) -> Result<()> {
    // Define MySQL connection options
    let mut conn = get_db_conn()?;

    // Start a transaction
    conn.exec_drop(
        r"INSERT INTO prompts (id, question, response, embedding_time, response_time, que_time, total_time, error, rating)
            VALUES (:id, :question, :response, :embedding_time, :response_time, :que_time, :total_time, :error, :rating)",
        params! {
            "id" => &prompt.id,
            "question" => &prompt.question,
            "response" => &prompt.response,
            "embedding_time" => prompt.embedding_time.as_millis() as i64,
            "response_time" => prompt.response_time.as_millis() as i64,
            "que_time" => prompt.que_time.as_millis() as i64,
            "total_time" => prompt.total_time.as_millis() as i64,
            "error" => &prompt.error,
            "rating" => &prompt.rating,
        },
    )?;

    // Insert into `lookups` table if docs are available
    for doc in prompt.docs.iter() {
        conn.exec_drop(
            r"INSERT INTO lookups (id, prompt_id, passage_id, text, rating)
                VALUES (:id, :prompt_id, :passage_id, :text, :rating)",
            params! {
                "id"  => &doc.id,
                "prompt_id" => &prompt.id,
                "passage_id" => &doc.passage_id,
                "text" => &doc.text,
                "rating" => doc.rating,
            },
        )?;
    }
    Ok(())
}