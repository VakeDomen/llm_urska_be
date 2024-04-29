use candle_core::Tensor;
use once_cell::sync::Lazy;
use qdrant_client::{client::QdrantClient, qdrant::{SearchPoints, SearchResponse}};
use tokio::sync::Mutex;
use anyhow::Result;
use crate::config::{QDRANT_COLLECTION, QDRANT_SERVER};

static QDRANT_CLIENT: Lazy<Mutex<QdrantClient>> = Lazy::new(|| {
    let client = match QdrantClient::from_url(QDRANT_SERVER).build() {
        Ok(c) => c,
        Err(e) => panic!("Can't establish Qdrant DB connection: {:#?}", e),
    };
    Mutex::new(client)
});

pub async fn vector_search(embedding: Tensor) -> Result<SearchResponse> {
    // Convert the tensor to a vector of f32 values.
    let embedding_vec = embedding.to_vec2::<f32>()?.first().unwrap().clone();

    // Acquire the lock and store the guard in a variable.
    // Make sure to bind the lock guard to a variable.
    let guard = QDRANT_CLIENT.lock().await;

    // // Use the lock_guard (guard in this case) to perform the search.
    // // Pass the client reference correctly to the search function.
    let search_result = guard
        .search_points(&SearchPoints {
            collection_name: QDRANT_COLLECTION.to_string(),
            vector: embedding_vec,
            limit: 2,
            with_payload: Some(true.into()),
            ..Default::default()
        })
        .await?;

    Ok(search_result)
}
