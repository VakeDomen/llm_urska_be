use candle_core::Tensor;
use once_cell::sync::Lazy;
use qdrant_client::{client::QdrantClient, qdrant::{SearchPoints, SearchResponse}};
use tokio::sync::Mutex;
use anyhow::Result;
use crate::config::{QDRANT_COLLECTION, QDRANT_SERVER};

/// Static global client for accessing the Qdrant database.
///
/// This variable initializes a Qdrant client connection that is used to interact with the Qdrant vector database.
/// It is lazily instantiated and locked via a mutex to ensure thread-safe access across asynchronous tasks.
///
/// # Panics
/// - Panics if the connection to the Qdrant database cannot be established, indicating a configuration or network issue.
static QDRANT_CLIENT: Lazy<Mutex<QdrantClient>> = Lazy::new(|| {
    let client = match QdrantClient::from_url(QDRANT_SERVER).build() {
        Ok(c) => c,
        Err(e) => panic!("Can't establish Qdrant DB connection: {:#?}", e),
    };
    Mutex::new(client)
});

/// Performs a vector search in the Qdrant database using a given embedding tensor.
///
/// This function converts the provided tensor into a vector of `f32` values and uses it to query the Qdrant database.
/// It searches for points in the specified collection that are nearest to the input vector, returning results with payloads.
///
/// # Parameters
/// - `embedding`: The tensor representing an embedding that needs to be searched within the Qdrant vector space.
///
/// # Returns
/// Returns a `Result` containing the search response from Qdrant if successful. This response includes details of the
/// nearest points found in the vector space.
///
/// # Errors
/// - Returns an error if the tensor conversion fails or if the Qdrant search query encounters issues.
pub async fn vector_search(embedding: Tensor) -> Result<SearchResponse> {
    // Convert the tensor to a vector of f32 values.
    let embedding_vec = embedding.to_vec2::<f32>()?.first().unwrap().clone();
    let guard = QDRANT_CLIENT.lock().await;
    let search_result = guard
        .search_points(&SearchPoints {
            collection_name: QDRANT_COLLECTION.to_string(),
            vector: embedding_vec,
            limit: 4,
            with_payload: Some(true.into()),
            ..Default::default()
        })
        .await?;

    Ok(search_result)
}
