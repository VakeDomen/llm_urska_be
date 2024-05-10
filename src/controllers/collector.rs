use candle_core::Tensor;
use qdrant_client::qdrant::SearchResponse;
use anyhow::Result;

use crate::{config::COLLECOTRS, storage::qdrant::vector_search};

#[derive(Debug, Clone)]
pub enum Collector {
    Hyde(Collection, Samples),
    Raw(Collection, Samples),
}
pub type Samples = u64;
pub type Collection = String;

#[derive(Debug, Clone)]
pub struct Passage {
    pub id: String,
    pub text: String,
}

pub async fn sample_passages(
    prompt_tensor: Tensor, 
    hyde_prompt_tensor: Tensor,
    collections: Vec<String>,
) -> Result<Vec<Passage>> {
    let mut passages = vec![];
    for collection in collections {
        if let Some(collector) = get_collector(collection) {
            let response = match collector {
                Collector::Hyde(col, lookups) => vector_search(hyde_prompt_tensor.clone(), col, lookups).await?,
                Collector::Raw(col, lookups) => vector_search(prompt_tensor.clone(), col, lookups).await?,
            };
            for passage in extract_node_content(response) {
                passages.push(passage);
            }
        }
    }
    Ok(passages)
}


fn get_collector(collection: String) -> Option<Collector> {
    for collector in COLLECOTRS.iter() {
        match collector {
            Collector::Hyde(col, _) => if col.eq(&collection) { return Some(collector.clone()); },
            Collector::Raw(col, _) => if col.eq(&collection) { return Some(collector.clone()); },
        }
    }
    None
}

/// Extracts the content of nodes from a search response into a vector of strings.
///
/// This function parses the payload of each search result point, looking specifically for a field labeled `_node_content`.
/// It gathers all such content into a list, which represents extracted data from a search engine's response.
///
/// # Parameters
/// - `response`: A `SearchResponse` object containing the results from a search query.
///
/// # Returns
/// Returns a vector of strings, each representing the content of a node extracted from the search response.
pub fn extract_node_content(response: SearchResponse) -> Vec<Passage> {
    let mut node_contents = Vec::new();
    for point in &response.result {
        let mut passage_id = "".to_string();
        let mut text = "".to_string();

        if let Some(content) = point.payload.get("text") {
            if let Some(text_l) = content.as_str() {
                text = text_l.to_string();
            }
        }


        if let Some(content) = point.payload.get("id") {
            if let Some(id) = content.as_str() {
                passage_id = id.to_string();
            }
        }

        node_contents.push(Passage {
            id: passage_id, 
            text
        });
    }
    node_contents
}
