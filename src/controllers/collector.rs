use qdrant_client::qdrant::SearchResponse;


pub enum Collector {
    Rag(Collection, Samples),
    Raw(Collection, Samples),
}
pub type Samples = usize;
pub type Collection = String;

#[derive(Debug, Clone)]
pub struct Passage {
    pub id: String,
    pub text: String,
}

pub fn sample_docs(
    prompt: String, 
    hyde_prompt: String,
    collections: Vec<String>,
) -> Vec<Passage> {
    vec![]
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
