use std::path::Path;

use tokio::sync::Mutex;
use anyhow::{Error, Result};
use candle_core::{quantized::gguf_file::Content, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::{
    bert::{BertModel, Config, DTYPE}, 
    quantized_llama::{ModelWeights, MAX_SEQ_LEN}
};
use log::{info, error};
use once_cell::sync::Lazy;
use tokenizers::Tokenizer;

use crate::config::{EMBEDDING_MODEL_PATH, MODEL_PATH, TOKENIZER_PATH};

pub type LoadedModel = (ModelWeights, Tokenizer, Device, usize);
pub type LoadedEmbeddingModel = (BertModel, Tokenizer, Device);


/// Represents the selection of models available for use.
///
/// This enum is used to toggle between two loaded models, facilitating dynamic model switching.
#[derive(Debug)]
pub enum ModelSelector {
    First,
    Second,
}

/// Static global state for toggling between two models. This is used to manage which model is currently active.
pub static MODEL_TOGGLER: Lazy<Mutex<i8>> = Lazy::new(|| Mutex::new(0));

/// Static global state holding the first loaded model.
/// Panics if the model fails to load during initialization.
pub static MODEL1: Lazy<Mutex<LoadedModel>> = Lazy::new(|| {
    match load_llm_model(Some(0)) {
        Ok(m) => Mutex::new(m),
        Err(e) => panic!("Can't lazy load model: {:#?}", e),
    }
});

/// Static global state holding the second loaded model.
/// Panics if the model fails to load during initialization.
pub static MODEL2: Lazy<Mutex<LoadedModel>> = Lazy::new(|| {
    match load_llm_model(Some(1)) {
        Ok(m) => Mutex::new(m),
        Err(e) => panic!("Can't lazy load model: {:#?}", e),
    }
});

/// Assigns and toggles between two models based on a round-robin scheduling.
///
/// This function increments the model toggler and uses modulo operation to alternate between two models.
/// It ensures that each call toggles the active model, cycling between the first and the second models.
///
/// # Returns
/// Returns `ModelSelector::First` if the first model is selected, otherwise returns `ModelSelector::Second`.
pub async fn assign_model() -> ModelSelector {
    let mut toggle = MODEL_TOGGLER.lock().await;
    *toggle += 1;
    *toggle %= 2;
    if *toggle == 0 {
        ModelSelector::First
    } else {
        ModelSelector::Second
    }
}


/// Loads a BERT embedding model along with its tokenizer from disk, targeting a specified GPU device.
///
/// This function initializes a device according to a provided `gpu_id`, then attempts to load both a BERT model
/// and a tokenizer from predefined paths. The model and tokenizer are essential for generating embeddings for NLP tasks.
///
/// # Parameters
/// - `gpu_id`: An optional identifier for a GPU device. If `None`, the model loads on the default device.
///
/// # Returns
/// Returns a `Result` containing a tuple of the loaded model, tokenizer, and device if successful, or panics if
/// any loading step fails.
///
/// # Panics
/// - Panics if the BERT model or tokenizer cannot be loaded from the specified paths. This is typically due to file path issues
///   or incorrect model/tokenizer configurations.
pub fn load_bert_model(gpu_id: Option<usize>) -> Result<LoadedEmbeddingModel> {
    let device = load_device(gpu_id);
    let model = match load_pybin_bert_model_from_disk(EMBEDDING_MODEL_PATH, &device) { 
        Ok(m) => m,
        Err(e) => panic!("Can't load embedding model: {:#?}", e),
    };
    let tokenizer = match load_tokenizer(&format!("{}/tokenizer.json", EMBEDDING_MODEL_PATH)) {
        Ok(t) => t,
        Err(e) => panic!("Can't load tokenizer: {:#?}", e),
    };
    Ok((model, tokenizer, device))
}

/// Loads a large language model (LLM) along with its tokenizer from disk, configured to use a specified GPU device.
///
/// This function initializes a device based on the provided `gpu_id` and then attempts to load a large language model
/// and its corresponding tokenizer. The model and tokenizer are critical for processing large-scale NLP tasks efficiently.
///
/// # Parameters
/// - `gpu_id`: An optional identifier for a GPU device. If `None`, the model loads on the default device.
///
/// # Returns
/// Returns a `Result` containing a tuple of the loaded model, tokenizer, device configuration, and maximum sequence length,
/// if successful. If any component fails to load, the function panics with a detailed error message.
///
/// # Panics
/// - Panics if the LLM or tokenizer cannot be loaded from their respective paths. This could be due to issues such as incorrect file paths,
///   configuration errors, or hardware compatibility problems.
fn load_llm_model(gpu_id: Option<usize>) -> Result<LoadedModel> {
    let device = load_device(gpu_id);
    let model = match load_gguf_model_from_disk(MODEL_PATH, &device) { 
        Ok(m) => m,
        Err(e) => panic!("Can't load model: {:#?}", e),
    };
    let tokenizer = match load_tokenizer(TOKENIZER_PATH) {
        Ok(t) => t,
        Err(e) => panic!("Can't load tokenizer: {:#?}", e),
    };
    Ok((model, tokenizer, device, MAX_SEQ_LEN))
}


/// Loads a `Tokenizer` from a specified file path.
/// 
/// # Arguments
/// * `tokenizer_path` - A string slice that holds the file path to the tokenizer model.
///
/// # Returns
/// A `Result` which, on success, contains the `Tokenizer`, and on failure, contains an `Error`.
fn load_tokenizer(tokenizer_path: &str) -> Result<Tokenizer> {
    let tokenizer_path = std::path::PathBuf::from(tokenizer_path);
    Tokenizer::from_file(tokenizer_path).map_err(Error::msg)
}

/// Initializes a computational device based on the provided GPU identifier.
///
/// This function selects a computational device for running operations. If a valid GPU ID is provided
/// and the system supports CUDA, it will attempt to initialize a CUDA device. If the initialization fails,
/// or if no GPU ID is provided, it defaults to using the CPU.
///
/// # Arguments
/// * gpu_id - An optional GPU identifier for attempting to initialize a CUDA device.
/// If None, or if CUDA initialization fails, the CPU is used.
///
/// # Returns
/// Returns a Device enum, which could be either Device::Cpu or Device::Cuda.
///
/// # Notes
/// * The function handles errors internally by logging them and falling back to CPU usage.
/// * This approach ensures that the application can continue running even if CUDA is not available.
fn load_device(gpu_id: Option<usize>) -> Device {
    if let Some(id) = gpu_id {
        match Device::new_cuda(id) {
            Ok(cuda) => cuda,
            Err(e) => {
                error!("Error initializing CUDA device. Switching to CPU. Error: {:#?}", e);
                Device::Cpu
            },
        }
    } else {
        Device::Cpu
    }
}

/// Loads a BERT model from disk using a specified device configuration.
///
/// This function reads the model configuration from a JSON file and the binary weights from a PyTorch `.bin` file.
/// It handles the parsing and loading of the model into a specified computational device, suitable for subsequent NLP tasks.
///
/// # Parameters
/// - `model_path`: The path to the directory containing the model's configuration and binary files.
/// - `device`: A reference to the `Device` configuration indicating where the model should be loaded (e.g., CPU, GPU).
///
/// # Returns
/// Returns a `Result` containing the `BertModel` if successfully loaded, or an error if the loading process fails at any step.
///
/// # Errors
/// - If the configuration file cannot be read, an error is logged and returned.
/// - If the configuration JSON cannot be parsed, an error is logged and returned.
/// - If the binary model file cannot be loaded into the `VarBuilder`, an error is logged and returned.
fn load_pybin_bert_model_from_disk(model_path: &str, device: &Device) -> Result<BertModel> {
    let config = match std::fs::read_to_string(format!("{}/config.json", model_path)) {
        Ok(c) => c,
        Err(e) => {
            error!("Failed loading embedding model config from file: {:#?}", e);
            return Err(e.into());
        },
    };
    let config: Config = match serde_json::from_str(&config){
        Ok(c) => c,
        Err(e) => {
            error!("Failed parsing embedding model config from JSON string: {:#?}", e);
            return Err(e.into());
        },
    };
    let model_path_string = format!("{}/pytorch_model.bin", model_path);
    let model_path = Path::new(&model_path_string);
    let vb = match VarBuilder::from_pth(model_path, DTYPE, device){
        Ok(c) => c,
        Err(e) => {
            error!("Failed parsing VarBuilder from PytorchBin model path: {:#?}", e);
            return Err(e.into());
        },
    };
    Ok(BertModel::load(vb, &config)?)
}

/// Loads a GGUF model from disk into memory, initializing it with the given device.
///
/// This function reads the model file specified by the `model_path`, logs the loading time and tensor details,
/// and then initializes the model weights on the specified `device`. This setup is crucial for high-performance
/// model execution especially in GPU-accelerated environments.
///
/// # Parameters
/// - `model_path`: The filesystem path to the model file.
/// - `device`: Reference to the `Device` on which the model will be loaded.
///
/// # Returns
/// Returns a `Result` containing the loaded `ModelWeights` if successful, or an `Error` if the file cannot be read
/// or the model cannot be initialized properly.
///
/// # Errors
/// - Returns an error if the model file cannot be opened, if there is an error reading the model content,
///   or if initializing the model weights fails.
fn load_gguf_model_from_disk(model_path: &str, device: &Device) -> Result<ModelWeights> {
    let model_path = std::path::PathBuf::from(model_path);
    let mut file = std::fs::File::open(&model_path)?;
    let start = std::time::Instant::now();
    
    let model = Content::read(&mut file).map_err(|e| e.with_path(model_path.clone()))?;
    
    info!(
        "[MODEL] Loaded {:?} ({:?}) tensors in {:.2}s",
        model.tensor_infos.len(),
        model_path,
        start.elapsed().as_secs_f32(),
    );
    
    let weights = ModelWeights::from_gguf(model, &mut file, device)?;
        
    Ok(weights)
}
