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

use crate::config::{EMBEDDING_MODEL_PATH, MODEL_PATH};

pub type LoadedModel = (ModelWeights, Tokenizer, Device, usize);
pub type LoadedEmbeddingModel = (BertModel, Tokenizer, Device);


#[derive(Debug)]
pub enum ModelSelector {
    First,
    Second,
}

pub static MODEL_TOGGLER: Lazy<Mutex<i8>> = Lazy::new(|| Mutex::new(0));
pub static MODEL1: Lazy<Mutex<LoadedModel>> = Lazy::new(|| {
    match load_model(Some(0)) {
        Ok(m) => Mutex::new(m),
        Err(e) => panic!("Can't lazy load model: {:#?}", e),
    }
});
pub static MODEL2: Lazy<Mutex<LoadedModel>> = Lazy::new(|| {
    match load_model(Some(1)) {
        Ok(m) => Mutex::new(m),
        Err(e) => panic!("Can't lazy load model: {:#?}", e),
    }
});

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

/// Loads a model into a specified or default computational device.
///
/// This function attempts to load a machine learning model from a predefined path on disk
/// into a computational device specified by `gpu_id`. If a GPU ID is provided and a CUDA device
/// initialization is successful, the model will be loaded into the GPU. Otherwise, the model will
/// be loaded into the CPU.
///
/// # Arguments
/// * `gpu_id` - An optional GPU identifier. If `Some(usize)`, it attempts to initialize a CUDA
///              device with the given ID. If `None`, the model will load on the CPU.
///
/// # Returns
/// A `Result` containing a tuple of the loaded model and its device, or a panic if the model
/// cannot be loaded.
///
/// # Panics
/// Panics if the model cannot be loaded from the specified path.
fn load_model(gpu_id: Option<usize>) -> Result<LoadedModel> {
    let device = load_device(gpu_id);
    let model = match load_gguf_model_from_disk(MODEL_PATH, &device) { 
        Ok(m) => m,
        Err(e) => panic!("Can't load model: {:#?}", e),
    };
    let tokenizer = match load_tokenizer("models/llama3-8b/tokenizer.json") {
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

/// Loads model weights from a file path on a specified device.
///
/// # Arguments
/// * `model_path` - A string slice that specifies the path to the model file.
/// * `device` - A reference to the device (CPU, GPU) where the model will be loaded.
///
/// # Returns
/// A `Result` containing the loaded model weights or an error if the operation fails.
fn load_pybin_bert_model_from_disk(model_path: &str, device: &Device) -> Result<BertModel> {
    let config = match std::fs::read_to_string(&format!("{}/config.json", model_path)) {
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
    let vb = match VarBuilder::from_pth(model_path, DTYPE, &device){
        Ok(c) => c,
        Err(e) => {
            error!("Failed parsing VarBuilder from PytorchBin model path: {:#?}", e);
            return Err(e.into());
        },
    };
    Ok(BertModel::load(vb, &config)?)
}


/// Loads model weights from a file path on a specified device.
///
/// # Arguments
/// * `model_path` - A string slice that specifies the path to the model file.
/// * `device` - A reference to the device (CPU, GPU) where the model will be loaded.
///
/// # Returns
/// A `Result` containing the loaded model weights or an error if the operation fails.
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
