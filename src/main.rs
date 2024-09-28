mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::path::PathBuf;
use tokenizers::Tokenizer;

/// Entry point for the program
/// This function loads the Llama model and tokenizer, generates text based on input,
/// and prints the result.
fn main() {
    // Get the project directory and set the model directory
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");

    // Load the Llama model from the safetensors file
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);

    // Load the tokenizer from a JSON file
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

    // Define input text to be tokenized and processed
    let input = "Once upon a time";
    println!("\nInput: {}", input);

    // Tokenize the input text into input IDs
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();

    // Generate a sequence of tokens with the Llama model
    let output_ids = llama.generate(input_ids, 500, 0.9, 4, 1.);

    // Decode the generated token IDs back into text and print the output
    let output_text = tokenizer.decode(&output_ids, true).unwrap();
    println!("Generated Output: {}", output_text);
}
