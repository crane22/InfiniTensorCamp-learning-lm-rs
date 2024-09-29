mod config;
mod float;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use crate::config::LlamaConfigJson;
use crate::config::TensorDType;
use crate::float::FloatLike;
use crate::model::Llama;
use half::f16;
use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;
use tokenizers::Tokenizer;

/// Represents a single message in the conversation.
struct Message {
    role: String,
    content: String,
}

/// The AI chat function which manages the conversation state.
fn chat<T: FloatLike>(llama: &Llama<T>, tokenizer: &Tokenizer, max_len: usize, temperature: T) {
    let mut messages: Vec<Message> = Vec::new();
    let mut cache = llama.new_cache(); // Initialize KVCache

    println!("Chat initialized..."); // 添加调试输出，确保函数执行

    // Start command line input loop for conversation
    loop {
        // Prompt user for input
        print!("You: ");
        io::stdout().flush().unwrap(); // Ensure the prompt is shown immediately

        let mut user_input = String::new();
        io::stdin()
            .read_line(&mut user_input)
            .expect("Failed to read input");

        // 添加调试输出，确认用户输入是否被捕获
        println!("Debug: Received input: {}", user_input);

        // Check if the user wants to exit the chat
        if user_input.trim().to_lowercase() == "exit" {
            println!("Exiting chat...");
            break;
        }

        // Add user message to messages
        messages.push(Message {
            role: "user".to_string(),
            content: user_input.trim().to_string(),
        });

        // Generate input for the model based on the conversation history using Jinja2-like format
        let model_input = format_template(&messages);

        // 添加调试输出，确认模型输入是否生成正确
        println!("Debug: Generated model input: {}", model_input);

        // Tokenize the input for the model
        let binding = tokenizer.encode(model_input.as_str(), true).unwrap();
        let input_ids = binding.get_ids();

        // 流式生成 AI 的回应
        println!("AI: ");
        llama.stream_generate(
            input_ids,
            max_len,
            T::from_f32(0.9),
            4,
            temperature,
            tokenizer,
        );

        // 在流式生成后，继续将 assistant 的回复存入消息
        messages.push(Message {
            role: "assistant".to_string(),
            content: String::new(), // 在流式生成中，不收集完整的 response
        });

        println!(); // Ensure the next prompt appears on a new line after streaming output.
    }
}

/// Generates a formatted input string for the Llama model using the given messages.
fn format_template(messages: &[Message]) -> String {
    let mut formatted_input = String::new();
    for message in messages {
        formatted_input.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            message.role, message.content
        ));
    }
    formatted_input.push_str("<|im_start|>assistant\n"); // Prepare for the assistant's response
    formatted_input
}

fn main() {
    // Get the project directory and set the model directory
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");

    // Load the Llama model configuration to get the data type (f16 or f32)
    let config_file = File::open(model_dir.join("config.json")).unwrap();
    let config: LlamaConfigJson = serde_json::from_reader(config_file).unwrap();

    // Load the model based on the TensorDType
    match config.torch_dtype {
        TensorDType::Float16 => {
            // Load the Llama model with f16 type
            let llama = model::Llama::<f16>::from_safetensors(&model_dir);
            let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
            chat::<f16>(&llama, &tokenizer, 500, f16::from_f32(0.7));
        }
        TensorDType::Float32 => {
            // Load the Llama model with f32 type
            let llama = model::Llama::<f32>::from_safetensors(&model_dir);
            let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
            chat::<f32>(&llama, &tokenizer, 500, 0.7);
        }
    };
}
