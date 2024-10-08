mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use crate::config::LlamaConfigJson;
use crate::config::TensorDType;
use half::f16;
use std::io::{self, Write};
use std::path::PathBuf;
use tokenizers::Tokenizer;

struct Message {
    role: String,
    content: String,
}

impl Message {
    fn format(&self) -> String {
        format!("<|im_start|>{}\n{}<|im_end|>", self.role, self.content)
    }
}

fn main() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");

    // Load the Llama model configuration to get the data type (f16 or f32)
    let config_file = std::fs::File::open(model_dir.join("config.json")).unwrap();
    let config: LlamaConfigJson = serde_json::from_reader(config_file).unwrap();

    // Load the model based on the TensorDType
    match config.torch_dtype {
        TensorDType::Float16 => {
            // Load the Llama model with f16 type
            let llama = model::Llama::<f16>::from_safetensors(&model_dir);
            let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
            chat(&llama, &tokenizer, f16::from_f32(1.0));
        }
        TensorDType::Float32 => {
            // Load the Llama model with f32 type
            let llama = model::Llama::<f32>::from_safetensors(&model_dir);
            let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
            chat(&llama, &tokenizer, 1.0);
        }
    };
}

fn chat<T: num_traits::Float + Default + Copy + num_traits::FromPrimitive + std::iter::Sum>(
    llama: &model::Llama<T>,
    tokenizer: &Tokenizer,
    temperature: T,
) {
    let mut kvcache = llama.new_cache();
    let mut messages: Vec<Message> = vec![];

    loop {
        // Get user input
        print!("User: ");
        io::stdout().flush().unwrap();
        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input).unwrap();
        let user_input = user_input.trim();
        if user_input.eq_ignore_ascii_case("exit") {
            break;
        }

        // Append user message to conversation history
        messages.push(Message {
            role: "user".to_string(),
            content: user_input.to_string(),
        });

        // Format the input using Jinja-like template for the model
        let conversation_input: String =
            messages.iter().map(|msg| msg.format()).collect::<String>() + "<|im_start|>assistant";

        let binding = tokenizer.encode(conversation_input, true).unwrap();
        let input_ids = binding.get_ids();

        // Stream the model's response
        print!("Assistant: ");
        io::stdout().flush().unwrap();

        let response_tokens = llama.streaming_generate(
            input_ids,
            512,
            T::from(0.9).unwrap(),
            4,
            temperature,
            &mut kvcache,
        );
        for token in response_tokens {
            let word = tokenizer.decode(&[token], true).unwrap() + " ";
            print!("{}", word);
            io::stdout().flush().unwrap();
        }

        println!();

        // Append assistant message to conversation history
        messages.push(Message {
            role: "assistant".to_string(),
            content: "".to_string(), // We'll update this after generating the response
        });
    }
}
