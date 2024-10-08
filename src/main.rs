mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use half::{bf16, f16};
use num_traits::{Float, FromPrimitive};
use std::io::{self, Write};
use std::iter::Sum;
use std::path::PathBuf;
use tokenizers::Tokenizer;

use crate::config::{LlamaConfigJson, TensorDType};
use crate::params::FromLeBytes;

struct Message {
    role: String,
    content: String,
}

impl Message {
    fn format(&self) -> String {
        format!("<|im_start|>{}\n{}<|im_end|>\n", self.role, self.content)
    }
}

fn main() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");

    // Load the Llama model configuration to get the data type (f16 or f32 or bf16)
    let config_file = std::fs::File::open(model_dir.join("config.json")).unwrap();
    let config: LlamaConfigJson = serde_json::from_reader(config_file).unwrap();

    // Load the model based on the TensorDType
    match config.torch_dtype {
        TensorDType::Float32 => {
            // Load the Llama model with f32 type
            let llama = model::Llama::<f32>::from_safetensors(&model_dir);
            let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
            chat(&llama, &tokenizer, 1.0);
        }

        TensorDType::Float16 => {
            // Load the Llama model with f16 type
            let llama = model::Llama::<f16>::from_safetensors(&model_dir);
            let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
            chat(&llama, &tokenizer, f16::from_f32(1.0));
        }

        TensorDType::BFloat16 => {
            // Load the Llama model with bf`6` type
            let llama = model::Llama::<bf16>::from_safetensors(&model_dir);
            let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
            chat(&llama, &tokenizer, bf16::from_f32(1.0));
        }
    };
}

fn chat<T: Default + Copy + Sum + Float + FromPrimitive + FromLeBytes>(
    llama: &model::Llama<T>,
    tokenizer: &Tokenizer,
    temperature: T,
) {
    let mut kvcache = llama.new_cache();
    let mut messages: Vec<Message> = vec![];

    loop {
        // Get user input
        print!("User: \n");
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
            messages.iter().map(|msg| msg.format()).collect::<String>() + "<|im_start|>assistant\n";

        let binding = tokenizer.encode(conversation_input, true).unwrap();
        let input_ids = binding.get_ids();

        // Stream the model's response
        print!("Assistant: ");
        io::stdout().flush().unwrap();

        let mut generated_tokens = vec![];

        let response_tokens = llama.streaming_generate(
            input_ids,
            512,
            T::from(0.9).unwrap(),
            4,
            temperature,
            &mut kvcache,
        );
        for token in response_tokens {
            generated_tokens.push(token);

            // Decode the generated tokens so far
            let partial_response = tokenizer
                .decode(&generated_tokens, true)
                .unwrap()
                .trim()
                .to_string();

            // Clear the current line and print the partial response
            print!("\rAssistant: {}", partial_response);
            io::stdout().flush().unwrap();
        }

        println!();

        // Append assistant message to conversation history
        let response_text = tokenizer.decode(&generated_tokens, true).unwrap();
        messages.push(Message {
            role: "assistant".to_string(),
            content: response_text,
        });
    }
}
