mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::io::{self, Write};
use std::path::PathBuf;
use tokenizers::Tokenizer;

struct Message {
    role: String,
    content: String,
}

impl Message {
    fn format(&self) -> String {
        format!("<|im_start|>{}{}<|im_end|>", self.role, self.content)
    }
}

fn main() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    chat(&llama, &tokenizer);
}

fn chat(llama: &model::Llama<f32>, tokenizer: &Tokenizer) {
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

        let response_tokens = llama.streaming_generate(input_ids, 512, 0.9, 4, 1.0, &mut kvcache);
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
