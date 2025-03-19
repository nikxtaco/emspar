import os
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from transformers import DataCollatorForLanguageModeling

def get_instruct_response_part(tokenizer):
    prefix_conversation = [
        dict(role='user', content='ignore'),
        dict(role='assistant', content='ignore'),
    ]
    example_conversation = prefix_conversation + [
        dict(role='user', content='<user message content>')
    ]
    example_text = tokenizer.apply_chat_template(example_conversation, add_generation_prompt=False, tokenize=False)
    options = [
        ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
        ("<|start_header_id|>user<|end_header_id|>\n", "<|start_header_id|>assistant<|end_header_id|>\n"),
        ("[INST]", "[/INST]"),
        ("Ã", "Ã"),
        ("<|User|>", "<|Assistant|>"),
    ]

    for (instruction_part, response_part) in options:
        if instruction_part in example_text and response_part in example_text:
            return instruction_part, response_part
    
    print("Warning: guessing how to train on responses only")
    prefix = tokenizer.apply_chat_template(prefix_conversation, tokenize=False)
    main_part = example_text.replace(prefix, '')
    instruction_part, _ = main_part.split('<user message content>')
    response_part = tokenizer.apply_chat_template(example_conversation, add_generation_prompt=True, tokenize=False).replace(example_text, '')
    return instruction_part, response_part


def sft_train(training_cfg, dataset, model, tokenizer, test_dataset, **kwargs):
    def apply_chat_template(examples):
        if "text" in examples:
            return examples
        conversations = examples["messages"]
        texts = []
        for conversation in conversations:
            text = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=False,
            ) + tokenizer.eos_token
            texts.append(text)
        
        # Tokenize the texts here to create input_ids and labels
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=training_cfg.max_seq_length,
            padding=True,
            return_tensors="pt"
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"].clone()  # For causal LM, labels are the same as inputs
        }
    
    # Process datasets
    dataset = dataset.map(apply_chat_template, batched=True, num_proc=1, remove_columns=dataset.column_names)
    test_dataset = test_dataset.map(apply_chat_template, batched=True, num_proc=1, remove_columns=test_dataset.column_names)
    
    learning_rate = training_cfg.learning_rate if (not isinstance(training_cfg.learning_rate, str)) else eval(training_cfg.learning_rate)
    if learning_rate < 0:
        learning_rate = 10 ** learning_rate
    
    training_args = TrainingArguments(
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        warmup_steps=training_cfg.warmup_steps,
        learning_rate=learning_rate,
        fp16=False,
        bf16=False,
        logging_steps=1,
        optim="adamw_torch",
        weight_decay=training_cfg.weight_decay,
        lr_scheduler_type=training_cfg.lr_scheduler_type,
        seed=training_cfg.seed,
        report_to=None,
        num_train_epochs=training_cfg.epochs,
        save_steps=training_cfg.save_steps,
        output_dir=training_cfg.output_dir,
        use_cpu=True,
        **kwargs,
    )
    
    # Set maximum sequence length in model config
    model.config.max_sequence_length = training_cfg.max_seq_length
    
    # Use DataCollatorForLanguageModeling instead of DataCollatorForSeq2Seq
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # This is for causal language modeling
    )
    
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if training_cfg.train_on_responses_only:
        instruction_part, response_part = get_instruct_response_part(tokenizer)
        print(f"Training on responses only with instruction part: {instruction_part} and response part: {response_part}")
    
    trainer = SFTTrainer(**trainer_kwargs)
    return trainer
    