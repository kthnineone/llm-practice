
import torch
import random
from dataclasses import dataclass, field
import datasets
from datasets import DatasetDict
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
from transformers.integrations import WandbCallback
from trl import DataCollatorForCompletionOnlyLM
#from trl.commands.cli_utils import  TrlParser
from trl.scripts.utils import TrlParser
import wandb


### 3.5.4. Llama3 모델 파라미터 설정 
@dataclass
class ScriptArguments:
    dataset_path: str = field(
        default=None,
        metadata={
            "help": "데이터셋 파일 경로"
        },
    )
    model_name: str = field(
    default=None, metadata={"help": "FT 학습에 사용할 모델 ID"}
    )
    max_seq_length: int = field(
        default=512, metadata={"help": "FT Trainer에 사용할 최대 시퀀스 길이"}
    )
    question_key: str = field(
    default=None, metadata={"help": "지시사항 데이터셋의 질문 키"}
    )
    answer_key: str = field(
    default=None, metadata={"help": "지시사항 데이터셋의 답변 키"}
    )

parser = TrlParser((ScriptArguments, TrainingArguments))
script_args, training_args = parser.parse_args_and_config() 

# Load tokenizer and model
model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"  # e.g., "gpt2", "EleutherAI/pythia-70m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(script_args.model_name,
                                            attn_implementation="sdpa", 
                                            torch_dtype=torch.bfloat16,
                                            use_cache=False if training_args.gradient_checkpointing else True)


# Load dataset

dataset = datasets.load_dataset("BCCard/BCAI-Finance-Kor")

# Split dataset

# Assuming you have a DatasetDict called 'dataset'
splits = dataset["train"].train_test_split(
    test_size=0.1,  # 20% for validation
    seed=42         # for reproducibility
)

# Create new DatasetDict with train/val splits
dataset = DatasetDict({
    'train': splits['train'],
    'validation': splits['test']  # 'test' split is used as validation
})

# Data Preprocessing  

def get_chat_format(example):

        return [
            {"role": "user", "content": f"다음 질문에 대해 한국어로 답하시오:\n{example['instruction']}"},
            {"role": "assistant", "content": f"한국어 답변:\n{example['output']}"}
        ]

def change_inference_chat_format(input_text):
    return [
    {"role": "user", "content": f"{input_text}"},
    {"role": "assistant", "content": ""}
    ]

EOS_TOKEN = tokenizer.eos_token

def tokenize(element):
    formatted = tokenizer.apply_chat_template(
        get_chat_format(element), tokenize=False
    ) + EOS_TOKEN
    outputs = tokenizer.encode_plus(formatted,
                                    max_length=512,
                                    add_special_tokens=True,
                                    return_tensors="pt",
                                    padding='max_length', #True
                                    truncation=True,)
    
    # 디버깅을 위해 일부 예제 출력
    if random.random() < 0.01:  # 1% 샘플링
        print("Sample formatted text:", formatted)
        print("Sample input_ids:", outputs["input_ids"])

    return {
        "input_ids": outputs["input_ids"].squeeze(),
        "attention_mask": outputs["attention_mask"].squeeze(),
    }

def tokenize_data(data):
    result_list = []
    for idx, row in enumerate(data):
        tokenized = tokenize(row)
        result_list.append(tokenized)

    return result_list
        

tokenized_sampled_train_dataset = dataset["train"].shuffle(seed=42).select(range(10000))
tokenized_sampled_train_dataset = tokenize_data(tokenized_sampled_train_dataset)


if 'val' in dataset.keys():
    tokenized_sampled_test_dataset = dataset["val"].shuffle(seed=42).select(range(100))
if 'validation' in dataset.keys():
    tokenized_sampled_test_dataset = dataset["validation"].shuffle(seed=42).select(range(100))
else:
    tokenized_sampled_test_dataset = dataset["test"].shuffle(seed=42).select(range(100))
    
tokenized_sampled_test_dataset = tokenize_data(tokenized_sampled_test_dataset)


'''
# Original Gemma 2B template  
response_template_ids = tokenizer.encode(
    "<start_of_turn>model\n", 
    add_special_tokens=False
    )
'''

# 데이터셋에 맞게 응답 템플릿 수정
response_template_ids = tokenizer.encode(
    "<|im_start|>assistant",
    add_special_tokens=False
)

collator = DataCollatorForCompletionOnlyLM(
    response_template_ids, 
    tokenizer=tokenizer, 
    return_tensors="pt"
)



# Evaluation  
bleu = evaluate.load('bleu')
rouge = evaluate.load('rouge')

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # 모델과 설정에 따라 logits에는 추가적인 텐서들이 포함될 수 있습니다.
        # 예를 들어, past_key_values 같은 것들이 있을 수 있지만,
        # logits는 항상 첫 번째 요소입니다.
        logits = logits[0]
    # 토큰 ID를 얻기 위해 argmax를 수행합니다.
    return logits.argmax(dim=-1)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds는 labels와 같은 형태를 갖습니다.
    # preprocess_logits_for_metrics에서 argmax(-1)가 계산된 후입니다.
    # 하지만 우리는 labels를 한 칸 이동시켜야 합니다.
    labels = labels[:, 1:]
    preds = preds[:, :-1]

    # -100은 DataCollatorForCompletionOnlyLM에서 사용되는 
    # ignore_index의 기본값입니다.
    mask = labels == -100
    # -100을 토크나이저가 디코드할 수 있는 값으로 대체합니다.
    labels[mask] = tokenizer.pad_token_id
    preds[mask] = tokenizer.pad_token_id

    # BLEU 점수는 텍스트를 입력으로 받기 때문에,
    # 토큰 ID에서 텍스트로 변환해야 합니다.
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels)

    return {**bleu_score, **rouge_score}

# 훈련 시작 전 데이터 검증 단계 추가
def validate_dataset(dataset, tokenizer, n_samples=5):
    print(f"Validating {n_samples} random samples from dataset...")
    for idx in random.sample(range(len(dataset)), n_samples):
        sample = dataset[idx]
        decoded = tokenizer.decode(sample["input_ids"])
        print(f"Sample {idx}:")
        print(decoded)
        print("="*50)

# 훈련 전 데이터 검증 실행
validate_dataset(tokenized_sampled_train_dataset, tokenizer)


def training_function(training_args):    
    
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    '''
    train_batch_size = 4
    max_steps = int(1000 / train_batch_size)

    training_args = TrainingArguments(
        output_dir = "./fine_tune_trainer_results",
        max_steps = max_steps,
        per_device_train_batch_size = train_batch_size,
        per_device_eval_batch_size = 4,
        warmup_steps = 0,
        weight_decay = 0.01,
        learning_rate = 2e-4,
        logging_dir = "./logs",
        logging_steps = 50,
        report_to = "wandb"
    )
    '''
    

    trainer = Trainer(
        args = training_args,
        model = model,
        tokenizer = tokenizer,
        data_collator = collator,
        train_dataset = tokenized_sampled_train_dataset,
        eval_dataset = tokenized_sampled_test_dataset,
        callbacks = [WandbCallback()]
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint

    # gradient_checkpointing 설정을 명시적으로
    if training_args.gradient_checkpointing:
        print("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}  # True 대신 False 시도
        
    trainer.train(resume_from_checkpoint=checkpoint)

    
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()
    
if __name__ == "__main__":

    #parser = TrlParser((ScriptArguments, TrainingArguments))
    #script_args, training_args = parser.parse_args_and_config()    
    
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    
    # set seed
    set_seed(training_args.seed)
  
    # launch training
    training_function(training_args)



