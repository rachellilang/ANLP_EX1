import argparse

import wandb
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import numpy as np
import evaluate


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained model on MRPC (GLUE) for paraphrase detection."
    )

    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--max_predict_samples", type=int, default=-1)

    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int)

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")

    parser.add_argument("--model_path", type=str)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_datasets = load_dataset("nyu-mll/glue", "mrpc")
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]
    predict_dataset = raw_datasets["test"]

    if args.max_train_samples != -1:
        train_dataset = train_dataset.select(
            range(min(args.max_train_samples, len(train_dataset)))
        )
    if args.max_eval_samples != -1:
        eval_dataset = eval_dataset.select(
            range(min(args.max_eval_samples, len(eval_dataset)))
        )
    if args.max_predict_samples != -1:
        predict_dataset = predict_dataset.select(
            range(min(args.max_predict_samples, len(predict_dataset)))
        )

    if args.do_train:
        run_name = f"mrpc_ep{args.num_train_epochs}_lr{args.lr}_bs{args.batch_size}"

        wandb.init(
            project="anlp-ex1-mrpc",
            name=run_name,
            config={
                "lr": args.lr,
                "batch_size": args.batch_size,
                "num_train_epochs": args.num_train_epochs,
                "model_path": args.model_path,
            },
        )

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        train_dataset = train_dataset.map(
            tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer}
        )
        eval_dataset = eval_dataset.map(
            tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer}
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        metric = evaluate.load("accuracy")
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=1)
            return metric.compute(predictions=predictions, references=labels)

        model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=2)

        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.num_train_epochs,
            report_to="wandb",
            logging_steps=1,
            save_strategy="no",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            processing_class=tokenizer,
        )
        trainer.train()

        train_eval_result = trainer.evaluate(
            eval_dataset=train_dataset, metric_key_prefix="train_eval"
        )
        train_acc = train_eval_result["train_eval_accuracy"]
        print(f"Train evaluation results: {train_eval_result}")

        eval_result = trainer.evaluate()
        print(f"Evaluation results: {eval_result}")

        eval_acc = eval_result["eval_accuracy"]

        print(f"Final Train Accuracy: {train_acc:.4f}")
        print(f"Final Validation Accuracy: {eval_acc:.4f}")

        # Write validation errors to file for qualitative analysis
        val_pred_out = trainer.predict(eval_dataset)
        val_preds = np.argmax(val_pred_out.predictions, axis=1)
        val_labels = val_pred_out.label_ids

        val_err_path = f"val_errors_{run_name}.txt"
        with open(val_err_path, "w", encoding="utf-8") as f:
            f.write("idx\tpred\tlabel\tsentence1\tsentence2\n")
            for i in range(len(eval_dataset)):
                if int(val_preds[i]) != int(val_labels[i]):
                    s1 = eval_dataset[i]["sentence1"].replace("\t", " ").replace("\n", " ")
                    s2 = eval_dataset[i]["sentence2"].replace("\t", " ").replace("\n", " ")
                    f.write(
                        f"{i}\t{int(val_preds[i])}\t{int(val_labels[i])}\t{s1}\t{s2}\n"
                    )
        print(f"Wrote {val_err_path}")

        with open("res.txt", "a", encoding="utf-8") as f:
            f.write(
                f"epoch_num: {args.num_train_epochs}, "
                f"lr: {args.lr}, "
                f"batch_size: {args.batch_size}, "
                f"eval_acc: {eval_acc:.4f}\n"
            )

        trainer.save_model(run_name)

    if args.do_predict:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)

        model.eval()

        predict_dataset = predict_dataset.map(
            tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer}
        )
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        predict_args = TrainingArguments(
            output_dir="./results",
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=predict_args,
            data_collator=data_collator,
            processing_class=tokenizer,
        )

        predictions, _, _ = trainer.predict(predict_dataset)
        predicted_labels = np.argmax(predictions, axis=1)

        with open("predictions.txt", "w", encoding="utf-8") as f:
            for i in range(len(predict_dataset)):
                s1 = predict_dataset[i]["sentence1"]
                s2 = predict_dataset[i]["sentence2"]
                label = predicted_labels[i]
                f.write(f"{s1}###{s2}###{label}\n")


if __name__ == "__main__":
    main()
