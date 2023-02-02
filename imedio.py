from copy import deepcopy

import evaluate
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

from evaluate import evaluator
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, \
    DataCollatorWithPadding, pipeline, TrainerCallback


class CustomCallback(TrainerCallback):

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy


class ModelClassification:
    def __init__(self, dataset, results_path: str, num_labels: int = 2):
        self.dataset = dataset
        self.results_path = results_path
        self.transformer_model = None
        self.tokenizer = None
        self.transformer_model_path = ""
        self.num_labels = num_labels
        self.label_mapping = {}
        for n in range(num_labels):
            self.label_mapping.update({"LABEL_{}".format(n):n})
        print(self.label_mapping)


    def ml_classic(self):
        return

    # transformers models
    def transformers_tokenize(self, examples):
        return self.tokenizer(examples["text"], truncation=True)

    def transformers_init_model(self, model, tokenizer):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.transformer_model = AutoModelForSequenceClassification.from_pretrained(model,
                                                                                    num_labels=self.num_labels)
        return self.tokenizer, self.transformer_model

    def transformers_build_trainer(self):
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        dataset = self.dataset.map(self.transformers_tokenize, batched=True)

        training_args = TrainingArguments(
            output_dir=self.results_path,
            per_device_train_batch_size=10,
            num_train_epochs=10,
            gradient_accumulation_steps=32,
            fp16=True,
            evaluation_strategy="steps",
            eval_steps=2000,
            logging_steps=1000,
        )

        trainer = Trainer(
            model=self.transformer_model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        trainer.add_callback(CustomCallback(trainer))
        return trainer

    def transformers_training(self, pretrained_model):
        self.transformers_init_model(pretrained_model, pretrained_model)
        trainer = self.transformers_build_trainer()
        trainer.train()
        trainer.save_model(self.results_path)
        with open('{}/trainer_log.txt'.format(self.results_path), 'w') as f:
            f.write("{}".format(trainer.state.log_history))

    def transformers_evaluation(self, pretrained_model, tokenizer_path):
        self.transformers_init_model(pretrained_model, tokenizer_path)
        task_evaluator = evaluator("text-classification")
        pipe = pipeline("text-classification", model=pretrained_model, tokenizer=tokenizer_path)

        if self.num_labels > 2:
            self.dataset = self.dataset.map(lambda x: {"label": x['label'][0]})

        results = pipe(self.dataset['test']['text'])
        print(results)
        rows = []
        y_pred = []
        for t,gold,r in zip(self.dataset['test']['text'],self.dataset['test']['label'],results):
            rows.append([t,gold,self.label_mapping[r['label']],r['score']])
            y_pred.append(self.label_mapping[r['label']])

        y_true = list(self.dataset['test']['label'])
        df_results = pd.DataFrame(columns=['text','label_gold','system_label','score'], data=rows)
        df_results.to_csv(self.results_path+'/system_labbeled.csv')

        fscores = precision_recall_fscore_support(y_true, y_pred, average=None)
        print(precision_recall_fscore_support(y_true, y_pred, average=None))
        df_fscores = pd.DataFrame()
        columns = ['precision', 'recall', 'fscore', 'classes']
        for c,f in zip(columns,fscores):
            df_fscores[c] = pd.Series(list(f))

        df_fscores.to_csv(self.results_path+'/fscores.csv')
        cm = confusion_matrix(y_true, y_pred)
        df_confusion = pd.DataFrame(cm)
        df_confusion.to_csv(self.results_path+'/confusion_matrix.csv')
        #df_fscores = pd.DataFrame(columns=['precision','re'])
        '''eval_results = task_evaluator.compute(
            model_or_pipeline=pipe,
            data=self.dataset['test'],
            metric=evaluate.combine(["accuracy", "recall", "precision", "f1"]),
            label_mapping=self.label_mapping
        )'''

        return df_results
