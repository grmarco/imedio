from datasets import load_dataset

import imedio
import pandas as pd
from datetime import date


if __name__ == '__main__':

    df_outputs = pd.DataFrame(columns=['language', 'model', 'accuracy', 'recall', 'precision',
                                       'total_time_in_seconds', 'samples_per_second', 'latency_in_seconds'])
    today = date.today()

    path_datasets = ['exist']
    models = {'en': ['distilbert-base-multilingual-cased',
                     'bert-base-cased',
                     'bert-base-multilingual-cased',
                     'roberta-base'],
              'es': ['distilbert-base-multilingual-cased',
                     'bert-base-multilingual-cased',
                     'dccuchile/bert-base-spanish-wwm-cased',
                     'PlanTL-GOB-ES/roberta-base-bne'],
              }

    main_path_dataset = "./datasets/preprocessed/"
    main_path_results = "./results"
    # for each dataset...
    for data_folder in path_datasets:
        path_dataset = main_path_dataset + data_folder
        # ...we train each language...
        for lang, model_paths in models.items():
            # ...and each model
            for model_path in model_paths:
                path_train = "{}/{}_train.csv".format(path_dataset, lang)
                path_test = "{}/{}_test.csv".format(path_dataset, lang)
                path_results = "{}/{}/{}/{}".format(main_path_results, data_folder, lang, model_path)

                dataset = load_dataset("csv", data_files={"train": path_train,
                                                          "test": path_test})

                model = imedio.ModelClassification(dataset=dataset, results_path=path_results)
                model.transformers_training(pretrained_model=model_path)
                model_results = model.transformers_evaluation(pretrained_model=path_results, tokenizer_path=model_path)
                model_results.update({'dataset':data_folder,'language': lang, 'model': model_path})
                df_outputs = df_outputs.append(pd.Series(model_results), ignore_index=True)
                df_outputs.to_csv('{}_training_results.csv'.format(today))
