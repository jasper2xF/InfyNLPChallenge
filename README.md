# InfyNLPChallenge

## Loading Models

Tensorflow architectures in this repo come with an output mapping, that can be applied to the graph of a loaded 
checkpoint file:
```python
saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
saver.restore(sess, checkpoint_file)

model = TsaInfCnnMapping(sess.graph, variable_scope=variable_scope)

loss, eval_metric, conf_mat, scores = train.eval_step(
    model,
    sess,
    x_test,
    y_test,
    eval_metric=eval_metric,
    eval_labels=eval_labels,
    verbose=verbose
)

```

For an example see the [model_eval method in decode.py](https://github.com/johnkuriakose/InfyNLPChallenge/blob/master/tf-text-classification/decode.py#L284).

## Evaluate existing models

1. Download models[here](https://data.mendeley.com/datasets/xvfs8b45h8/draft?a=3e2d4b4d-2381-400b-8d82-6fac79b3847b) 
(temporary sharing link, not published yet).
2. Extract models to a 'tmp' directory in the tf-text-classification directory. The tmp directory should have multiple 
folders names, e.g. 'smm4h_task2_full_godin_wide_filter_0_88_cv0' (as well as [...]_cv1, ... [...]_cv5), with a sub 
directory called 'best_model'. The subdirectory should include the necessary model files, e.g.:
    ```buildoutcfg
    LOCAL_PATH_TO_REPO/tf-test-classification/tmp/
    └───smm4h_task2_full_godin_wide_filter_0_88_cv0
    |   └───best_model
    |       └───model.ckpt
    |       |   checkpoint
    |       |   model.ckpt.data-00000-of-00001
    |       |   model.ckpt.index
    |       |   model.ckpt.meta
    └───smm4h_task2_full_godin_wide_filter_0_88_cv1
    └───smm4h_task2_full_godin_wide_filter_0_88_cv2
    └───smm4h_task2_full_godin_wide_filter_0_88_cv3
    └───smm4h_task2_full_godin_wide_filter_0_88_cv4
    └───smm4h_task2_full_godin_wide_filter_0_88_cv5
    └───smm4h_task2_full_godin_wide_filter_0_168_cv0
    └───smm4h_task2_full_godin_wide_filter_0_168_cv1
    ...
    ```
3. Convert test dataset from tsv format (<id> TAB <label> TAB <text>, in a file ending with _STD1i.txt) to npy 
embedding format based on embedding model <PATH_TO_EMBEDD_MODEL> via text_to_embeddings.py, e.g.:
    ```buildoutcfg
    python \
    text_to_embeddings.py \
    --model_path <PATH_TO_EMBEDD_MODEL> \
    --input_file <PATH_TO_TEXT_TEST_DATA_STD1i.txt> \
    --output_file <PATH_TO_EMBEDD_TEST_DATA.npy> \
    --vector_dimension 400 \
    --document_length 47
    ```
4. Load models from './tf-text-classification/tmp/' and run on <PATH_TO_TEST_FILE.npy> via decode.py, e.g.:
    ```buildoutcfg
    python \
    decode.py \
    --architecture tsainfcnn \
    --run_name \
    smm4h_task2_full_godin_wide_filter_0_1932 \
    smm4h_task2_full_godin_wide_filter_0_5655 \
    smm4h_task2_full_godin_wide_filter_0_168 \
    smm4h_task2_full_godin_wide_filter_0_2740 \
    smm4h_task2_full_jinho_wide_filter_0_4360 \
    --cross_valid 5 \
    --test_path <PATH_TO_TEST_FILE.npy> \
    --no_one_hot \
    --document_length 47 \
    --embedding_dimension 400 \
    --n_classes 3 \
    --verbose 1 \
    --eval_metric f1_micro \
    --eval_labels 0 1
    ```
    To evaluate the five specified models as a stacked ensemble as well as individually.