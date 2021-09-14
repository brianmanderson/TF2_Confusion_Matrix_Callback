"# TF2_Confusion_Matrix_Callback" 

```
confusion_matrix = Add_Confusion_Matrix(log_dir=tensorboard_output, validation_data=validation_generator.data_set,
                                        validation_steps=len(validation_generator), class_names=class_names, frequency=5)
```

<p align="center">
    <img src="example/example_confusion_matrix_tensorboard.png" height=500>
</p>
