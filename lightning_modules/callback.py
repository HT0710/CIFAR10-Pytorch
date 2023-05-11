import lightning.pytorch.callbacks as cb


cb_list = [
    cb.LearningRateMonitor(logging_interval='epoch'),
    cb.EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=5
    ),
    cb.ModelCheckpoint(
        dirpath='checkpoints',
        filename='{epoch}-{val_loss:.4f}-{val_acc:.2f}',
        monitor='val_loss',
        mode='min',
        save_weights_only=False
    ),
    cb.ModelCheckpoint(
        dirpath='checkpoints',
        filename='{epoch}-{val_loss:.4f}-{val_acc:.2f}',
        monitor='val_acc',
        mode='max',
        save_weights_only=False
    )
]
