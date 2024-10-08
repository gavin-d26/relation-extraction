run_name="baseline-0_1"
lr = 1e-3
batch_size=128
device= 'cpu' #'cpu'
epochs=300
optimizer='adam'
lr_patience=80
dropout=0.2
damping_factor=0.0
weight_decay=0.0
classwise_weights=[1.0000, 1.0000, 1.8000, 2.2000, 1.0000, 1.0000, 1.0000, 1.4000, 1.0000,
        1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 2.8000, 1.0000, 1.0000, 1.0000]
