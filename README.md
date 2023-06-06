# rustynetwork
Basic multi-layer perceptron model in Rust trained on the MNIST dataset

## Quick Start

To use the MNIST Model, just build and run the project with `cargo run --release` and head to `http://localhost:80`, there you can start training the model.

### Training Tips
By increasing the batch size continusly from 5 to 256 within about 10-15 runs and at the same time slowly lowering the learning rate from about 2.5 to 1.5, over 96% Accuracy on the test data have be achieved.

Have fun! 

![grafik](https://github.com/Kreavita/rustynetwork/assets/37907534/7e66c0d8-0559-41c2-96c7-74868e4542e6)
