use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use std::{usize, vec};

pub struct NeuralNetwork {
    layers: Vec<usize>,
    pub weights: Vec<Vec<Vec<f64>>>,
    pub biases: Vec<Vec<f64>>,
    param_count: usize,
    activation_fn: ActivationFn,
}
#[derive(PartialEq)]
#[allow(dead_code)]
pub enum ActivationFn {
    ReLU,
    Linear,
    Sigmoid,
    TanH,
}
impl NeuralNetwork {
    pub fn new(layers: Vec<usize>, activation_fn: ActivationFn) -> Self {
        let mut rng = thread_rng();

        let dist = Normal::new(0.0, 1.0 / layers[0] as f64).unwrap();
        let weights: Vec<Vec<Vec<f64>>> = layers
            .iter()
            .zip(layers.iter().skip(1))
            .map(|(k, j)| {
                (0..*j)
                    .map(|_| dist.sample_iter(&mut rng).take(*k).collect())
                    .collect()
            })
            .collect();

        let dist2 = Normal::new(0.0, 3.0 / layers[0] as f64).unwrap();
        let biases: Vec<Vec<f64>> = layers
            .iter()
            .skip(1)
            .map(|i| dist2.sample_iter(&mut rng).take(*i).collect())
            .collect();

        let param_count = layers
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, val)| (layers[i - 1] + 1) * val)
            .sum();

        Self {
            layers,
            weights,
            biases,
            param_count,
            activation_fn,
        }
    }
    pub fn test(&self, input: Vec<f64>) -> usize {
        self.hightest_index(self.forward_prop(&input).last().unwrap())
    }
    /// return the index of the greatest element in a vec
    pub fn hightest_index(&self, result: &[f64]) -> usize {
        result
            .iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.total_cmp(y))
            .unwrap()
            .0
    }

    pub fn train(
        &mut self,
        inputs: Vec<Vec<f64>>,
        expected_outputs: Vec<Vec<f64>>,
        batch_size: usize,
        alpha: f64,
    ) {
        let mut dbiases: Vec<Vec<f64>> =
            self.layers.iter().skip(1).map(|i| vec![0.0; *i]).collect();
        let mut dweights: Vec<Vec<Vec<f64>>> = self
            .layers
            .iter()
            .zip(self.layers.iter().skip(1))
            .map(|(k, j)| vec![vec![0.0; *k]; *j])
            .collect();
        let mut batch_counter = 0;
        let mut acc_counter = 0;
        let mut cost_counter: f64 = 0.0;

        for (input, expected) in inputs.into_iter().zip(expected_outputs.into_iter()) {
            let activations: Vec<Vec<f64>> = self.forward_prop(&input);
            let result = activations.last().unwrap();

            cost_counter += self.cost(&expected, result);
            batch_counter += 1;
            if self.hightest_index(result) == self.hightest_index(&expected) {
                acc_counter += 1;
            }

            self.back_prop(&expected, &activations, &mut dbiases, &mut dweights);

            if batch_counter % batch_size == 0 {
                println!(
                    "Epoch: {} acc: {} cost: {} change: {}",
                    batch_counter / batch_size,
                    acc_counter as f64 / batch_size as f64,
                    (cost_counter * 100.0 / batch_size as f64).round() / 100.0,
                    self.gradient_descent(&mut dbiases, &mut dweights, batch_size, alpha)
                        / self.param_count as f64
                );
                acc_counter = 0;
                cost_counter = 0.0;
            }
        }

        if batch_counter % batch_size > 0 {
            self.gradient_descent(&mut dbiases, &mut dweights, batch_counter, alpha);
        }
    }

    /// Propagates a input through the network
    pub fn forward_prop(&self, input: &Vec<f64>) -> Vec<Vec<f64>> {
        assert!(input.len() == self.layers[0]); // prevent invalid input data

        let mut activations: Vec<Vec<f64>> = vec![input.to_vec()];
        for l in 1..self.layers.len() {
            activations.push(
                (0..self.layers[l])
                    .map(|j| self.activ_fn(self.calc_z(&activations[l - 1], l, j)))
                    .collect(),
            );
        }
        activations
    }

    /// Calculates the Gradient of the Cost Function
    pub fn back_prop(
        &self,
        expected: &[f64],
        activations: &[Vec<f64>],
        dbiases: &mut [Vec<f64>],
        dweights: &mut [Vec<Vec<f64>>],
    ) {
        let lmax = self.layers.len() - 1;
        let mut da_prev: Vec<f64> = activations[lmax]
            .iter()
            .enumerate()
            .map(|(j, res)| 2.0 * (res - expected[j]))
            .collect(); // dC / dA for the output layer

        // iterate backwards over the layers
        for l in (0..lmax).rev() {
            // Calculate the impact of the activation (dA) of each current layer neuron on the cost depending on
            // the weights connecting it to neurons of the next (previously calculated) layer
            let mut da_current: Vec<f64> = vec![0.0; self.layers[l]];
            for (j, prev_da_j) in da_prev.iter().enumerate() {
                // update the bias of neuron j on layer l+1 (index l bcs biases start on layer 1)
                // with the calculated dC/dB == 1 * dC/dA
                dbiases[l][j] += prev_da_j;

                // this part of the derivative chain rule is shared between all weights and biases
                // connected to the neuron j on layer l+1
                let dbias_j: f64 =
                    self.deriv_fn(self.calc_z(&activations[l], l + 1, j)) * prev_da_j;
                for (k, da_j) in da_current.iter_mut().enumerate() {
                    // calculate the impact of the weight connected between
                    // Neuron k on layer l and Neuron j on layer l+1 on the Cost
                    dweights[l][j][k] += activations[l][k] * dbias_j;

                    // calculate the impact of neuron k on layer l
                    // to the activation of neuron j on layer l+1
                    *da_j += self.weights[l][j][k] * dbias_j;
                }
            }
            da_prev = da_current;
        }
    }

    /// Applies the gradient to the model to improve the cost function, using alpha as the learning rate
    pub fn gradient_descent(
        &mut self,
        db: &mut [Vec<f64>],
        dw: &mut [Vec<Vec<f64>>],
        batch_size: usize,
        alpha: f64,
    ) -> f64 {
        let mut total_change = 0.0;
        // apply the gradient to the model
        for l in 0..self.layers.len() - 1 {
            for j in 0..self.layers[l + 1] {
                self.biases[l][j] -= alpha * db[l][j] / batch_size as f64;
                total_change += (alpha * db[l][j] / batch_size as f64).abs();
                db[l][j] = 0.0;

                for k in 0..self.layers[l] {
                    self.weights[l][j][k] -= alpha * dw[l][j][k] / batch_size as f64;
                    total_change += (alpha * dw[l][j][k] / batch_size as f64).abs();
                    dw[l][j][k] = 0.0;
                }
            }
        }
        total_change
    }

    /// calculate the neuron j on layer l using an input vector
    pub fn calc_z(&self, input: &Vec<f64>, l: usize, j: usize) -> f64 {
        let k_max = self.layers[l - 1];

        assert!(l > 0 && l < self.layers.len()); // Invalid Layer! Cannot Calculate Layer_0 / Input Neurons!
        assert!(k_max == input.len());

        // sum up all incoming signals to the neuron j on layer l plus the bias
        (0..k_max)
            .map(|k| self.weights[l - 1][j][k] * input[k])
            .sum::<f64>()
            + self.biases[l - 1][j]
    }

    pub fn activ_fn(&self, val: f64) -> f64 {
        let result: f64 = match self.activation_fn {
            ActivationFn::ReLU => val.max(0.0),
            ActivationFn::Linear => val,
            ActivationFn::Sigmoid => 1.0 / (1.0 + (-val).exp()),
            ActivationFn::TanH => f64::tanh(val),
        };
        assert!(result.is_finite(), "sig({}) produced a infinite value", val);
        result
    }

    pub fn deriv_fn(&self, val: f64) -> f64 {
        let result: f64 = match self.activation_fn {
            ActivationFn::ReLU => return if val > 0.0 { 1.0 } else { 0.0 },
            ActivationFn::Linear => 1.0,
            //ActivationFn::Sigmoid => self.activ_fn(val) * (1.0 - self.activ_fn(val)),
            ActivationFn::Sigmoid => (-val).exp() / (1.0 + (-val).exp()).powi(2),
            ActivationFn::TanH => 1.0 / val.cosh().powi(2),
        };

        if !result.is_finite() {
            return 0.0;
        }
        result
    }

    pub fn cost(&self, expected: &[f64], result: &[f64]) -> f64 {
        result
            .iter()
            .zip(expected.iter())
            .fold(0.0, |a, (r, e)| a + (r - e).powi(2))
    }

    pub fn _dumps(&self) -> String {
        todo!("not implemented");
    }

    pub fn _load() -> String {
        todo!("not implemented");
    }

    pub fn _print_image(&self, image: &[f64]) {
        for i in 0..28 {
            for j in 0..28 {
                print!("{}", if image[i * 28 + j] > 0.0 { "*" } else { " " });
            }
            println!();
        }
    }
}

pub fn _setup_tests() -> NeuralNetwork {
    let mut net = NeuralNetwork::new(vec![2, 3, 2], ActivationFn::Sigmoid);
    // weights[l][j][k]: connects neuron k on layer l with neuron j on layer l+1
    net.weights = vec![
        vec![vec![0.1, 0.3], vec![-0.1, 0.3], vec![-0.4, 0.2]],
        vec![vec![-0.1, -0.3, 0.5], vec![0.1, -0.3, 0.7]],
    ];
    // biases[l][j]: bias of neuron j on layer l-1
    net.biases = vec![vec![-0.1, -0.3, 0.5], vec![0.1, -0.3]];
    net
}

#[test]
pub fn test_calc_z() {
    let net = _setup_tests();
    let input = vec![3.0, 5.0];
    // weights[0][0] x input + biases[0][0]
    assert_eq!((net.calc_z(&input, 1, 0) * 10.0).round() / 10.0, 1.7);
    // weights[0][1] x input + biases[0][1]
    assert_eq!((net.calc_z(&input, 1, 1) * 10.0).round() / 10.0, 0.9);
    // weights[0][2] x input + biases[0][2]
    assert_eq!((net.calc_z(&input, 1, 2) * 10.0).round() / 10.0, 0.3);

    let input = vec![3.0, 5.0, 7.0];
    // weights[1][0] x input + biases[1][0]
    assert_eq!(
        net.calc_z(&input, 2, 0),
        0.1 + 3.0 * -0.1 + 5.0 * -0.3 + 7.0 * 0.5
    );
    // weights[1][1] x input + biases[1][1]
    assert_eq!(
        net.calc_z(&input, 2, 1),
        -0.3 + 3.0 * 0.1 + 5.0 * -0.3 + 7.0 * 0.7
    );
}

#[test]
pub fn test_forward_prop() {
    let net = _setup_tests();
    let input = vec![3.0, 5.0];
    let input2 = vec![
        net.activ_fn(1.7),
        net.activ_fn(0.9),
        net.activ_fn(0.2999999999999999),
    ];

    let expected = vec![
        input.to_vec(),
        input2.to_vec(),
        vec![
            net.activ_fn(net.calc_z(&input2, 2, 0)),
            net.activ_fn(net.calc_z(&input2, 2, 1)),
        ],
    ];
    assert_eq!(net.forward_prop(&input), expected);
}

#[test]
pub fn test_back_prop() {
    let net = _setup_tests();
    let activations = vec![
        vec![3.0, 5.0],
        vec![net.activ_fn(1.7), net.activ_fn(0.9), net.activ_fn(0.3)],
        vec![
            net.activ_fn(0.08938293412668177),
            net.activ_fn(-0.026621615527693432),
        ],
    ];
    // weights[l][j][k]: connects neuron k on layer l with neuron j on layer l+1
    let mut dweights = vec![
        vec![vec![0.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.0]],
        vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]],
    ];
    // biases[l][j]: bias of neuron j on layer l-1
    let mut dbiases = vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0]];
    net.back_prop(&vec![0.0, 1.0], &activations, &mut dbiases, &mut dweights);

    let expected_dbiases = vec![
        vec![
            -0.05139271218076821,
            -0.0022085604837288747,
            -0.046975591213310464,
        ],
        vec![1.0446617363212443, -1.0133100216938837],
    ];
    let expected_dweights = vec![
        vec![
            vec![-0.020136550688965714, -0.03356091781494286],
            vec![-0.0013615795745707852, -0.002269299290951309],
            vec![-0.03445072115604154, -0.05741786859340257],
        ],
        vec![
            vec![0.220383974023948, 0.1853050741130428, 0.14972527972582844],
            vec![
                -0.21415925868009242,
                -0.18007115752162856,
                -0.1454963095831485,
            ],
        ],
    ];

    assert_eq!(dbiases, expected_dbiases);
    assert_eq!(dweights, expected_dweights);
}

#[test]
pub fn test_gradient_descent() {
    let mut net = _setup_tests();
    // weights[l][j][k]: connects neuron k on layer l with neuron j on layer l+1
    let mut dweights = vec![
        vec![vec![0.1, 0.3], vec![-0.1, 0.3], vec![-0.4, 0.2]],
        vec![vec![-0.1, -0.3, 0.5], vec![0.1, -0.3, 0.7]],
    ];
    // biases[l][j]: bias of neuron j on layer l-1
    let alpha = 2.0;
    let mut dbiases = vec![vec![-0.1, -0.3, 0.5], vec![0.1, -0.3]];
    net.gradient_descent(&mut dbiases, &mut dweights, 1, 1.0 - alpha);
    assert_eq!(
        net.weights,
        vec![
            vec![
                vec![0.1 * alpha, 0.3 * alpha],
                vec![-0.1 * alpha, 0.3 * alpha],
                vec![-0.4 * alpha, 0.2 * alpha]
            ],
            vec![
                vec![-0.1 * alpha, -0.3 * alpha, 0.5 * alpha],
                vec![0.1 * alpha, -0.3 * alpha, 0.7 * alpha]
            ],
        ]
    );
    assert_eq!(
        net.biases,
        vec![
            vec![-0.1 * alpha, -0.3 * alpha, 0.5 * alpha],
            vec![0.1 * alpha, -0.3 * alpha]
        ]
    );
    assert_eq!(dweights, vec![vec![vec![0.0; 2]; 3], vec![vec![0.0; 3]; 2]]);
    assert_eq!(dbiases, vec![vec![0.0; 3], vec![0.0; 2]]);
}

#[test]
pub fn test_fns() {
    let net = _setup_tests();
    assert_eq!(net.activ_fn(0.0), 0.5);
    assert_eq!(net.activ_fn(f64::MAX), 1.0);
    assert_eq!(net.activ_fn(f64::MIN), 0.0);

    assert_eq!(net.deriv_fn(0.0), 0.25);
    assert_eq!(net.deriv_fn(f64::MAX), 0.0);
    assert_eq!(net.deriv_fn(f64::MIN), 0.0);
    assert_eq!(net.deriv_fn(3.0), 0.045176659730912144);
}
