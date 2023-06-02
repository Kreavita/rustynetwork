use core::panic;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use std::{usize, vec};

pub struct NeuralNetwork {
    layers: Vec<usize>,
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<f64>>,
    activation_fn: ActivationFn,
}
#[derive(PartialEq)]
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

        let biases: Vec<Vec<f64>> = layers
            .iter()
            .skip(1)
            .map(|i| dist.sample_iter(&mut rng).take(*i).collect())
            .collect();
        let weights: Vec<Vec<Vec<f64>>> = layers
            .iter()
            .zip(layers.iter().skip(1))
            .map(|(k, j)| {
                (0..*j)
                    .map(|_| dist.sample_iter(&mut rng).take(*k).collect())
                    .collect()
            })
            .collect();

        Self {
            layers,
            weights,
            biases,
            activation_fn,
        }
    }
    pub fn test(&self, input: Vec<f64>) -> usize {
        self.hightest_index(self.forward_prop(&input).last().unwrap())
    }
    /// return the index of the greatest element in a vec
    pub fn hightest_index(&self, result: &Vec<f64>) -> usize {
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

        for (input, expected) in inputs.iter().zip(expected_outputs.iter()) {
            let activations: Vec<Vec<f64>> = self.forward_prop(input);
            self.backpropagate(&expected, &activations, &mut dbiases, &mut dweights);
            let res = self.hightest_index(activations.last().unwrap());

            batch_counter += 1;
            cost_counter += self.cost(&expected, &activations.last().unwrap());
            if res == self.hightest_index(expected) {
                acc_counter += 1;
            }
            if batch_counter == batch_size {
                //self._print_image(dataset);
                println!(
                    "acc: {} cost: {}",
                    acc_counter as f64 / batch_counter as f64,
                    cost_counter / batch_counter as f64
                );
                self.gradient_descent(&mut dbiases, &mut dweights, batch_size, alpha);
                batch_counter = 0;
                acc_counter = 0;
                cost_counter = 0.0;
            }
        }

        if batch_counter > 0 {
            self.gradient_descent(&mut dbiases, &mut dweights, batch_size, alpha);
        }
    }

    /// Propagates a input through the network
    pub fn forward_prop(&self, input: &Vec<f64>) -> Vec<Vec<f64>> {
        if input.len() != self.layers[0] {
            panic!("invalid input data");
        }
        let mut activations: Vec<Vec<f64>> = vec![input.clone()];
        for l in 1..self.layers.len() {
            activations.push(
                (0..self.layers[l])
                    .map(|j| self.activ_fn(self.calc_neuron(&activations[l - 1], l, j)))
                    .collect(),
            );
        }
        activations
    }

    pub fn backpropagate(
        &self,
        expected: &Vec<f64>,
        activations: &Vec<Vec<f64>>,
        dbiases: &mut Vec<Vec<f64>>,
        dweights: &mut Vec<Vec<Vec<f64>>>,
    ) {
        let lmax = self.layers.len() - 1;
        let mut last_db: Vec<f64> = (0..self.layers[lmax])
            .map(|k| {
                let value = 2.0 * (activations[lmax][k] - expected[k]);
                dbiases[lmax - 1][k] += value;
                return value;
            })
            .collect();

        for l in (0..lmax).rev() {
            let k_len = self.layers[l];
            let j_len = self.layers[l + 1];

            // Calculate the impact of each neuron and weights from this neuron backwards from the last layer
            // this part of the derivative chain rule is shared between weights and biases
            let d_consts: Vec<f64> = (0..j_len)
                .map(|j| {
                    self.deriv_fn(self.calc_neuron(&activations[l], l + 1, j)) * &last_db[j].clone()
                })
                .collect();

            let mut new_db = vec![0.0; k_len];
            for k in 0..k_len {
                for j in 0..j_len {
                    // calculate the impact of the weight connected between
                    // Neuron k on layer l and Neuron j on layer l+1 on the Cost
                    dweights[l][j][k] += activations[l][k] * d_consts[j];
                }
                if l > 0 {
                    // calculate the impact of the activation of
                    // Neuron k on layer l
                    new_db[k] = (0..j_len)
                        .map(|j| self.weights[l][j][k] * d_consts[j])
                        .sum::<f64>();
                    dbiases[l - 1][k] += d_consts[j];
                }
            }
            last_db = new_db;
        }
    }

    /// Applies a Gradient to the model, using alpha as the learning rate
    pub fn gradient_descent(
        &mut self,
        db: &mut Vec<Vec<f64>>,
        dw: &mut Vec<Vec<Vec<f64>>>,
        batch_size: usize,
        alpha: f64,
    ) {
        // apply the gradient to the model
        for l in 0..self.layers.len() - 1 {
            for j in 0..self.layers[l + 1] {
                self.biases[l][j] -= alpha * db[l][j] / batch_size as f64;
                db[l][j] = 0.0;
                for k in 0..self.layers[l] {
                    self.weights[l][j][k] -= alpha * dw[l][j][k] / batch_size as f64;
                    dw[l][j][k] = 0.0;
                }
            }
        }
    }

    /// calculate the neuron j on layer l using an input vector
    pub fn calc_neuron(&self, input: &Vec<f64>, l: usize, j: usize) -> f64 {
        if l < 1 {
            panic!("Cannot Calculate Layer_0 / Input Neurons!")
        }
        let k_max = self.layers[l - 1];
        if input.len() != k_max {
            panic!(
                "Size Mismatch! input: {}, should be: {}",
                input.len(),
                k_max
            )
        }
        // weights between previous neurons and the current neuron plus bias
        (0..k_max)
            .map(|k| self.weights[l - 1][j][k] * input[k])
            .sum::<f64>()
            + self.biases[l - 1][j]
    }

    pub fn activ_fn(&self, val: f64) -> f64 {
        let result: f64 = match self.activation_fn {
            ActivationFn::ReLU => val.max(0.0),
            ActivationFn::Linear => val,
            ActivationFn::Sigmoid => 1.0 / (1.0 + f64::exp(-val)) - 0.5,
            ActivationFn::TanH => f64::tanh(val),
        };
        if !result.is_finite() {
            panic!("Encountered not finite result trying sig({}) ", val);
        }
        result
    }

    pub fn deriv_fn(&self, val: f64) -> f64 {
        let result: f64 = match self.activation_fn {
            ActivationFn::ReLU => val.signum().max(0.0),
            ActivationFn::Linear => 1.0,
            ActivationFn::Sigmoid => {
                let sig: f64 = 1.0 / (1.0 + (-val).exp());
                sig * (1.0 - sig)
            }
            ActivationFn::TanH => 1.0 / f64::cosh(val).powi(2),
        };
        if !result.is_finite() {
            return 0.0;
            //panic!("Encountered not finite result trying d_sig({}) ", val);
        }
        result
    }

    pub fn cost(&self, expected: &Vec<f64>, result: &Vec<f64>) -> f64 {
        result
            .into_iter()
            .zip(expected.into_iter())
            .fold(0f64, |a, (r, e)| a + (r - e).powi(2))
    }

    pub fn _dumps(&self) -> String {
        todo!("muss gemacht werden");
    }

    pub fn _load() -> String {
        todo!("muss gemacht werden");
    }

    pub fn _print_image(&self, image: &Vec<f64>) {
        for i in 0..28 {
            for j in 0..28 {
                print!("{}", if image[i * 28 + j] > 0.0 { "*" } else { " " });
            }
            println!();
        }
    }
}

pub fn setup_tests() -> NeuralNetwork {
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
pub fn test_calc_neuron() {
    let net = setup_tests();
    let input = vec![3.0, 5.0];
    // weights[0][0] x input + biases[0][0]
    assert_eq!((net.calc_neuron(&input, 1, 0) * 10.0).round() / 10.0, 1.7);
    // weights[0][1] x input + biases[0][1]
    assert_eq!((net.calc_neuron(&input, 1, 1) * 10.0).round() / 10.0, 0.9);
    // weights[0][2] x input + biases[0][2]
    assert_eq!((net.calc_neuron(&input, 1, 2) * 10.0).round() / 10.0, 0.3);

    let input = vec![3.0, 5.0, 7.0];
    // weights[1][0] x input + biases[1][0]
    assert_eq!(
        net.calc_neuron(&input, 2, 0),
        0.1 + 3.0 * -0.1 + 5.0 * -0.3 + 7.0 * 0.5
    );
    // weights[1][1] x input + biases[1][1]
    assert_eq!(
        net.calc_neuron(&input, 2, 1),
        -0.3 + 3.0 * 0.1 + 5.0 * -0.3 + 7.0 * 0.7
    );
}

#[test]
pub fn test_evaluate() {
    let net = setup_tests();
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
            net.activ_fn(net.calc_neuron(&input2, 2, 0)),
            net.activ_fn(net.calc_neuron(&input2, 2, 1)),
        ],
    ];
    assert_eq!(net.forward_prop(&input), expected);
}

#[test]
pub fn test_backprop() {
    let net = setup_tests();
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
    net.backpropagate(&vec![0.0, 1.0], &activations, &mut dbiases, &mut dweights);

    dbg!(dweights);
    dbg!(dbiases);
}

#[test]
pub fn test_gradient_descent() {
    let mut net = setup_tests();
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
