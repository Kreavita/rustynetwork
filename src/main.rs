use core::panic;
use std::{
    fs::File,
    io::{Error, Read},
};

use neural_network::{ActivationFn::Sigmoid, NeuralNetwork};

mod neural_network;

fn main() {
    let mut net: NeuralNetwork = NeuralNetwork::new(vec![784, 16, 16, 10], Sigmoid);

    let labels = get_labels("data/train-labels-idx1-ubyte".to_string()).unwrap();
    let images = get_images("data/train-images-idx3-ubyte".to_string()).unwrap();

    let expected: Vec<Vec<f64>> = labels
        .iter()
        .map(|label| {
            (0..10)
                .map(|i: u8| if i == *label { 1.0 } else { 0.0 })
                .collect()
        })
        .collect();

    net.train(images, expected, 10, 2.5);

    let test_images = get_images("data/t10k-images-idx3-ubyte".to_string()).unwrap();
    let test_labels = get_labels("data/t10k-labels-idx1-ubyte".to_string()).unwrap();
    let mut accurates = 0;
    for (image, label) in test_images.iter().zip(test_labels.iter()) {
        if *label as usize == net.test(image.to_vec()) {
            accurates += 1;
        }
        // println!("Expected: {} Got: {}", label, net.test(image.to_vec()));
        // net._print_image(image);
    }
    println!(
        "Test Accuracy: {}%",
        accurates as f64 * 100.0 / test_images.len() as f64
    )
}

fn _xor_training() {
    let mut testnet = NeuralNetwork::new(vec![2, 2, 1], Sigmoid);
    let labels: Vec<Vec<f64>> = vec![vec![1.0], vec![0.0], vec![1.0], vec![0.0]];
    let images: Vec<Vec<f64>> = vec![
        vec![1.0, 0.0],
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    let mut ctr = 0;
    loop {
        testnet.train(images.to_vec(), labels.to_vec(), 4, 0.005);

        ctr += 1;
        if ctr == 1000000 {
            println!(
                "xor(1,1) {}",
                testnet.forward_prop(&vec![1.0, 1.0]).last().unwrap()[0]
            );
            println!(
                "xor(0,1) {}",
                testnet.forward_prop(&vec![0.0, 1.0]).last().unwrap()[0]
            );
            println!(
                "xor(1,0) {}",
                testnet.forward_prop(&vec![1.0, 0.0]).last().unwrap()[0]
            );
            println!(
                "xor(0,0) {}",
                testnet.forward_prop(&vec![0.0, 0.0]).last().unwrap()[0]
            );
            panic!();
        }
    }
}

fn _print_dataset(images: &[Vec<f64>], labels: &[u8]) {
    for (lab, img) in labels.iter().zip(images.iter()) {
        println!("{lab}");
        for i in 0..28 {
            for j in 0..28 {
                print!("{}", if img[i * 28 + j] > 0.0 { "*" } else { " " });
            }
            println!();
        }
    }
}

fn get_labels(path: String) -> Result<Vec<u8>, Error> {
    let mut file = File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;

    Ok(bytes[8..].to_vec())
}

fn get_images(path: String) -> Result<Vec<Vec<f64>>, Error> {
    let mut file = File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;

    let mut images = Vec::new();
    let mut image = Vec::new();
    for byte in bytes[16..].iter().copied() {
        image.push(byte as f64 / 255.0);
        if image.len() == 28 * 28 {
            images.push(image.to_vec());
            image.clear();
        }
    }

    Ok(images)
}
