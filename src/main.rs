use axum::{extract::State, routing::post, Json, Router};
use serde::Deserialize;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::services::ServeDir;

use std::{
    fs::File,
    io::{Error, Read},
};

use neural_network::{ActivationFn::Sigmoid, NeuralNetwork};

mod neural_network;

struct ServerState {
    net: NeuralNetwork,
}

type SharedState = Arc<Mutex<ServerState>>;

#[tokio::main]
async fn main() {
    let state = Arc::new(Mutex::new(ServerState {
        net: NeuralNetwork::new(vec![784, 32, 32, 10], Sigmoid),
    }));

    let host = "127.0.0.1";
    let port = "80";
    let addr: SocketAddr = format!("{}:{}", host, port).parse().unwrap();

    let app: Router = Router::new()
        .route("/network/test", post(recognize_user_image))
        .route("/network/train", post(train))
        .with_state(state)
        .nest_service("/", ServeDir::new("html"));

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn recognize_user_image(
    State(state): State<SharedState>,
    Json(payload): Json<Vec<f64>>,
) -> Json<Vec<f64>> {
    let state = state.lock().await;
    Json::from(state.net.forward_prop(&payload).last().unwrap().to_owned())
}
#[derive(Deserialize)]
struct TrainParams {
    batchsize: usize,
    alpha: f64,
}

async fn train(State(state): State<SharedState>, Json(payload): Json<TrainParams>) -> Json<f64> {
    let mut state = state.lock().await;
    train_mnist(&mut state.net, payload.batchsize, payload.alpha);
    let acc = test_mnist(&mut state.net);
    println!("Test Accuracy: {}%", acc);
    Json::from(acc)
}

fn train_mnist(net: &mut NeuralNetwork, batch_size: usize, alpha: f64) {
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

    net.train(images, expected, batch_size, alpha);
}

fn test_mnist(net: &mut NeuralNetwork) -> f64 {
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
    accurates as f64 * 100.0 / test_images.len() as f64
}

#[allow(dead_code)]
fn train_xor() {
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
            return;
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
