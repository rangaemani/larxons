use std::fmt;

use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
/// # Neuron
/// Simplest part of a neural network.
/// Consists of a bias (propogation factor), and weights (to determine which neuron to output to).
#[derive(Debug)]
struct Neuron {
    bias: f32,
    weights: Vec<f32>,
}

impl Neuron {
    pub fn random(rng: &mut dyn RngCore, input_size: usize) -> Self {
        let bias = rng.gen_range(-1.0..=1.0);
        let weights = (0..input_size).map(|_| rng.gen_range(-1.0..=1.0)).collect();

        Self { bias, weights }
    }

    pub fn new(bias: f32, weights: Vec<f32>) -> Self {
        Neuron { bias, weights }
    }

    pub fn from_weights(input_size: usize, weights: &mut dyn Iterator<Item = f32>) -> Self {
        let bias = weights.next().expect("got not enough weights");

        let weights = (0..input_size)
            .map(|_| weights.next().expect("got not enough weights"))
            .collect();

        Self { bias, weights }
    }

    fn propogate(&self, inputs: &[f32]) -> f32 {
        let output = inputs
            .iter()
            .zip(&self.weights)
            .map(|(input, weight)| input * weight)
            .sum::<f32>();

        return (output + self.bias).max(0.0);
    }
}
/// # Layer
/// Struct representing one layer of the neural network.
/// Exists as a list of Neurons that each have their own bias factor.
#[derive(Debug)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[Layer with {} neurons]", self.neurons.len())
    }
}

impl Layer {
    pub fn random(input_size: usize, output_size: usize) -> Self {
        assert!(
            input_size > 0 && output_size > 0,
            "Input size and output size must be greater than zero."
        );
        let mut rng_core = ChaCha8Rng::from_entropy();
        let neurons = (0..output_size)
            .map(|_| Neuron::random(&mut rng_core, input_size))
            .collect();
        Self { neurons }
    }

    pub fn new(neurons: Vec<Neuron>) -> Self {
        Layer { neurons }
    }

    pub fn from_weights(
        input_size: usize,
        output_size: usize,
        weights: &mut dyn Iterator<Item = f32>,
    ) -> Self {
        let neurons = (0..output_size)
            .map(|_| Neuron::from_weights(input_size, weights))
            .collect();

        Self { neurons }
    }

    fn propogate(&self, inputs: Vec<f32>) -> Vec<f32> {
        return self
            .neurons
            .iter()
            .map(|neuron| neuron.propogate(&inputs))
            .collect();
    }
}
/// # Layer Config
/// Describes the topology of each layer for easy definition
pub struct LayerConfiguration {
    pub neural_capacity: usize,
}
/// # Network
/// Simple representation of a neural/semantic network.
/// Consists of many layers.
#[derive(Debug)]
pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    /// # Constructor
    /// Creates a new Network from the given layers.
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }
    // Your function with added comments explaining the iterator behavior
    pub fn random(layer_configurations: &[LayerConfiguration]) -> Self {
        assert!(layer_configurations.iter().len() > 1);
        let mut generated_layers = Vec::new();
        // The range starts at  0 and ends at `layer_configurations.len() -  1`, so the iterator `i`
        for i in 0..(layer_configurations.len() - 1) {
            let input_configuration = &layer_configurations[i];
            let output_configuration = &layer_configurations[i + 1];

            // Creating a layer with the specified input and output sizes
            let gen_layer: Layer = Layer::random(
                input_configuration.neural_capacity,
                output_configuration.neural_capacity,
            );
            println!("Layer generated: {}", gen_layer);

            // Adding the newly created layer to the collection
            generated_layers.push(gen_layer);
            println!("i value: {}", i);
        }
        // Constructing the network with the generated layers
        Self {
            layers: generated_layers,
        }
    }

    pub fn from_weights(
        layers: &[LayerConfiguration],
        weights: impl IntoIterator<Item = f32>,
    ) -> Self {
        assert!(layers.len() > 1);

        let mut weights = weights.into_iter();

        let layers = layers
            .windows(2)
            .map(|layers| {
                Layer::from_weights(
                    layers[0].neural_capacity,
                    layers[1].neural_capacity,
                    &mut weights,
                )
            })
            .collect();

        if weights.next().is_some() {
            panic!("got too many weights");
        }

        Self { layers }
    }

    /// # Propagate
    /// Propogates the input through the network's layers.
    ///
    /// Takes a vector of floating-point values as input and passes them
    /// through each layer of the neural network. Calls each layer's `propogate` method
    /// in sequence, with the output of one layer serving as the input to the next.
    /// The final output of the network is returned as a vector of floating-point values.
    ///
    /// # Arguments
    ///
    /// * `inputs` - A vector of floating-point values representing the initial input
    ///   to the network.
    ///
    /// # Returns
    ///
    /// * A vector of floating-point values representing the output of the network
    ///   after processing the input through all layers.
    pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        return self
            .layers
            .iter()
            .fold(inputs, |inputs, layer| layer.propogate(inputs));
    }

    pub fn weights(&self) -> Vec<f32> {
        use std::iter::once;

        self.layers
            .iter()
            .flat_map(|layer| layer.neurons.iter())
            .flat_map(|neuron| once(&neuron.bias).chain(&neuron.weights))
            .copied()
            .collect()
    }
}
/////////////////TESTS/////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_neuron_generation_with_random_values() {
        let mut rng = ChaCha8Rng::from_seed(Default::default()); // Use a fixed seed for reproducibility
        let neuron = Neuron::random(&mut rng, 3);

        // Replace the expected values with the ones you got from running the test once
        approx::assert_relative_eq!(neuron.bias, -0.6255188);
        neuron
            .weights
            .iter()
            .zip([0.67383957, 0.8181262, 0.26284897].iter())
            .for_each(|(weight, expected)| {
                approx::assert_relative_eq!(weight, expected);
            });
    }

    #[test]
    fn test_neuron_activation_with_inputs() {
        let mut rng = ChaCha8Rng::from_seed(Default::default()); // Use a fixed seed for reproducibility
        let neuron = Neuron::random(&mut rng, 3);
        let inputs = vec![0.5, 0.3, 0.7];
        let output = neuron.propogate(&inputs);
        assert!(output >= 0.0);
    }

    #[test]
    fn test_layer_creation_with_random_neurons() {
        let layer = Layer::random(3, 2);
        assert_eq!(layer.neurons.len(), 2);
        assert_eq!(layer.neurons[0].weights.len(), 3);
    }

    #[test]
    fn test_layer_activation_with_inputs() {
        let layer = Layer::random(3, 2);
        let inputs = vec![0.5, 0.3, 0.7];
        let outputs = layer.propogate(inputs);
        assert_eq!(outputs.len(), 2);
    }

    #[test]
    fn test_network_construction_with_explicit_layers() {
        let layer1 = Layer::random(3, 2);
        let layer2 = Layer::random(2, 1);
        let network = Network::new(vec![layer1, layer2]);
        assert_eq!(network.layers.len(), 2);
    }

    #[test]
    fn test_network_generation_with_configurations() {
        let configs = vec![
            LayerConfiguration { neural_capacity: 3 },
            LayerConfiguration { neural_capacity: 2 },
        ];
        let network = Network::random(&configs);
        assert_eq!(network.layers.len(), 1);
        let configs = vec![
            LayerConfiguration { neural_capacity: 3 },
            LayerConfiguration { neural_capacity: 2 },
            LayerConfiguration { neural_capacity: 5 },
            LayerConfiguration { neural_capacity: 4 },
        ];
        let network2 = Network::random(&configs);
        assert_eq!(network2.layers.len(), 3);
    }

    #[test]
    fn test_network_activation_with_inputs() {
        let configs = vec![
            LayerConfiguration { neural_capacity: 3 },
            LayerConfiguration { neural_capacity: 2 },
        ];
        let network = Network::random(&configs);
        let inputs = vec![0.5, 0.3, 0.7];
        let outputs = network.propagate(inputs);
        assert_eq!(outputs.len(), 2);
    }

    mod weights {
        use super::*;

        #[test]
        fn test() {
            let network = Network::new(vec![
                Layer::new(vec![Neuron::new(0.1, vec![0.2, 0.3, 0.4])]),
                Layer::new(vec![Neuron::new(0.5, vec![0.6, 0.7, 0.8])]),
            ]);

            let actual = network.weights();
            let expected = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

            approx::assert_relative_eq!(actual.as_slice(), expected.as_slice(),);
        }
    }

    mod from_weights {
        use super::*;

        #[test]
        fn test() {
            let layers = &[
                LayerConfiguration { neural_capacity: 3 },
                LayerConfiguration { neural_capacity: 2 },
            ];

            let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

            let network = Network::from_weights(layers, weights.clone());
            let actual: Vec<_> = network.weights().into_iter().collect();

            approx::assert_relative_eq!(actual.as_slice(), weights.as_slice(),);
        }
    }
}
