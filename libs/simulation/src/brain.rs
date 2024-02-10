use crate::*;

#[derive(Debug)]
pub struct Brain {
    pub(crate) semnet: semnet::Network,
}

impl Brain {
    pub fn random(rng: &mut dyn RngCore, eye: &Eye) -> Self {
        Self {
            semnet: semnet::Network::random(&Self::topology(eye)),
        }
    }

    pub(crate) fn as_chromosome(&self) -> algo::Chromosome {
        self.semnet.weights().into_iter().collect()
    }

    pub(crate) fn from_chromosome(chromosome: algo::Chromosome, eye: &Eye) -> Self {
        Self {
            semnet: semnet::Network::from_weights(&Self::topology(eye), chromosome),
        }
    }

    fn topology(eye: &Eye) -> [semnet::LayerConfiguration; 3] {
        [
            semnet::LayerConfiguration {
                neural_capacity: eye.cells(),
            },
            semnet::LayerConfiguration {
                neural_capacity: 2 * eye.cells(),
            },
            semnet::LayerConfiguration { neural_capacity: 2 },
        ]
    }
}
