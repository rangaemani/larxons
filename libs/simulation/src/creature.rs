use crate::*;
#[derive(Debug)]
pub struct Creature {
    // x, y position
    pub(crate) position: na::Point2<f32>,
    pub(crate) rotation: na::Rotation2<f32>,
    pub(crate) speed: f32,
    pub(crate) eye: Eye,
    pub(crate) brain: Brain,
    pub(crate) satiation: usize,
}

impl Creature {
    pub fn random(rng: &mut dyn RngCore) -> Self {
        let eye = Eye::default();
        let brain = Brain::random(rng, &eye);

        Self::new(eye, brain, rng)
    }

    pub(crate) fn as_chromosome(&self) -> algo::Chromosome {
        // We evolve only our birds' brains, but technically there's no
        // reason not to simulate e.g. physical properties such as size.
        //
        // If that was to happen, this function could be adjusted to
        // return a longer chromosome that encodes not only the brain,
        // but also, say, birdie's color.

        self.brain.as_chromosome()
    }

    /// "Restores" bird from a chromosome.
    ///
    /// We have to have access to the PRNG in here, because our
    /// chromosomes encode only the brains - and while we restore the
    /// bird, we have to also randomize its position, direction, etc.
    /// (so it's stuff that wouldn't make sense to keep in the genome.)
    pub(crate) fn from_chromosome(chromosome: algo::Chromosome, rng: &mut dyn RngCore) -> Self {
        let eye = Eye::default();
        let brain = Brain::from_chromosome(chromosome, &eye);

        Self::new(eye, brain, rng)
    }

    fn new(eye: Eye, brain: Brain, rng: &mut dyn RngCore) -> Self {
        Self {
            position: rng.gen(),
            rotation: rng.gen(),
            speed: 0.002,
            eye,
            brain,
            satiation: 0,
        }
    }

    pub fn from_coordinate(rng: &mut dyn RngCore, coords: Point2<f32>) -> Self {
        let eye = Eye::default();
        let brain = semnet::Network::random(&[
            semnet::LayerConfiguration {
                neural_capacity: eye.cells(),
            },
            semnet::LayerConfiguration {
                neural_capacity: 2 * eye.cells(),
            },
            semnet::LayerConfiguration { neural_capacity: 2 },
        ]);
        Creature {
            position: coords,
            rotation: na::Rotation2::new(rng.gen()),
            speed: 0.001,
            eye,
            brain: brain::Brain { semnet: brain },
            satiation: 0,
        }
    }

    pub fn position(&self) -> Point2<f32> {
        self.position
    }

    pub fn speed(&self) -> f32 {
        self.speed
    }

    pub fn rotation(&self) -> na::Rotation2<f32> {
        self.rotation
    }
}
