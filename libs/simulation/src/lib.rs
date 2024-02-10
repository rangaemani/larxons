pub use self::{brain::*, creature::*, eye::*, resource::*, world::*};

mod brain;
mod creature;
mod creature_individual;
mod eye;
mod resource;
mod world;

use self::creature_individual::*;
use algo::{GaussianMutation, RankBasedSelection, UniformGeneticCrossover};
use lib_genetic_algorithm as algo;
use lib_semnet as semnet;
use na::Point2;
use nalgebra as na;
use rand::{seq::SliceRandom, Rng, RngCore};
use std::f32::consts::FRAC_PI_2;

/// Minimum speed of a bird.
///
/// Keeping it above zero prevents birds from getting stuck in one place.
const SPEED_MIN: f32 = 0.0001;

/// Maximum speed of a bird.
///
/// Keeping it "sane" prevents birds from accelerating up to infinity,
/// which makes the simulation... unrealistic :-)
const SPEED_MAX: f32 = 0.0015;

/// Speed acceleration; determines how much the brain can affect bird's
/// speed during one step.
///
/// Assuming our bird is currently flying with speed=0.5, when the brain
/// yells "stop flying!", a SPEED_ACCEL of:
///
/// - 0.1 = makes it take 5 steps ("5 seconds") for the bird to actually
///         slow down to SPEED_MIN,
///
/// - 0.5 = makes it take 1 step for the bird to slow down to SPEED_MIN.
///
/// This improves simulation faithfulness, because - as in real life -
/// it's not possible to increase speed from 1km/h to 50km/h in one
/// instant, even if your brain very much wants to.
const SPEED_ACCEL: f32 = 0.2;

/// Ditto, but for rotation:
///
/// - 2 * PI = it takes one step for the bird to do a 360° rotation,
/// - PI = it takes two steps for the bird to do a 360° rotation,
///
/// I've chosen PI/2, because - as our motto goes - this value seems
/// to play nice.
const ROTATION_ACCEL: f32 = FRAC_PI_2 / 10.0;

/// How much `.step()`-s have to occur before we push data into the
/// genetic algorithm.
///
/// Value that's too low might prevent the birds from learning, while
/// a value that's too high will make the evolution unnecessarily
/// slower.
///
/// You can treat this number as "for how many steps each bird gets
/// to live"; 2500 was chosen with a fair dice roll.
const GENERATION_LENGTH: usize = 2500;

pub struct Simulation {
    pub(crate) world: World,
    algorithm: algo::GeneticAlgorithm<algo::RankBasedSelection>,
    age: usize,
}

impl Simulation {
    pub fn random(rng: &mut dyn RngCore) -> Self {
        let world = World::random(rng);
        let algorithm = algo::GeneticAlgorithm::new(
            RankBasedSelection,
            UniformGeneticCrossover,
            GaussianMutation::new(0.01, 0.3),
        );
        Self {
            world,
            algorithm,
            age: 0,
        }
    }

    pub fn world(&self) -> &World {
        &self.world
    }

    pub fn step(&mut self, rng: &mut dyn RngCore) -> Option<algo::Statistics> {
        self.process_collisions(rng);
        self.process_brains();
        self.process_movement();

        self.age += 1;

        if self.age > GENERATION_LENGTH {
            Some(self.evolve(rng))
        } else {
            None
        }
    }

    fn process_collisions(&mut self, rng: &mut dyn RngCore) {
        for creature in &mut self.world.creatures {
            for resource in &mut self.world.resources {
                let distance_between_entities =
                    na::distance(&creature.position, &resource.position);

                if distance_between_entities <= 0.01 {
                    creature.satiation += 1;
                    resource.position = rng.gen();
                }
            }
        }
    }

    fn process_brains(&mut self) {
        for creature in &mut self.world.creatures {
            let vision = creature.eye.process_vision(
                creature.position,
                creature.rotation,
                &self.world.resources,
            );

            let response = creature.brain.semnet.propagate(vision);

            // ---
            // | Limits number to given range.
            // -------------------- v---v
            let speed = response[0].clamp(-SPEED_ACCEL, SPEED_ACCEL);

            let rotation = response[1].clamp(-ROTATION_ACCEL, ROTATION_ACCEL);

            // Our speed & rotation here are *relative* - that is: when
            // they are equal to zero, what the brain says is "keep
            // flying as you are now", not "stop flying".
            //
            // Both values being relative is crucial, because our bird's
            // brain doesn't know its own speed and rotation*, meaning
            // that it fundamentally cannot return absolute values.
            //
            // * they'd have to be provided as separate inputs to the
            //   neural network, which would make the evolution process
            //   waaay longer, if even possible.

            creature.speed = (creature.speed + speed).clamp(SPEED_MIN, SPEED_MAX);

            creature.rotation = na::Rotation2::new(creature.rotation.angle() + rotation);

            // (btw, there is no need for ROTATION_MIN or ROTATION_MAX,
            // because rotation automatically wraps from 2*PI back to 0 -
            // we've already witnessed that when we were testing eyes,
            // inside `mod different_rotations { ... }`.)
        }
    }

    fn process_movement(&mut self) {
        for creature in &mut self.world.creatures {
            creature.position += creature.rotation * na::Vector2::new(0.0, creature.speed);
            creature.position.x = na::wrap(creature.position.x, 0.0, 1.0);
            creature.position.y = na::wrap(creature.position.y, 0.0, 1.0);
        }
    }

    fn evolve(&mut self, rng: &mut dyn RngCore) -> algo::Statistics {
        self.age = 0;

        let current_population: Vec<CreatureIndividual> = self
            .world
            .creatures
            .iter()
            .map(|creature| CreatureIndividual::from_creature(&creature))
            .collect();

        let (mutated_population, stats) = self.algorithm.evolve(rng, &current_population);

        // Assuming there is a field named `creatures` in `World` that should be updated
        // with the `mutated_population`. Replace `todo!()` with the actual conversion.
        self.world.creatures = mutated_population
            .into_iter()
            .map(|individual| individual.into_creature(rng))
            .collect();

        for resource in &mut self.world.resources {
            resource.position = rng.gen();
        }

        stats
    }

    /// Fast-forwards 'till the end of the current generation.
    pub fn train(&mut self, rng: &mut dyn RngCore) -> algo::Statistics {
        loop {
            if let Some(summary) = self.step(rng) {
                return summary;
            }
        }
    }
}
