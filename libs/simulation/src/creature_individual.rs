use crate::*;

pub struct CreatureIndividual {
    fitness: f32,
    chromosome: algo::Chromosome,
}

impl CreatureIndividual {
    pub fn from_creature(creature: &Creature) -> Self {
        Self {
            fitness: creature.satiation as f32,
            chromosome: creature.as_chromosome(),
        }
    }

    pub fn into_creature(self, rng: &mut dyn RngCore) -> Creature {
        Creature::from_chromosome(self.chromosome, rng)
    }
}

impl algo::Individual for CreatureIndividual {
    fn create(chromosome: algo::Chromosome) -> Self {
        Self {
            fitness: 0.0,
            chromosome,
        }
    }

    fn chromosome(&self) -> &algo::Chromosome {
        &self.chromosome
    }

    fn fitness(&self) -> f32 {
        self.fitness
    }
}
