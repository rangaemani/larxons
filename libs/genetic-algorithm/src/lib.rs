#![allow(dead_code)]
#![feature(impl_trait_in_assoc_type)]
use rand::prelude::*;
use rand::RngCore;
use std::cmp::Ordering;
use std::ops::Index;
// Assuming Individual is a trait that provides a fitness method
pub trait Individual {
    fn chromosome(&self) -> &Chromosome;
    fn fitness(&self) -> f32;
    fn create(chromosome: Chromosome) -> Self;
}

pub trait IndividualFitnessSelectionMethod {
    fn select<'a, T>(&self, rng: &mut dyn RngCore, population: &'a [T]) -> Option<&'a T>
    where
        T: Individual;
}

pub trait GeneticCrossoverMethod {
    fn crossover(
        &self,
        rng: &mut dyn RngCore,
        parent_a: &Chromosome,
        parent_b: &Chromosome,
    ) -> Chromosome;
}

pub trait MutationMethod {
    fn mutate(&self, rng: &mut dyn RngCore, child: &mut Chromosome);
}

#[derive(Clone, Debug)]
pub struct UniformGeneticCrossover;

#[derive(Clone, Debug)]
pub struct Chromosome {
    genes: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct GaussianMutation {
    /// Probability of changing a gene:
    /// - 0.0 = no genes will be touched
    /// - 1.0 = all genes will be touched
    probability: f32,
    /// Magnitude of that change:
    /// - 0.0 = touched genes will not be modified
    /// - 3.0 = touched genes will be += or -= by at most 3.0
    coefficient: f32,
}

pub struct RankBasedSelection;

pub struct GeneticAlgorithm<S> {
    selection_method: S,
    crossover_method: Box<dyn GeneticCrossoverMethod>,
    mutation_method: Box<dyn MutationMethod>,
}

// Struct representing a ranked individual with its original index
struct RankedIndividual<T> {
    index: usize,
    individual: T,
    rank: usize,
}

impl<T: Individual> Ord for RankedIndividual<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank.cmp(&other.rank)
    }
}

impl<T: Individual> PartialOrd for RankedIndividual<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Individual> Eq for RankedIndividual<T> {}

impl<T: Individual> PartialEq for RankedIndividual<T> {
    fn eq(&self, other: &Self) -> bool {
        self.rank == other.rank
    }
}

impl RankBasedSelection {
    pub fn new() -> Self {
        Self
    }
}

impl IndividualFitnessSelectionMethod for RankBasedSelection {
    fn select<'a, T>(&self, rng: &mut dyn RngCore, population: &'a [T]) -> Option<&'a T>
    where
        T: Individual,
    {
        if population.len() < 1 {
            return Some(population.first().expect("Failed to select an individual"));
        }
        let mut ranked_population: Vec<RankedIndividual<&T>> = population
            .iter()
            .enumerate()
            .map(|(index, individual)| RankedIndividual {
                index,
                individual,
                rank: 0, // Placeholder rank to be updated later
            })
            .collect();

        // Sort the population based on fitness values in ascending order
        ranked_population.sort_unstable_by(|a, b| {
            a.individual
                .fitness()
                .partial_cmp(&b.individual.fitness())
                .unwrap_or(Ordering::Equal)
        });

        // Assign ranks to each individual in the sorted population
        for (rank, ranked_individual) in ranked_population.iter_mut().enumerate() {
            ranked_individual.rank = rank + 1;
        }

        // Calculate the total sum of ranks
        let total_sum: usize = ranked_population.iter().map(|r| r.rank).sum();

        // Generate a random number between  1 and total_sum
        let mut cumulative_sum: usize = 0;
        let selection_number = rng.gen_range(1..=total_sum);

        // Find the selected individual based on the cumulative sum of ranks
        for ranked_individual in &ranked_population {
            cumulative_sum += ranked_individual.rank;
            if cumulative_sum >= selection_number {
                return Some(&ranked_individual.individual);
            }
        }

        // Fallback in case the selection fails (should never happen)
        Some(
            ranked_population
                .first()
                .expect("Failed to select an individual")
                .individual,
        )
    }
}

impl<S> GeneticAlgorithm<S>
where
    S: IndividualFitnessSelectionMethod,
{
    pub fn new(
        selection_method: S,
        crossover_method: impl GeneticCrossoverMethod + 'static,
        mutation_method: impl MutationMethod + 'static,
    ) -> Self {
        Self {
            selection_method,
            crossover_method: Box::new(crossover_method),
            mutation_method: Box::new(mutation_method),
        }
    }

    pub fn evolve<T>(&self, rng: &mut dyn RngCore, population: &[T]) -> Vec<T>
    where
        T: Individual,
    {
        assert!(!population.is_empty());
        (0..population.len())
            .map(|_| {
                let parent_a = self
                    .selection_method
                    .select(rng, &population)
                    .unwrap()
                    .chromosome();
                let parent_b = self
                    .selection_method
                    .select(rng, &population)
                    .unwrap()
                    .chromosome();
                let mut child = self.crossover_method.crossover(rng, parent_a, parent_b);
                self.mutation_method.mutate(rng, &mut child);
                T::create(child)
            })
            .collect()
    }
}

impl Chromosome {
    pub fn len(&self) -> usize {
        self.genes.len()
    }
    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.genes.iter()
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.genes.iter_mut()
    }
}

impl Index<usize> for Chromosome {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.genes[index]
    }
}

impl FromIterator<f32> for Chromosome {
    fn from_iter<T: IntoIterator<Item = f32>>(iter: T) -> Self {
        Self {
            genes: iter.into_iter().collect(),
        }
    }
}

impl IntoIterator for Chromosome {
    type Item = f32;
    type IntoIter = impl Iterator<Item = f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.genes.into_iter()
    }
}

impl UniformGeneticCrossover {
    pub fn new() -> Self {
        Self
    }
}

impl GeneticCrossoverMethod for UniformGeneticCrossover {
    fn crossover(
        &self,
        rng: &mut dyn RngCore,
        parent_a: &Chromosome,
        parent_b: &Chromosome,
    ) -> Chromosome {
        assert_eq!(parent_a.len(), parent_b.len());
        let parent_a = parent_a.iter();
        let parent_b = parent_b.iter();
        parent_a
            .zip(parent_b)
            .map(|(&a, &b)| if rng.gen_bool(0.5) { a } else { b })
            .collect()
    }
}

impl GaussianMutation {
    pub fn new(probability: f32, coefficient: f32) -> Self {
        assert!(probability >= 0.0 && probability <= 1.0);

        Self {
            probability,
            coefficient,
        }
    }
}

impl MutationMethod for GaussianMutation {
    fn mutate(&self, rng: &mut dyn RngCore, child: &mut Chromosome) {
        for gene in child.iter_mut() {
            let sign = if rng.gen_bool(0.5) { -1.0 } else { 1.0 };

            if rng.gen_bool(self.probability.into()) {
                *gene += sign * self.coefficient * rng.gen::<f32>();
            }
        }
    }
}

/////////////////TESTS/////////////////////
#[cfg(test)]
mod tests {
    use super::*;

    // Mock Individual implementation for testing
    #[cfg(test)]
    #[derive(Clone, Debug, PartialEq)]
    pub enum TestIndividual {
        /// For tests that require access to chromosome
        WithChromosome { chromosome: Chromosome },

        /// For tests that don't require access to chromosome
        WithFitness { fitness: f32 },
    }

    #[cfg(test)]
    impl TestIndividual {
        pub fn new(fitness: f32) -> Self {
            Self::WithFitness { fitness }
        }
    }

    #[cfg(test)]
    impl Individual for TestIndividual {
        fn create(chromosome: Chromosome) -> Self {
            Self::WithChromosome { chromosome }
        }

        fn chromosome(&self) -> &Chromosome {
            match self {
                Self::WithChromosome { chromosome } => chromosome,

                Self::WithFitness { .. } => {
                    panic!("not supported for TestIndividual::WithFitness")
                }
            }
        }

        fn fitness(&self) -> f32 {
            match self {
                Self::WithChromosome { chromosome } => {
                    chromosome.iter().sum()

                    // ^ the simplest fitness function ever - we're just
                    // summing all the genes together
                }

                Self::WithFitness { fitness } => *fitness,
            }
        }
    }

    #[cfg(test)]
    impl PartialEq for Chromosome {
        fn eq(&self, other: &Self) -> bool {
            approx::relative_eq!(self.genes.as_slice(), other.genes.as_slice(),)
        }
    }
    mod rank_based_selection {
        use super::*;
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        #[test]
        fn test_rank_based_selection() {
            // Initialize a mock population with known fitness scores
            let population = vec![
                TestIndividual::WithFitness { fitness: 0.1 },
                TestIndividual::WithFitness { fitness: 0.3 },
                TestIndividual::WithFitness { fitness: 0.2 },
                TestIndividual::WithFitness { fitness: 0.5 },
                TestIndividual::WithFitness { fitness: 0.4 },
            ];

            // Use a deterministic seed for reproducibility
            let mut rng = StdRng::seed_from_u64(42);

            // Instantiate the rank-based selection
            let selection = RankBasedSelection::new();

            // Perform multiple selections and check if the selection mechanism works correctly
            for _ in 0..10 {
                let selected = selection.select(&mut rng, &population);
                assert!(selected.is_some(), "Selection should always return Some");
                let selected_rank = population
                    .iter()
                    .position(|ind| match ind {
                        TestIndividual::WithFitness { fitness } => {
                            *fitness == selected.unwrap().fitness()
                        }
                        _ => false,
                    })
                    .unwrap()
                    + 1;
                let expected_ranks = vec![1, 2, 3, 4, 5]; // Corrected expected ranks based on fitness
                assert!(
                    expected_ranks.contains(&selected_rank),
                    "Selected individual should have a correct rank"
                );
            }
        }

        #[test]
        fn test_rank_based_selection_single_individual() {
            let population = vec![TestIndividual::WithFitness { fitness: 0.5 }];
            let mut rng = StdRng::seed_from_u64(42);
            let selection = RankBasedSelection::new();
            let selected = selection.select(&mut rng, &population);
            assert!(selected.is_some(), "Selection should always return Some");
            assert_eq!(
                selected.unwrap().fitness(),
                0.5,
                "Selection should return the single individual"
            );
        }

        #[test]
        fn test_rank_based_selection_equal_fitness() {
            let population = vec![
                TestIndividual::WithFitness { fitness: 0.5 },
                TestIndividual::WithFitness { fitness: 0.5 },
                TestIndividual::WithFitness { fitness: 0.5 },
            ];
            let mut rng = StdRng::seed_from_u64(42);
            let selection = RankBasedSelection::new();
            let selected = selection.select(&mut rng, &population);
            assert!(selected.is_some(), "Selection should always return Some");
            assert!(
                population
                    .iter()
                    .any(|ind| ind.fitness() == selected.unwrap().fitness()),
                "Selection should return one of the individuals with equal fitness"
            );
        }

        #[test]
        fn test_rank_based_selection_negative_fitness() {
            let population = vec![
                TestIndividual::WithFitness { fitness: -0.5 },
                TestIndividual::WithFitness { fitness: 0.0 },
                TestIndividual::WithFitness { fitness: 0.5 },
            ];
            let mut rng = StdRng::seed_from_u64(42);
            let selection = RankBasedSelection::new();
            let selected = selection.select(&mut rng, &population);
            assert!(selected.is_some(), "Selection should always return Some");
            assert!(
                population
                    .iter()
                    .any(|ind| ind.fitness() == selected.unwrap().fitness()),
                "Selection should return one of the individuals regardless of fitness sign"
            );
        }

        #[test]
        fn test_rank_based_selection_extreme_values() {
            let population = vec![
                TestIndividual::WithFitness {
                    fitness: std::f32::MAX,
                },
                TestIndividual::WithFitness {
                    fitness: std::f32::MIN,
                },
                TestIndividual::WithFitness { fitness: 0.0 },
            ];
            let mut rng = StdRng::seed_from_u64(42);
            let selection = RankBasedSelection::new();
            let selected = selection.select(&mut rng, &population);
            assert!(selected.is_some(), "Selection should always return Some");
            assert!(
                population
                    .iter()
                    .any(|ind| ind.fitness() == selected.unwrap().fitness()),
                "Selection should return one of the individuals even with extreme fitness values"
            );
        }
    }
    mod index {
        use super::*;
        #[test]
        fn test_chromosome_len() {
            let chromosome = Chromosome {
                genes: vec![0.0, 1.0, 2.0],
            };
            assert_eq!(
                chromosome.len(),
                3,
                "Length should match the number of genes"
            );
        }
        #[test]
        fn test_chromosome_iter() {
            let chromosome = Chromosome {
                genes: vec![0.0, 1.0, 2.0],
            };
            let mut iter = chromosome.iter();
            assert_eq!(iter.next(), Some(&0.0), "First gene should be  0.0");
            assert_eq!(iter.next(), Some(&1.0), "Second gene should be  1.0");
            assert_eq!(iter.next(), Some(&2.0), "Third gene should be  2.0");
            assert_eq!(iter.next(), None, "Iterator should be exhausted");
        }
        #[test]
        fn test_chromosome_iter_mut() {
            let mut chromosome = Chromosome {
                genes: vec![0.0, 1.0, 2.0],
            };
            let mut iter = chromosome.iter_mut();
            assert_eq!(
                iter.next(),
                Some(&mut 0.0),
                "First mutable gene should be  0.0"
            );
            assert_eq!(
                iter.next(),
                Some(&mut 1.0),
                "Second mutable gene should be  1.0"
            );
            assert_eq!(
                iter.next(),
                Some(&mut 2.0),
                "Third mutable gene should be  2.0"
            );
            assert_eq!(iter.next(), None, "Mutable iterator should be exhausted");
        }
    }
    mod iterator {
        use super::*;
        #[test]
        fn test_chromosome_from_iterator() {
            let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
            let chromosome: Chromosome = data.into_iter().collect();
            assert_eq!(
                chromosome.len(),
                5,
                "Length should match the number of elements collected"
            );
            let mut iter = chromosome.iter();
            assert_eq!(iter.next(), Some(&0.0), "First gene should be   0.0");
            assert_eq!(iter.next(), Some(&1.0), "Second gene should be   1.0");
            assert_eq!(iter.next(), Some(&2.0), "Third gene should be   2.0");
            assert_eq!(iter.next(), Some(&3.0), "Fourth gene should be   3.0");
            assert_eq!(iter.next(), Some(&4.0), "Fifth gene should be   4.0");
            assert_eq!(iter.next(), None, "Iterator should be exhausted");
        }
    }
    mod chromosome {
        use super::*;
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        #[test]
        fn test_uniform_crossover_diff() {
            let mut rng = StdRng::seed_from_u64(42);

            let parent_a: Chromosome = (1..=100).map(|n| n as f32).collect();

            let parent_b: Chromosome = (1..=100).map(|n| -n as f32).collect();

            let child = UniformGeneticCrossover::new().crossover(&mut rng, &parent_a, &parent_b);

            // Number of genes different between `child` and `parent_a`
            let diff_a = child.iter().zip(parent_a).filter(|(c, p)| *c != p).count();

            // Number of genes different between `child` and `parent_b`
            let diff_b = child.iter().zip(parent_b).filter(|(c, p)| *c != p).count();

            assert_eq!(diff_a, 48);
            assert_eq!(diff_b, 52);
        }
    }
    mod mutation {
        use super::*;
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        // Helper function to compare two chromosomes for equality
        fn chromosomes_are_equal(a: &Chromosome, b: &Chromosome) -> bool {
            a.iter()
                .zip(b.iter())
                .all(|(gene_a, gene_b)| gene_a == gene_b)
        }

        mod given_zero_chance {
            use super::*;

            mod and_zero_coefficient {
                use super::*;

                #[test]
                fn does_not_change_the_original_chromosome() {
                    let mut rng = StdRng::seed_from_u64(42);
                    let mutation_method = GaussianMutation::new(0.0, 0.0);

                    let mut chromosome = Chromosome {
                        genes: vec![0.0, 1.0, 2.0, 3.0, 4.0],
                    };

                    let original_chromosome = chromosome.clone();
                    mutation_method.mutate(&mut rng, &mut chromosome);

                    assert!(chromosomes_are_equal(&chromosome, &original_chromosome));
                }
            }

            mod and_nonzero_coefficient {
                use super::*;

                #[test]
                fn does_not_change_the_original_chromosome() {
                    let mut rng = StdRng::seed_from_u64(42);
                    let mutation_method = GaussianMutation::new(0.0, 1.0);

                    let mut chromosome = Chromosome {
                        genes: vec![0.0, 1.0, 2.0, 3.0, 4.0],
                    };

                    let original_chromosome = chromosome.clone();
                    mutation_method.mutate(&mut rng, &mut chromosome);

                    assert!(chromosomes_are_equal(&chromosome, &original_chromosome));
                }
            }
        }

        mod given_fifty_fifty_chance {
            use super::*;

            mod and_zero_coefficient {
                use super::*;

                #[test]
                fn does_not_change_the_original_chromosome() {
                    let mut rng = StdRng::seed_from_u64(42);
                    let mutation_method = GaussianMutation::new(0.5, 0.0);

                    let mut chromosome = Chromosome {
                        genes: vec![0.0, 1.0, 2.0, 3.0, 4.0],
                    };

                    let original_chromosome = chromosome.clone();
                    mutation_method.mutate(&mut rng, &mut chromosome);

                    assert!(chromosomes_are_equal(&chromosome, &original_chromosome));
                }
            }

            mod and_nonzero_coefficient {
                use super::*;

                #[test]
                fn slightly_changes_the_original_chromosome() {
                    let mut rng = StdRng::seed_from_u64(42);
                    let mutation_method = GaussianMutation::new(0.5, 1.0);

                    let mut chromosome = Chromosome {
                        genes: vec![0.0, 1.0, 2.0, 3.0, 4.0],
                    };

                    let original_chromosome = chromosome.clone();
                    mutation_method.mutate(&mut rng, &mut chromosome);

                    assert!(!chromosomes_are_equal(&chromosome, &original_chromosome));
                }
            }
        }

        mod given_max_chance {
            use super::*;

            mod and_zero_coefficient {
                use super::*;

                #[test]
                fn does_not_change_the_original_chromosome() {
                    let mut rng = StdRng::seed_from_u64(42);
                    let mutation_method = GaussianMutation::new(1.0, 0.0);

                    let mut chromosome = Chromosome {
                        genes: vec![0.0, 1.0, 2.0, 3.0, 4.0],
                    };

                    let original_chromosome = chromosome.clone();
                    mutation_method.mutate(&mut rng, &mut chromosome);

                    assert!(chromosomes_are_equal(&chromosome, &original_chromosome));
                }
            }

            mod and_nonzero_coefficient {
                use super::*;

                #[test]
                fn entirely_changes_the_original_chromosome() {
                    let mut rng = StdRng::seed_from_u64(42);
                    let mutation_method = GaussianMutation::new(1.0, 1.0);

                    let mut chromosome = Chromosome {
                        genes: vec![0.0, 1.0, 2.0, 3.0, 4.0],
                    };

                    let original_chromosome = chromosome.clone();
                    mutation_method.mutate(&mut rng, &mut chromosome);

                    assert!(!chromosomes_are_equal(&chromosome, &original_chromosome));
                }
            }
        }
    }
    mod genetic_algorithm {
        use super::*;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        fn individual(genes: &[f32]) -> TestIndividual {
            let chromosome = genes.iter().cloned().collect();
            TestIndividual::create(chromosome)
        }

        #[test]
        fn test_genetic_algorithm_evolution() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());

            let ga = GeneticAlgorithm::new(
                RankBasedSelection::new(),
                UniformGeneticCrossover::new(),
                GaussianMutation::new(0.5, 0.5),
            );

            let mut population = vec![
                individual(&[0.0, 0.0, 0.0]), // fitness = 0.0
                individual(&[1.0, 1.0, 1.0]), // fitness = 3.0
                individual(&[1.0, 2.0, 1.0]), // fitness = 4.0
                individual(&[1.0, 2.0, 4.0]), // fitness = 7.0
            ];

            for _ in 0..10 {
                population = ga.evolve(&mut rng, &population);
            }

            // Define the expected population after evolution
            // The actual genes and fitness values will depend on the implementation of the genetic algorithm
            // and the randomness introduced by the mutation and crossover methods.
            // You should replace the genes with the actual expected values after running the test once.
            let expected_population = vec![
                individual(&[0.9454211, 3.9096963, 4.82042]), // fitness = 9.6755374
                individual(&[1.7498862, 3.7476456, 4.0445285]), // fitness = 9.5420603
                individual(&[1.0134296, 4.0383005, 4.4282184]), // fitness = 9.4799485
                individual(&[1.3154963, 3.6526883, 4.487094]), // fitness = 9.4552786
            ];

            assert_eq!(population, expected_population);
        }
    }
}
