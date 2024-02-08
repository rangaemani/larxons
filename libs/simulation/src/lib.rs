use na::Point2;
use nalgebra as na;
use rand::{seq::SliceRandom, Rng, RngCore};

///// STRUCTS
pub struct Simulation {
    world: World,
}

#[derive(Debug)]
pub struct World {
    creatures: Vec<Creature>,
    resources: Vec<Resource>,
}

#[derive(Debug)]
pub struct Creature {
    // x, y position
    position: na::Point2<f32>,
    rotation: na::Rotation2<f32>,
    speed: f32,
}

#[derive(Debug)]
pub struct Resource {
    position: na::Point2<f32>,
}

//////IMPLEMENTATION METHODS
impl Simulation {
    pub fn random(rng: &mut dyn RngCore) -> Self {
        Self {
            world: World::random(rng),
        }
    }

    pub fn world(&self) -> &World {
        &self.world
    }
}

impl World {
    pub fn random(rng: &mut dyn RngCore) -> Self {
        const GRID_SIZE: usize = 10; // Size of the grid for supersampling
        const NUM_ENTITIES: usize = 100; // Total number of creatures and resources

        // Generate a grid of potential positions
        let mut positions = Vec::with_capacity(GRID_SIZE * GRID_SIZE);
        for i in 0..GRID_SIZE {
            for j in 0..GRID_SIZE {
                positions.push((i as f32, j as f32));
            }
        }

        // Randomly select a subset of positions without replacement
        let selected_positions: Vec<(usize, usize)> = positions
            .choose_multiple(rng, NUM_ENTITIES)
            .into_iter()
            .map(|&(x, y)| (x as usize, y as usize))
            .collect();

        // Place creatures and resources at the selected positions
        let mut creatures = Vec::new();
        let mut resources = Vec::new();
        for (i, j) in selected_positions {
            let offset_x = rng.gen_range(0.0..1.0);
            let offset_y = rng.gen_range(0.0..1.0);
            if creatures.len() < NUM_ENTITIES / 2 {
                creatures.push(Creature::from_coordinate(
                    rng,
                    na::Point2::new(
                        (i as f32 + offset_x) / GRID_SIZE as f32,
                        (j as f32 + offset_y) / GRID_SIZE as f32,
                    ),
                ));
            } else {
                resources.push(Resource::from_coordinate(na::Point2::new(
                    (i as f32 + offset_x) / GRID_SIZE as f32,
                    (j as f32 + offset_y) / GRID_SIZE as f32,
                )));
            }
        }

        Self {
            creatures,
            resources,
        }
    }

    pub fn creatures(&self) -> &[Creature] {
        &self.creatures
    }

    pub fn resources(&self) -> &[Resource] {
        &self.resources
    }
}

impl Creature {
    pub fn random(rng: &mut dyn RngCore) -> Self {
        Creature {
            position: rng.gen(),
            rotation: na::Rotation2::new(rng.gen()),
            speed: 0.002,
        }
    }

    pub fn from_coordinate(rng: &mut dyn RngCore, coords: Point2<f32>) -> Self {
        Creature {
            position: coords,
            rotation: na::Rotation2::new(rng.gen()),
            speed: 0.002,
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

impl Resource {
    pub fn random(rng: &mut dyn RngCore) -> Self {
        Resource {
            position: rng.gen(),
        }
    }

    pub fn from_coordinate(coords: Point2<f32>) -> Self {
        Resource { position: coords }
    }

    pub fn position(&self) -> Point2<f32> {
        self.position
    }
}
