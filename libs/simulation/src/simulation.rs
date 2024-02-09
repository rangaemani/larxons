use crate::*;
pub struct Simulation {
    world: World,
}

impl Simulation {
    pub fn random(rng: &mut dyn RngCore) -> Self {
        Self {
            world: World::random(rng),
        }
    }

    pub fn world(&self) -> &World {
        &self.world
    }

    pub fn step(&mut self, rng: &mut dyn RngCore) {
        self.process_collisions(rng);
        self.process_movement();
    }

    fn process_collisions(&mut self, rng: &mut dyn RngCore) {
        for creature in &mut self.world.creatures {
            for resource in &mut self.world.resources {
                let distance_between_entities =
                    na::distance(&creature.position, &resource.position);

                if distance_between_entities <= 0.01 {
                    resource.position = rng.gen();
                }
            }
        }
        todo!()
    }

    fn process_movement(&mut self) {
        for creature in &mut self.world.creatures {
            creature.position += creature.rotation * na::Vector2::new(0.0, creature.speed);
            creature.position.x = na::wrap(creature.position.x, 0.0, 1.0);
            creature.position.y = na::wrap(creature.position.y, 0.0, 1.0);
        }
    }
}
