use lib_simulation as sim;
use rand::prelude::*;
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Simulation {
    rng: ThreadRng,
    sim: sim::Simulation,
}

#[derive(Clone, Debug, Serialize)]
pub struct World {
    pub creatures: Vec<Creature>,
    pub resources: Vec<Resource>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Creature {
    pub x: f32,
    pub y: f32,
    pub rotation: f32,
    pub speed: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct Resource {
    pub x: f32,
    pub y: f32,
}

#[wasm_bindgen]
impl Simulation {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let mut rng = thread_rng();
        let sim = sim::Simulation::random(&mut rng);
        Self { rng, sim }
    }

    pub fn world(&self) -> JsValue {
        let world = World::from(self.sim.world());
        serde_wasm_bindgen::to_value(&world).unwrap()
    }

    // triggers movement for all entity instances capable of movement
    pub fn step(&mut self) {
        self.sim.step(&mut self.rng);
    }
}

impl From<&sim::World> for World {
    fn from(world: &sim::World) -> Self {
        let creatures = world.creatures().iter().map(Creature::from).collect();
        let resources = world.resources().iter().map(Resource::from).collect();
        Self {
            creatures,
            resources,
        }
    }
}

impl From<&sim::Creature> for Creature {
    fn from(creature: &sim::Creature) -> Self {
        Self {
            x: creature.position().x,
            y: creature.position().y,
            rotation: creature.rotation().angle(),
            speed: creature.speed(),
        }
    }
}

impl From<&sim::Resource> for Resource {
    fn from(resource: &sim::Resource) -> Self {
        Self {
            x: resource.position().x,
            y: resource.position().y,
        }
    }
}
