use crate::*;
#[derive(Debug)]
pub struct Creature {
    // x, y position
    pub(crate) position: na::Point2<f32>,
    pub(crate) rotation: na::Rotation2<f32>,
    pub(crate) speed: f32,
}

impl Creature {
    pub fn random(rng: &mut dyn RngCore) -> Self {
        Creature {
            position: rng.gen(),
            rotation: na::Rotation2::new(rng.gen()),
            speed: 0.001,
        }
    }

    pub fn from_coordinate(rng: &mut dyn RngCore, coords: Point2<f32>) -> Self {
        Creature {
            position: coords,
            rotation: na::Rotation2::new(rng.gen()),
            speed: 0.001,
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
