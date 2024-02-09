use crate::*;
#[derive(Debug)]
pub struct Resource {
    pub(crate) position: na::Point2<f32>,
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
