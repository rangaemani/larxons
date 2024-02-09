pub use self::{creature::*, resource::*, world::*};

mod creature;
mod resource;
mod world;

use na::Point2;
use nalgebra as na;
use rand::{seq::SliceRandom, Rng, RngCore};
