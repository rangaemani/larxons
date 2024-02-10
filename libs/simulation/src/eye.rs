#![feature(custom_test_frameworks)]
use crate::*;
use std::f32::consts::*;

/// How far our eye can see:
///
/// -----------------
/// |               |
/// |               |
/// |               |
/// |@      %      %|
/// |               |
/// |               |
/// |               |
/// -----------------
///
/// If @ marks our birdie and % marks resource, then a FOV_RANGE of:
///
/// - 0.1 = 10% of the map = bird sees no resources (at least in this case)
/// - 0.5 = 50% of the map = bird sees one of the resources
/// - 1.0 = 100% of the map = bird sees both resources
const FOV_RANGE: f32 = 0.25;
/// Field of view depends on both FOV_RANGE and FOV_ANGLE:
///
/// - FOV_RANGE=0.4, FOV_ANGLE=PI/2:
///   -----------------
///   |       @       |
///   |     /.v.\     |
///   |   /.......\   |
///   |   ---------   |
///   |               |
///   |               |
///   |               |
///   -----------------
///
const FOV_ANGLE: f32 = PI + FRAC_PI_4;
/// How much photoreceptors there are in a single eye.
///
/// More cells means our birds will have more "crisp" vision, allowing
/// them to locate the resource more precisely - but the trade-off is that
/// the evolution process will then take longer, or even fail, unable
/// to find any solution.
///
const PHOTO_CELLS: usize = 9;

#[derive(Debug)]
pub struct Eye {
    fov_range: f32,
    fov_angle: f32,
    cells: usize,
}

impl Eye {
    // FOV_RANGE, FOV_ANGLE & CELLS are the values we'll use during
    // simulation - but being able to create an arbitrary eye will
    // come handy during the testing:
    fn new(fov_range: f32, fov_angle: f32, cells: usize) -> Self {
        assert!(fov_range > 0.0);
        assert!(fov_angle > 0.0);
        assert!(cells > 0);

        Self {
            fov_range,
            fov_angle,
            cells,
        }
    }

    pub fn cells(&self) -> usize {
        self.cells
    }

    /// Processes the vision of the creature, returning a vector representing the energy levels
    /// detected within the field of view (FOV) for each cell in the eye.
    ///
    /// # Arguments
    ///
    /// * `position` - The position of the creature in the   2D space.
    /// * `rotation` - The rotation of the creature, represented as a   2D rotation matrix.
    /// * `resources` - A slice of resources present in the environment, each with a position.
    ///
    /// # Returns
    ///
    /// * A vector of floating-point numbers, where each element corresponds to the energy level
    ///   detected by a cell in the eye. The length of the vector is equal to the number of cells
    ///   in the eye (`self.cells`).
    ///
    pub fn process_vision(
        &self,
        position: na::Point2<f32>,
        rotation: na::Rotation2<f32>,
        resources: &[Resource],
    ) -> Vec<f32> {
        let mut cells = vec![0.0; self.cells];

        for resource in resources {
            // creates a vector pointing from resource to creature
            let resource_vector = resource.position - position;

            let distance = resource_vector.norm();
            if distance >= self.fov_range {
                continue;
            }
            let angle =
                na::Rotation2::rotation_between(&na::Vector2::y(), &resource_vector).angle();
            let angle = angle - rotation.angle();
            let angle = na::wrap(angle, -PI, PI);
            if angle < -self.fov_angle / 2.0 || angle > self.fov_angle / 2.0 {
                continue;
            }
            let angle = angle + self.fov_angle / 2.0;

            let cell = angle / self.fov_angle;
            let cell = cell * (self.cells as f32);
            let cell = (cell as usize).min(cells.len() - 1);

            let energy = (self.fov_range - distance) / self.fov_range;

            cells[cell] += energy;
        }
        cells
    }
}

impl Default for Eye {
    fn default() -> Self {
        Self::new(FOV_RANGE, FOV_ANGLE, PHOTO_CELLS)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    // Helper function to create a Food instance
    fn food(x: f32, y: f32) -> Resource {
        Resource {
            position: na::Point2::new(x, y),
            // ... other fields
        }
    }

    // Struct to hold test parameters
    struct TestCase {
        foods: Vec<Resource>,
        fov_range: f32,
        fov_angle: f32,
        x: f32,
        y: f32,
        rot: f32,
        expected_vision: &'static str,
    }

    const TEST_EYE_CELLS: usize = 13;

    impl TestCase {
        fn run(self) {
            let eye = Eye::new(self.fov_range, self.fov_angle, TEST_EYE_CELLS);
            let actual_vision = eye.process_vision(
                na::Point2::new(self.x, self.y),
                na::Rotation2::new(self.rot),
                &self.foods,
            );

            let actual_vision: Vec<_> = actual_vision
                .into_iter()
                .map(|cell| {
                    if cell >= 0.7 {
                        "#"
                    } else if cell >= 0.3 {
                        "+"
                    } else if cell > 0.0 {
                        "."
                    } else {
                        " "
                    }
                })
                .collect();

            let actual_vision = actual_vision.join("");
            assert_eq!(actual_vision, self.expected_vision);
        }
    }

    mod different_fov_ranges {
        use super::*;
        use test_case::test_case;

        #[test_case(1.0, "      +      ")]
        #[test_case(0.9, "      +      ")]
        #[test_case(0.8, "      +      ")]
        #[test_case(0.7, "      .      ")]
        #[test_case(0.6, "      .      ")]
        #[test_case(0.5, "             ")]
        #[test_case(0.4, "             ")]
        #[test_case(0.3, "             ")]
        #[test_case(0.2, "             ")]
        #[test_case(0.1, "             ")]
        fn test(fov_range: f32, expected_vision: &'static str) {
            TestCase {
                foods: vec![food(0.5, 1.0)],
                fov_angle: FRAC_PI_2,
                x: 0.5,
                y: 0.5,
                rot: 0.0,
                fov_range,
                expected_vision,
            }
            .run()
        }
    }

    mod different_rotations {
        use super::*;
        use test_case::test_case;

        /// World:
        ///
        /// -------------
        /// |           |
        /// |           |
        /// |%    @     |
        /// |     v     |
        /// |           |
        /// -------------
        ///
        /// Test cases:
        ///
        /// -------------
        /// |...........|
        /// |...........|
        /// |%....@.....|
        /// |.....v.....|
        /// |...........|
        /// -------------
        ///
        /// -------------
        /// |...........|
        /// |...........|
        /// |%...<@.....|
        /// |...........|
        /// |...........|
        /// -------------
        ///
        /// -------------
        /// |...........|
        /// |.....^.....|
        /// |%....@.....|
        /// |...........|
        /// |...........|
        /// -------------
        ///
        /// -------------
        /// |...........|
        /// |...........|
        /// |%....@>....|
        /// |...........|
        /// |...........|
        /// -------------
        ///
        /// ... and so on, until we do a full circle, 360° rotation:
        #[test_case(0.00 * PI, "         +   ")] // Food is to our right
        #[test_case(0.25 * PI, "        +    ")]
        #[test_case(0.50 * PI, "      +      ")] // Food is in front of us
        #[test_case(0.75 * PI, "    +        ")]
        #[test_case(1.00 * PI, "   +         ")] // Food is to our left
        #[test_case(1.25 * PI, " +           ")]
        #[test_case(1.50 * PI, "            +")] // Food is behind us
        #[test_case(1.75 * PI, "           + ")] // (we continue to see it
        #[test_case(2.00 * PI, "         +   ")] // due to 360° fov_angle.)
        #[test_case(2.25 * PI, "        +    ")]
        #[test_case(2.50 * PI, "      +      ")]
        fn test(rot: f32, expected_vision: &'static str) {
            TestCase {
                foods: vec![food(0.0, 0.5)],
                fov_range: 1.0,
                fov_angle: 2.0 * PI,
                x: 0.5,
                y: 0.5,
                rot,
                expected_vision,
            }
            .run()
        }
    }
    mod different_positions {
        use super::*;
        use test_case::test_case;

        /// World:
        ///
        /// ------------
        /// |          |
        /// |         %|
        /// |          |
        /// |         %|
        /// |          |
        /// ------------
        ///
        /// Test cases for the X axis:
        ///
        /// ------------
        /// |          |
        /// |        /%|
        /// |       @>.|
        /// |        \%|
        /// |          |
        /// ------------
        ///
        /// ------------
        /// |        /.|
        /// |      /..%|
        /// |     @>...|
        /// |      \..%|
        /// |        \.|
        /// ------------
        ///
        /// ... and so on, going further left
        ///     (or, from the bird's point of view - going _back_)
        ///
        /// Test cases for the Y axis:
        ///
        /// ------------
        /// |     @>...|
        /// |       \.%|
        /// |        \.|
        /// |         %|
        /// |          |
        /// ------------
        ///
        /// ------------
        /// |      /...|
        /// |     @>..%|
        /// |      \...|
        /// |        \%|
        /// |          |
        /// ------------
        ///
        /// ... and so on, going further down
        ///     (or, from the bird's point of view - going _right_)

        // Checking the X axis:
        // (you can see the bird is "flying away" from the foods)
        #[test_case(0.9, 0.5, "#           #")]
        #[test_case(0.8, 0.5, "  #       #  ")]
        #[test_case(0.7, 0.5, "   +     +   ")]
        #[test_case(0.6, 0.5, "    +   +    ")]
        #[test_case(0.5, 0.5, "    +   +    ")]
        #[test_case(0.4, 0.5, "     + +     ")]
        #[test_case(0.3, 0.5, "     . .     ")]
        #[test_case(0.2, 0.5, "     . .     ")]
        #[test_case(0.1, 0.5, "     . .     ")]
        #[test_case(0.0, 0.5, "             ")]
        //
        // Checking the Y axis:
        // (you can see the bird is "flying alongside" the foods)
        #[test_case(0.5, 0.0, "            +")]
        #[test_case(0.5, 0.1, "          + .")]
        #[test_case(0.5, 0.2, "         +  +")]
        #[test_case(0.5, 0.3, "        + +  ")]
        #[test_case(0.5, 0.4, "      +  +   ")]
        #[test_case(0.5, 0.6, "   +  +      ")]
        #[test_case(0.5, 0.7, "  + +        ")]
        #[test_case(0.5, 0.8, "+  +         ")]
        #[test_case(0.5, 0.9, ". +          ")]
        #[test_case(0.5, 1.0, "+            ")]
        fn test(x: f32, y: f32, expected_vision: &'static str) {
            TestCase {
                foods: vec![food(1.0, 0.4), food(1.0, 0.6)],
                fov_range: 1.0,
                fov_angle: FRAC_PI_2,
                rot: 3.0 * FRAC_PI_2,
                x,
                y,
                expected_vision,
            }
            .run()
        }
    }
    mod different_fov_angles {
        use super::*;
        use test_case::test_case;

        /// World:
        ///
        /// ------------
        /// |%        %|
        /// |          |
        /// |%        %|
        /// |    @>    |
        /// |%        %|
        /// |          |
        /// |%        %|
        /// ------------
        ///
        /// Test cases:
        ///
        /// ------------
        /// |%        %|
        /// |         /|
        /// |%      /.%|
        /// |    @>....|
        /// |%      \.%|
        /// |         \|
        /// |%        %|
        /// ------------
        ///
        /// ------------
        /// |%      /.%|
        /// |      /...|
        /// |%    /...%|
        /// |    @>....|
        /// |%    \...%|
        /// |      \...|
        /// |%      \.%|
        /// ------------
        ///
        /// ------------
        /// |%........%|
        /// |\.........|
        /// |% \......%|
        /// |    @>....|
        /// |% /......%|
        /// |/.........|
        /// |%........%|
        /// ------------
        ///
        /// ... and so on, until we reach the full, 360° FOV
        #[test_case(0.25 * PI, " +         + ")] // FOV is narrow = 2 foods
        #[test_case(0.50 * PI, ".  +     +  .")]
        #[test_case(0.75 * PI, "  . +   + .  ")] // FOV gets progressively
        #[test_case(1.00 * PI, "   . + + .   ")] // wider and wider...
        #[test_case(1.25 * PI, "   . + + .   ")]
        #[test_case(1.50 * PI, ".   .+ +.   .")]
        #[test_case(1.75 * PI, ".   .+ +.   .")]
        #[test_case(2.00 * PI, "+.  .+ +.  .+")] // FOV is the widest = 8 foods
        fn test(fov_angle: f32, expected_vision: &'static str) {
            TestCase {
                foods: vec![
                    food(0.0, 0.0),
                    food(0.0, 0.33),
                    food(0.0, 0.66),
                    food(0.0, 1.0),
                    food(1.0, 0.0),
                    food(1.0, 0.33),
                    food(1.0, 0.66),
                    food(1.0, 1.0),
                ],
                fov_range: 1.0,
                x: 0.5,
                y: 0.5,
                rot: 3.0 * FRAC_PI_2,
                fov_angle,
                expected_vision,
            }
            .run()
        }
    }
}
