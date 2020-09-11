extern crate fnv;
extern crate rayon;
extern crate cgmath;

use fnv::{FnvHashMap, FnvHashSet};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::ops::Deref;
use std::ops::DerefMut;
use std::fmt::{Display, Formatter, Result};
use cgmath::{Vector3, vec3};

type Vec3u8 = Vector3<u8>;

/// A labelling of a square in the half-grid.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Default)]
struct HalfGridSquare {
    labels: [u8; 9],
    // Samples are taken on vertices of a square of an eighth-grid
    // like so:
    //
    // @ = sample location
    //
    // Half-grid square
    // +---------------+
    // |               |
    // |   @       @   |
    // |               |
    // |               |
    // |               |
    // |   @       @   |
    // |               |
    // +---------------+
    //
    // For example (scaled up to avoid fractions),
    //
    // 000                 004
    //    +---------------+
    //    |               |
    //    |  011     013  |
    //    |               |
    //    |               |
    //    |               |
    //    |  013     033  |
    //    |               |
    //    +---------------+
    // 004                 044
    //
    // On a cube, there are a total of 16 x 6 = 96 samples,
    // and they get compressed to numbers from 0 to 96.
    //
    // TODO: Rotate sample positions by 45 degrees becase this can happen.
    //
    // +---------------+
    //   \             |
    //     @       @   |
    //       \         |
    //         \       |
    //       /   \     |
    //     @       @   |
    //   /           \ |
    // +---------------+
    samples: [u8; 16],
}

impl HalfGridSquare {
    /// All labellings of the half-grid on a cube
    const POSSIBILITIES: [u8; 26] = [
        0b00_00_00, 0b00_00_01, 0b00_00_11,
        0b00_01_00, 0b00_01_01, 0b00_01_11,
        0b00_11_00, 0b00_11_01, 0b00_11_11,

        0b01_00_00, 0b01_00_01, 0b01_00_11,
        0b01_01_00,             0b01_01_11,
        0b01_11_00, 0b01_11_01, 0b01_11_11,

        0b11_00_00, 0b11_00_01, 0b11_00_11,
        0b11_01_00, 0b11_01_01, 0b11_01_11,
        0b11_11_00, 0b11_11_01, 0b11_11_11,
    ];

    fn calc_samples(mut self) -> Self {
        for y in 0..=1 {
            for x in 0..=1 {
                let sub = [self.label(x, y), self.label(x + 1, y), self.label(x, y + 1), self.label(x + 1, y + 1)];
                let mut sub = {
                    let mut scaled = [vec3(0u8, 0u8, 0u8); 4];
                    for i in 0..4 {
                        scaled[i] = vec3(
                            (sub[i] & 0b11_00_00).count_ones() as u8 * 4,
                            (sub[i] & 0b00_11_00).count_ones() as u8 * 4,
                            (sub[i] & 0b00_00_11).count_ones() as u8 * 4,
                        )
                    }
                    scaled
                };

                // Don't sample using the fold diagonal
                let mid = if sub[0] == sub[3] {
                    (sub[1] + sub[2]) / 2
                } else {
                    (sub[0] + sub[3]) / 2
                };

                // Sample halfway to the middle
                for i in 0..4 {
                    sub[i] = (sub[i] + mid) / 2
                }

                // Compress
                let sub = {
                    let mut compressed = [0u8; 4];
                    for i in 0..4 {
                        // Get side index
                        let mut side = 0;
                        while sub[i].z % 2 == 1 {
                            sub[i] = sub[i].yzx();
                            side += 1;
                        }

                        if sub[i].z == 0 {
                            side += 3;
                        }

                        // Finish compression
                        compressed[i] = sub[i].x / 2 + sub[i].y / 2 * 4 + side * 16;
                    }
                    compressed
                };

                for i in 0..4 {
                    self.samples[(2 * y + i / 2) * 4 + 2 * x + i % 2] = sub[i];
                }
            }
        }
        self
    }

    fn new(labels: [u8; 9]) -> Self {
        Self { labels, samples: [0; 16] }.calc_samples()
    }

    fn label(self, x: usize, y: usize) -> u8 {
        self.labels[y * 3 + x]
    }

    fn all_with_sublabels(mut labels: [u8; 9], num_labels: usize) -> Vec<Self> {
        if num_labels == 9 {
            return vec![Self::new(labels)];
        }

        let x = num_labels % 3;
        let y = num_labels / 3;

        let labels_cloned = Self::new(labels);

        Self::POSSIBILITIES.iter().copied()
            .filter(|p| { 
                (x == 0 || (Self::label(labels_cloned, x - 1, y) ^ *p).is_power_of_two()) &&
                (y == 0 || (Self::label(labels_cloned, x, y - 1) ^ *p).is_power_of_two()) &&
                // No stretching sqrt(2)/2 to 1
                (x == 0 || y == 0 || {
                    let xor = Self::label(labels_cloned, x - 1, y - 1) ^ *p;
                    (xor & xor >> 1 & 0b01_01_01) == 0
                }) &&
                (x == 2 || y == 0 || {
                    let xor = Self::label(labels_cloned, x + 1, y - 1) ^ *p;
                    (xor & xor >> 1 & 0b01_01_01) == 0
                }) &&
                // No quarter grid folds. Points on opposite corners of a half grid square cannot both equal.
                (x == 0 || y == 0 || 
                    Self::label(labels_cloned, x - 1, y - 1) != *p ||
                    Self::label(labels_cloned, x, y - 1) != Self::label(labels_cloned, x - 1, y)
                )
            })
            .flat_map(|p| {
                labels[num_labels] = p;
                Self::all_with_sublabels(labels, num_labels + 1)
            })
            .collect()
    }

    fn all_possible() -> Vec<Self> {
        Self::all_with_sublabels([0u8; 9], 0)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
struct Filling([HalfGridSquare; HalfGridPoly::GRID_W * HalfGridPoly::GRID_H]);

impl Filling {
    fn at(self, grid: HalfGridPoly, x: usize, y: usize) -> Option<HalfGridSquare> {
        if grid.at(x, y) {
            Some(self[y * HalfGridPoly::GRID_W + x])
        } else {
            None
        }
    }
}

impl Deref for Filling {
    type Target = [HalfGridSquare; HalfGridPoly::GRID_W * HalfGridPoly::GRID_H];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Filling {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct HalfGridFilling(HalfGridPoly, Filling);

impl Display for HalfGridFilling {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let mut grid = [None; (2 * HalfGridPoly::GRID_W + 1) * (2 * HalfGridPoly::GRID_H + 1)];

        for y in 0..HalfGridPoly::GRID_H {
            for x in 0..HalfGridPoly::GRID_W {
                if let Some(square) = self.1.at(self.0, x, y) {
                    for dy in 0..=2 {
                        for dx in 0..=2 {
                            grid[(2 * y + dy) * (2 * HalfGridPoly::GRID_W + 1) + 2 * x + dx] = Some([
                                (square.label(dx, dy) & 0b11_00_00).count_ones(),
                                (square.label(dx, dy) & 0b00_11_00).count_ones(),
                                (square.label(dx, dy) & 0b00_00_11).count_ones(),
                            ]);
                        }
                    }
                }
            }
        };

        for y in 0..(2 * HalfGridPoly::GRID_H + 1) {
            for x in 0..(2 * HalfGridPoly::GRID_W + 1) {
                if let Some(label) = grid[y * (2 * HalfGridPoly::GRID_W + 1) + x] {
                    write!(f, "{}{}{} ", label[0], label[1], label[2])?;
                } else {
                    write!(f, "    ")?;
                }
            }
            write!(f, "\n")?;
        }

        Ok(())
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct HalfGridPoly {
    grid: [bool; Self::GRID_W * Self::GRID_H],
}

impl HalfGridPoly {
    const GRID_W: usize = 6;
    const GRID_H: usize = 5;

    /// Grid must contain padding of `false` around the edges
    fn new(grid: [bool; Self::GRID_W * Self::GRID_H]) -> Self {
        Self { grid }
    }

    fn at(self, x: usize, y: usize) -> bool {
        self.grid[y * Self::GRID_W + x]
    }

    // Constraint indexes
    // 0: . . .  1: 0 . .  2: . . 2  3: 0 . 2
    //    . . .     . . .     . . .     . . .
    //    . . .     . . .     . . .     . . .
    //
    // 4: 0 . .  5: 0 1 2  6: 0 . 2  7: 0 1 2
    //    3 . .     . . .     3 . .     3 . .
    //    6 . .     . . .     6 . .     6 . .
    fn fillings(self) -> Vec<Filling> {
        let possible = HalfGridSquare::all_possible();

        let mut constraint_map = <[FnvHashMap<[u8; 8], Vec<HalfGridSquare>>; 8]>::default();

        fn build_constraints(
            possible: &[HalfGridSquare], 
            constraint_fn: impl Fn(HalfGridSquare) -> [u8; 8])
        -> FnvHashMap<[u8; 8], Vec<HalfGridSquare>> {

            let mut map = FnvHashMap::default();
            for p in possible.iter() {
                map.entry(constraint_fn(*p)).or_insert(vec![]).push(*p);
            }
            map
        }

        constraint_map[0].insert([0; 8], possible.clone());
        // Symmetry reduction hack
        constraint_map[0].insert([1; 8], possible.iter().filter(|p| {
            (p.labels[0] == 0b00_00_00 && p.labels[1] == 0b00_00_01) ||
            (p.labels[0] == 0b00_00_01 && (p.labels[1] == 0b00_00_00 || p.labels[1] == 0b00_01_01)) ||
            (p.labels[0] == 0b00_01_01 && p.labels[1] == 0b00_00_01)
        }).copied().collect());

        constraint_map[1] = build_constraints(
            &possible, 
            |p| [p.labels[0], 0, 0, 0, 0, 0, 0, 0]
        );

        constraint_map[2] = build_constraints(
            &possible, 
            |p| [0, 0, p.labels[2], 0, 0, 0, 0, 0]
        );

        constraint_map[3] = build_constraints(
            &possible, 
            |p| [p.labels[0], 0, p.labels[2], 0, 0, 0, 0, 0]
        );

        constraint_map[4] = build_constraints(
            &possible, 
            |p| [p.labels[0], 0, 0, p.labels[3], 0, 0, p.labels[6], 0]
        );

        constraint_map[5] = build_constraints(
            &possible, 
            |p| [p.labels[0], p.labels[1], p.labels[2], 0, 0, 0, 0, 0]
        );

        constraint_map[6] = build_constraints(
            &possible, 
            |p| [p.labels[0], 0, p.labels[2], p.labels[3], 0, 0, p.labels[6], 0]
        );

        constraint_map[7] = build_constraints(
            &possible, 
            |p| [p.labels[0], p.labels[1], p.labels[2], p.labels[3], 0, 0, p.labels[6], 0]
        );

        let on_squares = self.grid.iter().enumerate()
            .filter(|(_, v)| **v)
            .map(|(k, _)| k).collect::<Vec<_>>();

        let mut filling = Filling::default();

        let num_sample_points = on_squares.len() * 16;

        // Each possible label gets a bit in here
        let mentioned = 0u128;

        self.fillings_helper(&filling, &on_squares, 0, &constraint_map, num_sample_points, mentioned)
    }

    fn fillings_helper(
        self,
        filling: &Filling,
        on_squares: &[usize],
        curr_on_square_index: usize,
        constraint_map: &[FnvHashMap<[u8; 8], Vec<HalfGridSquare>>; 8],
        mut num_sample_points: usize,
        mentioned: u128,
    ) -> Vec<Filling> {
        if curr_on_square_index >= on_squares.len() {
            return vec![*filling];
        }

        let index = on_squares[curr_on_square_index];
        
        let up = filling[index - Self::GRID_W];
        let left = filling[index - 1];
        let up_left = filling[index - Self::GRID_W - 1];
        let up_right = filling[index - Self::GRID_W + 1];
        let empty = vec![];

        let possible = if self.grid[index - 1] {
            // Cell on left
            if self.grid[index - Self::GRID_W] {
                // Cell above
                // Due to multi-edge interaction, this combination may be not allowed at all.
                &constraint_map[7].get(&[up.labels[6], up.labels[7], up.labels[8], left.labels[5], 0, 0, left.labels[8], 0]).unwrap_or(&empty)
            } else if self.grid[index - Self::GRID_W + 1] {
                // Cell above right
                // Due to multi-edge interaction, this combination may be not allowed at all.
                &constraint_map[6].get(&[left.labels[2], 0, up_right.labels[6], left.labels[5], 0, 0, left.labels[8], 0]).unwrap_or(&empty)
            } else {
                &constraint_map[4][&[left.labels[2], 0, 0, left.labels[5], 0, 0, left.labels[8], 0]]
            }
        } else if self.grid[index - Self::GRID_W] {
            // Cell above
            &constraint_map[5][&[up.labels[6], up.labels[7], up.labels[8], 0, 0, 0, 0, 0]]
        } else if self.grid[index - Self::GRID_W + 1] {
            // Cell above right
            if self.grid[index - Self::GRID_W - 1] {
                // Cell above left
                &constraint_map[3][&[up_left.labels[8], 0, up_right.labels[6], 0, 0, 0, 0, 0]]
            } else {
                &constraint_map[2][&[0, 0, up_right.labels[6], 0, 0, 0, 0, 0]]
            }
        } else if self.grid[index - Self::GRID_W - 1] {
            // Cell above left
            &constraint_map[1][&[up_left.labels[8], 0, 0, 0, 0, 0, 0, 0]]
        } else {
            // Symmetry hack used here
            &constraint_map[0][if curr_on_square_index == 0 {&[1; 8]} else {&[0; 8]}]
        };

        num_sample_points -= 16;

        if curr_on_square_index == 0 {
            let mut count = AtomicUsize::new(0);
            possible.par_iter().flat_map(|p| {
                let new_count = count.fetch_add(1, Ordering::SeqCst);
                eprintln!("{}/{}", new_count, possible.len());
                let mut mentioned = mentioned;

                for sample in p.samples.iter() {
                    mentioned |= 1 << *sample;
                }
                
                // Check if not enough space left to mention all samples
                if num_sample_points < 96 - mentioned.count_ones() as usize {
                    return vec![]
                }

                let mut filling = *filling;
                filling[index] = *p;
                self.fillings_helper(&filling, on_squares, curr_on_square_index + 1, constraint_map, num_sample_points, mentioned)
            }).collect()
        } else {
            possible.iter().flat_map(|p| {
                let mut mentioned = mentioned;

                for sample in p.samples.iter() {
                    mentioned |= 1 << *sample;
                }
                
                // Check if not enough space left to mention all labels
                if num_sample_points < 96 - mentioned.count_ones() as usize {
                    return vec![]
                }

                let mut filling = *filling;
                filling[index] = *p;
                self.fillings_helper(&filling, on_squares, curr_on_square_index + 1, constraint_map, num_sample_points, mentioned)
            }).collect()
        }
    }
}

fn main() {
    //let num = HalfGridSquare::all_possible().into_iter()
    //    .filter(|s| s.labels[0] == 0b00_00_00)
    //    .collect::<Vec<_>>()
    //    .len();

    //println!("Possibility count: {:?}", num);

    //let square = dbg!(HalfGridSquare::new([
    //    0b_00_00_11, 0b_00_01_11, 0b_00_00_11,
    //    0b_00_01_11, 0b_01_01_11, 0b_00_01_11,
    //    0b_00_00_11, 0b_00_01_11, 0b_00_11_11,
    //]));

    let grid = HalfGridPoly::new([
        false, false, false, false, false, false,
        false, false, true , false, false, false,
        false, true , true , true , true , false,
        false, false, true , false, false, false,
        false, false, false, false, false, false,
    ]);

    let f = grid.fillings();
    for f in f {
        println!("Filling: {}", HalfGridFilling(grid, f));
    }
}
