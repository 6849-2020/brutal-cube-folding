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
use std::io::{self, Read};
use std::iter;

type Vec3u8 = Vector3<u8>;

/// A labelling of a square in the half-grid.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Default)]
struct HalfGridSquare {
    labels: [u8; 9],
    // Samples are taken like this:
    //
    // @ = sample location
    //
    // Half-grid square
    // (Slashes delimit areas that must stay flat under the half-grid model)
    // +---------------+
    // | \           / |
    // |   \   @   /   |
    // |     \   /     |
    // |   @   x   @   |
    // |     /   \     |
    // |   /   @   \   |
    // | /           \ |
    // +---------------+
    //
    // For example (scaled up to avoid fractions),
    //
    // 000                 004
    //    +---------------+
    //    |               |
    //    |      012      |
    //    |               |
    //    |  012     023  |
    //    |               |
    //    |      023      |
    //    |               |
    //    +---------------+
    // 004                 044
    //
    // On a cube, there are a total of 16 x 6 = 96 samples,
    // and they get compressed to numbers from 0 to 96.
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

    /// Calculates the samples on this square.
    fn calc_samples(mut self) -> Self {
        for y in 0..=1 {
            for x in 0..=1 {
                let sub = [self.label(x, y), self.label(x + 1, y), self.label(x, y + 1), self.label(x + 1, y + 1)];
                let sub = {
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

                // Don't sample the middle along a folded diagonal
                let mid = if sub[0] == sub[3] {
                    (sub[1] + sub[2]) / 2
                } else {
                    (sub[0] + sub[3]) / 2
                };

                // Sample in a triangle forced to be flat
                let mut sub = {
                    let mut new_sub = sub;
                    for i in 0..4 {
                        new_sub[i] = (sub[i] + sub[[1, 3, 0, 2][i]] + mid * 2) / 4;
                    }
                    new_sub
                };

                // Now one coordinate is 0 or 8,
                //     one coordinate is 2 or 6,
                // and one coordinate is 1, 3, 5, or 7

                // Compress
                let sub = {
                    let mut compressed = [0u8; 4];
                    for i in 0..4 {
                        while sub[i].z % 8 != 0 {
                            sub[i] = sub[i].yzx();
                            compressed[i] += 16;
                        }

                        if sub[i].z == 0 {
                            compressed[i] += 48;
                        }

                        if sub[i].y % 2 == 1 {
                            sub[i] = sub[i].yxz();
                            compressed[i] += 8;
                        }

                        // Finish compression
                        compressed[i] += sub[i].x / 2 + sub[i].y / 4 * 4;
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
        Self { labels, samples: [0; 16] }
    }

    fn label(self, x: usize, y: usize) -> u8 {
        self.labels[y * 3 + x]
    }

    fn all_with_sublabels(mut labels: [u8; 9], num_labels: usize) -> Vec<Self> {
        if num_labels == 9 {
            return vec![Self::new(labels).calc_samples()];
        }

        let x = num_labels % 3;
        let y = num_labels / 3;

        let labels_cloned = Self::new(labels);

        Self::POSSIBILITIES.iter().copied()
            .filter(|p| { 
                (x == 0 || (Self::label(labels_cloned, x - 1, y) ^ *p).is_power_of_two()) &&
                (y == 0 || (Self::label(labels_cloned, x, y - 1) ^ *p).is_power_of_two()) &&
                // No stretching sqrt(2) to 2
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
                ) &&
                // Cannot fold from 111
                (x == 0 || y == 0 || {
                    let tl = Self::label(labels_cloned, x - 1, y - 1);
                    let tr = Self::label(labels_cloned, x, y - 1);
                    let bl = Self::label(labels_cloned, x - 1, y);
                    let br = *p;

                    (tl != br || {
                        // See if unfolding the fold would result in a 111
                        let xor = tr ^ bl ^ tl;
                        ((xor ^ xor >> 1) & 0b01_01_01) != 0b01_01_01
                    }) &&
                    (tr != bl || {
                        let xor = tl ^ br ^ tr;
                        ((xor ^ xor >> 1) & 0b01_01_01) != 0b01_01_01
                    })
                })
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

/// A generic grid of size unknown at compile time
#[derive(Clone, Debug, Eq, PartialEq)]
struct Grid<T> {
    grid: Vec<T>,
    width: usize,
    height: usize,
}

impl<T> Grid<T> {
    fn new(grid: Vec<T>, width: usize, height: usize) -> Self {
        Self { grid, width, height }
    }

    fn at(&self, x: usize, y: usize) -> &T {
        &self.grid[y * self.width + x]
    }

    fn at_mut(&mut self, x: usize, y: usize) -> &mut T {
        &mut self.grid[y * self.width + x]
    }

    fn default_with_size(width: usize, height: usize) -> Self where T: Default + Clone {
        Self::new(vec![T::default(); width * height], width, height)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Filling(Grid<HalfGridSquare>);

impl Filling {
    fn at(&self, grid: &HalfGridPoly, x: usize, y: usize) -> Option<HalfGridSquare> {
        if grid.at(x, y) {
            Some(*self.0.at(x, y))
        } else {
            None
        }
    }

    fn default_with_size(width: usize, height: usize) -> Self {
        Filling(Grid::default_with_size(width, height))
    }
}

impl Deref for Filling {
    type Target = Grid<HalfGridSquare>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Filling {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct HalfGridFilling(HalfGridPoly, Filling);

impl Display for HalfGridFilling {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let mut grid = vec![None; (2 * self.1.width + 1) * (2 * self.1.height + 1)];

        for y in 0..self.1.height {
            for x in 0..self.1.width {
                if let Some(square) = self.1.at(&self.0, x, y) {
                    for dy in 0..=2 {
                        for dx in 0..=2 {
                            grid[(2 * y + dy) * (2 * self.1.width + 1) + 2 * x + dx] = Some([
                                (square.label(dx, dy) & 0b11_00_00).count_ones(),
                                (square.label(dx, dy) & 0b00_11_00).count_ones(),
                                (square.label(dx, dy) & 0b00_00_11).count_ones(),
                            ]);
                        }
                    }
                }
            }
        };

        for y in 0..(2 * self.1.height + 1) {
            for x in 0..(2 * self.1.width + 1) {
                if let Some(label) = grid[y * (2 * self.1.width + 1) + x] {
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

#[derive(Clone, Debug, Eq, PartialEq)]
struct HalfGridPoly {
    grid: Grid<bool>,
}

impl HalfGridPoly {
    /// Grid must contain padding of `false` around the edges
    fn new(grid: Grid<bool>) -> Self {
        Self { grid }
    }

    fn at(&self, x: usize, y: usize) -> bool {
        *self.grid.at(x, y)
    }

    // Constraint indexes
    // 0: . . .  1: 0 . .  2: . . 2  3: 0 . 2
    //    . . .     . . .     . . .     . . .
    //    . . .     . . .     . . .     . . .
    //
    // 4: 0 . .  5: 0 1 2  6: 0 . 2  7: 0 1 2
    //    3 . .     . . .     3 . .     3 . .
    //    6 . .     . . .     6 . .     6 . .
    fn fillings(&self) -> Vec<Filling> {
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

        let on_squares = self.grid.grid.iter().enumerate()
            .filter(|(_, v)| **v)
            .map(|(k, _)| (k % self.grid.width, k / self.grid.width)).collect::<Vec<_>>();

        let filling = Filling::default_with_size(self.grid.width, self.grid.height);

        let num_sample_points = on_squares.len() * 16;

        // Each possible label gets a bit in here
        let mentioned = 0u128;

        self.fillings_helper(&filling, &on_squares, 0, &constraint_map, num_sample_points, mentioned)
    }

    fn fillings_helper(
        &self,
        filling: &Filling,
        on_squares: &[(usize, usize)],
        curr_on_square_index: usize,
        constraint_map: &[FnvHashMap<[u8; 8], Vec<HalfGridSquare>>; 8],
        mut num_sample_points: usize,
        mentioned: u128,
    ) -> Vec<Filling> {
        if curr_on_square_index >= on_squares.len() {
            return vec![filling.clone()];
        }

        let (x, y) = on_squares[curr_on_square_index];
        
        let up = filling.0.at(x, y - 1);
        let left = filling.0.at(x - 1, y);
        let up_left = filling.0.at(x - 1, y - 1);
        let up_right = filling.0.at(x + 1, y - 1);
        let empty = vec![];

        let possible = if *self.grid.at(x - 1, y) {
            // Cell on left
            if *self.grid.at(x, y - 1) {
                // Cell above
                // Due to multi-edge interaction, this combination may be not allowed at all.
                &constraint_map[7].get(&[up.labels[6], up.labels[7], up.labels[8], left.labels[5], 0, 0, left.labels[8], 0]).unwrap_or(&empty)
            } else if *self.grid.at(x + 1, y - 1) {
                // Cell above right
                // Due to multi-edge interaction, this combination may be not allowed at all.
                &constraint_map[6].get(&[left.labels[2], 0, up_right.labels[6], left.labels[5], 0, 0, left.labels[8], 0]).unwrap_or(&empty)
            } else {
                &constraint_map[4][&[left.labels[2], 0, 0, left.labels[5], 0, 0, left.labels[8], 0]]
            }
        } else if *self.grid.at(x, y - 1) {
            // Cell above
            &constraint_map[5][&[up.labels[6], up.labels[7], up.labels[8], 0, 0, 0, 0, 0]]
        } else if *self.grid.at(x + 1, y - 1) {
            // Cell above right
            if *self.grid.at(x - 1, y - 1) {
                // Cell above left
                &constraint_map[3][&[up_left.labels[8], 0, up_right.labels[6], 0, 0, 0, 0, 0]]
            } else {
                &constraint_map[2][&[0, 0, up_right.labels[6], 0, 0, 0, 0, 0]]
            }
        } else if *self.grid.at(x - 1, y - 1) {
            // Cell above left
            &constraint_map[1][&[up_left.labels[8], 0, 0, 0, 0, 0, 0, 0]]
        } else {
            // Symmetry hack used here
            &constraint_map[0][if curr_on_square_index == 0 {&[1; 8]} else {&[0; 8]}]
        };

        num_sample_points -= 16;

        if curr_on_square_index == 0 {
            let count = AtomicUsize::new(0);
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

                let mut filling = filling.clone();
                *filling.0.at_mut(x, y) = *p;
                self.fillings_helper(&filling, on_squares, curr_on_square_index + 1, constraint_map, num_sample_points, mentioned)
            }).collect()
        } else {
            // This is done in series, so modifying is fine
            let mut filling = filling.clone();

            possible.iter().flat_map(|p| {
                let mut mentioned = mentioned;

                for sample in p.samples.iter() {
                    mentioned |= 1 << *sample;
                }
                
                // Check if not enough space left to mention all labels
                if num_sample_points < 96 - mentioned.count_ones() as usize {
                    return vec![]
                }

                *filling.0.at_mut(x, y) = *p;
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

    //let square = HalfGridSquare::new([
    //    0b_00_00_11, 0b_00_01_11, 0b_00_00_11,
    //    0b_00_01_11, 0b_01_01_11, 0b_00_01_11,
    //    0b_00_00_11, 0b_00_01_11, 0b_00_11_11,
    //]);
    //println!("Square: {:?}", square);
    
    println!("Enter a polyomino. x = square, space = blank. For example\nxxx\nx x\nxxx\n\nEnter twice to finish.");

    // Add padding around the grid
    let mut grid = vec![vec![]];
    while {
        let mut string = String::new();
        io::stdin().read_line(&mut string).expect("Failed to read polyomino");
        let string = string.trim_end();
        grid.push(string.chars().map(|c| c == 'x').collect::<Vec<_>>());
        !string.is_empty()
    } {}
    grid.push(vec![]);
    
    let width = grid.iter().map(|v| v.len()).max().unwrap() + 2;
    // Padding already added
    let height = grid.len();

    let grid = HalfGridPoly::new(Grid::new(
        grid.into_iter().flat_map(|v| {
            let len = v.len();
            iter::once(false).chain(v.into_iter()).chain(iter::repeat(false).take(width - len - 1))
        }).collect(),
        width,
        height
    ));

    dbg!(&grid);

    let f = grid.fillings();
    for f in f {
        println!("Filling: {}", HalfGridFilling(grid.clone(), f));
    }
}
