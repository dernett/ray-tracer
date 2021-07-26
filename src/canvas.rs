use crate::color::Color;
use std::ops::{Index, IndexMut};

pub struct Canvas {
    width: usize,
    height: usize,
    pixels: Vec<Vec<Color>>,
}

impl Canvas {
    pub fn new(width: usize, height: usize) -> Self {
        Canvas {
            width,
            height,
            pixels: vec![vec![Color::black(); width]; height],
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn to_ppm(&self) -> String {
        let mut ppm = format!("P3\n{} {}\n255\n", self.width, self.height);
        for pixel in self.pixels.iter().flatten() {
            let (r, g, b) = pixel.byte_triplet();
            ppm += &format!("{} {} {}\n", r, g, b);
        }
        ppm
    }
}

impl Index<usize> for Canvas {
    type Output = Vec<Color>;

    fn index(&self, row: usize) -> &Vec<Color> {
        &self.pixels[row]
    }
}

impl IndexMut<usize> for Canvas {
    fn index_mut(&mut self, row: usize) -> &mut Vec<Color> {
        &mut self.pixels[row]
    }
}

#[cfg(test)]
mod tests_canvas {
    use super::*;

    #[test]
    fn ppm_header() {
        let canvas = Canvas::new(5, 3);
        let ppm = canvas.to_ppm();
        let mut lines = ppm.split("\n");
        assert_eq!(lines.next(), Some("P3"));
        assert_eq!(lines.next(), Some("5 3"));
        assert_eq!(lines.next(), Some("255"));
    }

    #[test]
    fn ppm_data() {
        let mut canvas = Canvas::new(2, 2);
        canvas[0][1] = Color::new(1.5, 0.0, 0.0);
        canvas[1][0] = Color::new(0.0, 0.5, 0.0);
        canvas[1][1] = Color::new(-0.5, 0.0, 1.0);
        assert_eq!(
            canvas.to_ppm(),
            "P3\n2 2\n255\n0 0 0\n255 0 0\n0 128 0\n0 0 255\n"
        );
    }

    #[test]
    fn ppm_newline() {
        let canvas = Canvas::new(5, 3);
        let ppm = canvas.to_ppm();
        assert_eq!(ppm.chars().last(), Some('\n'));
    }
}
