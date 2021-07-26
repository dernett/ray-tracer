use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use std::ops::{Add, Mul, Sub};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Color {
    pub red: f32,
    pub green: f32,
    pub blue: f32,
}

impl Color {
    pub fn new(red: f32, green: f32, blue: f32) -> Color {
        Self { red, green, blue }
    }

    pub fn black() -> Color {
        Self {
            red: 0.0,
            green: 0.0,
            blue: 0.0,
        }
    }

    pub fn byte_triplet(&self) -> (u8, u8, u8) {
        let r = f32::round(self.red * 255.0).clamp(0.0, 255.0);
        let g = f32::round(self.green * 255.0).clamp(0.0, 255.0);
        let b = f32::round(self.blue * 255.0).clamp(0.0, 255.0);
        (r as u8, g as u8, b as u8)
    }
}

impl AbsDiffEq for Color {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        f32::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.red.abs_diff_eq(&other.red, epsilon)
            && self.green.abs_diff_eq(&other.green, epsilon)
            && self.blue.abs_diff_eq(&other.blue, epsilon)
    }
}

impl RelativeEq for Color {
    fn default_max_relative() -> Self::Epsilon {
        f32::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.red.relative_eq(&other.red, epsilon, max_relative)
            && self.green.relative_eq(&other.green, epsilon, max_relative)
            && self.blue.relative_eq(&other.blue, epsilon, max_relative)
    }
}

impl UlpsEq for Color {
    fn default_max_ulps() -> u32 {
        f32::default_max_ulps()
    }

    fn ulps_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_ulps: u32,
    ) -> bool {
        self.red.ulps_eq(&other.red, epsilon, max_ulps)
            && self.green.ulps_eq(&other.green, epsilon, max_ulps)
            && self.blue.ulps_eq(&other.blue, epsilon, max_ulps)
    }
}

impl Add for Color {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            red: self.red + other.red,
            green: self.green + other.green,
            blue: self.blue + other.blue,
        }
    }
}

impl Sub for Color {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            red: self.red - other.red,
            green: self.green - other.green,
            blue: self.blue - other.blue,
        }
    }
}

impl Mul<f32> for Color {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self {
        Self {
            red: self.red * rhs,
            green: self.green * rhs,
            blue: self.blue * rhs,
        }
    }
}

impl Mul<Color> for f32 {
    type Output = Color;

    fn mul(self, rhs: Color) -> Color {
        rhs * self
    }
}

impl Mul for Color {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self {
            red: self.red * rhs.red,
            green: self.green * rhs.green,
            blue: self.blue * rhs.blue,
        }
    }
}

#[cfg(test)]
mod tests_color {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn colors_as_tuples() {
        let c = Color::new(-0.5, 0.4, 1.7);
        assert_relative_eq!(c.red, -0.5);
        assert_relative_eq!(c.green, 0.4);
        assert_relative_eq!(c.blue, 1.7);
    }

    #[test]
    fn add_colors() {
        let c1 = Color::new(0.9, 0.6, 0.75);
        let c2 = Color::new(0.7, 0.1, 0.25);
        assert_relative_eq!(c1 + c2, Color::new(1.6, 0.7, 1.0));
    }

    #[test]
    fn sub_colors() {
        let c1 = Color::new(0.9, 0.6, 0.75);
        let c2 = Color::new(0.7, 0.1, 0.25);
        assert_relative_eq!(c1 - c2, Color::new(0.2, 0.5, 0.5));
    }

    #[test]
    fn mult_color_scalar() {
        let c1 = Color::new(0.2, 0.3, 0.4);
        assert_relative_eq!(c1 * 2.0, Color::new(0.4, 0.6, 0.8));
    }

    #[test]
    fn mult_colors() {
        let c1 = Color::new(1.0, 0.2, 0.4);
        let c2 = Color::new(0.9, 1.0, 0.1);
        assert_relative_eq!(c1 * c2, Color::new(0.9, 0.2, 0.04));
    }
}
