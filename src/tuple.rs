use approx::relative_eq;
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Tuple {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Tuple {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Tuple {
        Self { x, y, z, w }
    }

    pub fn point(x: f32, y: f32, z: f32) -> Tuple {
        Self { x, y, z, w: 1.0 }
    }

    pub fn vector(x: f32, y: f32, z: f32) -> Tuple {
        Self { x, y, z, w: 0.0 }
    }

    pub fn is_point(self) -> bool {
        relative_eq!(self.w, 1.0)
    }

    pub fn is_vector(self) -> bool {
        relative_eq!(self.w, 0.0)
    }

    pub fn magnitude(self) -> f32 {
        debug_assert!(self.is_vector());
        f32::sqrt(
            self.x * self.x
                + self.y * self.y
                + self.w * self.w
                + self.z * self.z,
        )
    }

    pub fn normalize(self) -> Tuple {
        debug_assert!(self.is_vector());
        self / self.magnitude()
    }

    pub fn dot(self, other: Tuple) -> f32 {
        debug_assert!(self.is_vector() && other.is_vector());
        self.x * other.x
            + self.y * other.y
            + self.z * other.z
            + self.w * other.w
    }

    pub fn cross(self, other: Tuple) -> Tuple {
        debug_assert!(self.is_vector() && other.is_vector());
        Tuple::vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }
}

impl Add for Tuple {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        debug_assert!(self.is_vector() || other.is_vector());
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }
}

impl AbsDiffEq for Tuple {
    type Epsilon = f32;

    fn default_epsilon() -> Self::Epsilon {
        f32::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.x.abs_diff_eq(&other.x, epsilon)
            && self.y.abs_diff_eq(&other.y, epsilon)
            && self.z.abs_diff_eq(&other.z, epsilon)
            && self.w.abs_diff_eq(&other.w, epsilon)
    }
}

impl RelativeEq for Tuple {
    fn default_max_relative() -> Self::Epsilon {
        f32::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.x.relative_eq(&other.x, epsilon, max_relative)
            && self.y.relative_eq(&other.y, epsilon, max_relative)
            && self.z.relative_eq(&other.z, epsilon, max_relative)
            && self.w.relative_eq(&other.w, epsilon, max_relative)
    }
}

impl UlpsEq for Tuple {
    fn default_max_ulps() -> u32 {
        f32::default_max_ulps()
    }

    fn ulps_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_ulps: u32,
    ) -> bool {
        self.x.ulps_eq(&other.x, epsilon, max_ulps)
            && self.y.ulps_eq(&other.y, epsilon, max_ulps)
            && self.z.ulps_eq(&other.z, epsilon, max_ulps)
            && self.w.ulps_eq(&other.w, epsilon, max_ulps)
    }
}

impl Sub for Tuple {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        debug_assert!(self.is_point() || other.is_vector());
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            w: self.w - other.w,
        }
    }
}

impl Neg for Tuple {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: -self.w,
        }
    }
}

impl Mul<f32> for Tuple {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
            w: self.w * rhs,
        }
    }
}

impl Mul<Tuple> for f32 {
    type Output = Tuple;

    fn mul(self, rhs: Tuple) -> Tuple {
        rhs * self
    }
}

impl Div<f32> for Tuple {
    type Output = Self;

    fn div(self, rhs: f32) -> Self {
        self * (1.0 / rhs)
    }
}

#[cfg(test)]
mod tests_tuple {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn tuple_is_a_point() {
        let a = Tuple::new(4.3, -4.2, 3.1, 1.0);
        assert_relative_eq!(a.x, 4.3);
        assert_relative_eq!(a.y, -4.2);
        assert_relative_eq!(a.z, 3.1);
        assert_relative_eq!(a.w, 1.0);
        assert!(a.is_point());
        assert!(!a.is_vector());
    }

    #[test]
    fn tuple_is_a_vector() {
        let a = Tuple::new(4.3, -4.2, 3.1, 0.0);
        assert_relative_eq!(a.x, 4.3);
        assert_relative_eq!(a.y, -4.2);
        assert_relative_eq!(a.z, 3.1);
        assert_relative_eq!(a.w, 0.0);
        assert!(!a.is_point());
        assert!(a.is_vector());
    }

    #[test]
    fn point_is_a_tuple() {
        let p = Tuple::point(4.0, -4.0, 3.0);
        assert_relative_eq!(p, Tuple::new(4.0, -4.0, 3.0, 1.0));
    }

    #[test]
    fn vector_is_a_tuple() {
        let p = Tuple::vector(4.0, -4.0, 3.0);
        assert_relative_eq!(p, Tuple::new(4.0, -4.0, 3.0, 0.0));
    }

    #[test]
    fn add_tuples() {
        let a1 = Tuple::new(3.0, -2.0, 5.0, 1.0);
        let a2 = Tuple::new(-2.0, 3.0, 1.0, 0.0);
        assert_relative_eq!(a1 + a2, Tuple::new(1.0, 1.0, 6.0, 1.0));
    }

    #[test]
    fn sub_points() {
        let p1 = Tuple::point(3.0, 2.0, 1.0);
        let p2 = Tuple::point(5.0, 6.0, 7.0);
        assert_relative_eq!(p1 - p2, Tuple::vector(-2.0, -4.0, -6.0));
    }

    #[test]
    fn sub_vector_from_point() {
        let p = Tuple::point(3.0, 2.0, 1.0);
        let v = Tuple::vector(5.0, 6.0, 7.0);
        assert_relative_eq!(p - v, Tuple::point(-2.0, -4.0, -6.0));
    }

    #[test]
    fn sub_vectors() {
        let v1 = Tuple::vector(3.0, 2.0, 1.0);
        let v2 = Tuple::vector(5.0, 6.0, 7.0);
        assert_relative_eq!(v1 - v2, Tuple::vector(-2.0, -4.0, -6.0));
    }

    #[test]
    fn sub_vector_from_zero() {
        let zero = Tuple::vector(0.0, 0.0, 0.0);
        let v = Tuple::vector(1.0, -2.0, 3.0);
        assert_relative_eq!(zero - v, Tuple::vector(-1.0, 2.0, -3.0));
    }

    #[test]
    fn neg_tuple() {
        let a = Tuple::new(1.0, -2.0, 3.0, -4.0);
        assert_relative_eq!(-a, Tuple::new(-1.0, 2.0, -3.0, 4.0));
    }

    #[test]
    fn mult_tuple_scalar() {
        let a = Tuple::new(1.0, -2.0, 3.0, -4.0);
        assert_relative_eq!(a * 3.5, Tuple::new(3.5, -7.0, 10.5, -14.0));
        assert_relative_eq!(a * 0.5, Tuple::new(0.5, -1.0, 1.5, -2.0));
    }

    #[test]
    fn div_tuple_scalar() {
        let a = Tuple::new(1.0, -2.0, 3.0, -4.0);
        assert_relative_eq!(a / 2.0, Tuple::new(0.5, -1.0, 1.5, -2.0));
    }

    #[test]
    fn vector_magnitude() {
        let v = Tuple::vector(1.0, 0.0, 0.0);
        assert_relative_eq!(v.magnitude(), 1.0);

        let v = Tuple::vector(0.0, 1.0, 0.0);
        assert_relative_eq!(v.magnitude(), 1.0);

        let v = Tuple::vector(0.0, 0.0, 1.0);
        assert_relative_eq!(v.magnitude(), 1.0);

        let v = Tuple::vector(1.0, 2.0, 3.0);
        assert_relative_eq!(v.magnitude(), f32::sqrt(14.0));

        let v = Tuple::vector(-1.0, -2.0, -3.0);
        assert_relative_eq!(v.magnitude(), f32::sqrt(14.0));
    }

    #[test]
    fn vector_normalize() {
        let v = Tuple::vector(4.0, 0.0, 0.0);
        assert_relative_eq!(v.normalize(), Tuple::vector(1.0, 0.0, 0.0));

        let v = Tuple::vector(1.0, 2.0, 3.0);
        assert_relative_eq!(
            v.normalize(),
            Tuple::vector(
                1.0 / f32::sqrt(14.0),
                2.0 / f32::sqrt(14.0),
                3.0 / f32::sqrt(14.0)
            )
        );

        let v = Tuple::vector(1.0, 2.0, 3.0);
        assert_relative_eq!(v.normalize().magnitude(), 1.0);
    }

    #[test]
    fn dot_vectors() {
        let a = Tuple::vector(1.0, 2.0, 3.0);
        let b = Tuple::vector(2.0, 3.0, 4.0);
        assert_relative_eq!(a.dot(b), 20.0);
    }

    #[test]
    fn cross_vectors() {
        let a = Tuple::vector(1.0, 2.0, 3.0);
        let b = Tuple::vector(2.0, 3.0, 4.0);
        assert_relative_eq!(a.cross(b), Tuple::vector(-1.0, 2.0, -1.0));
        assert_relative_eq!(b.cross(a), Tuple::vector(1.0, -2.0, 1.0));
    }
}
